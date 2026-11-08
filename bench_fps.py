"""
Streaming throughput benchmark for Causal Forcing++ frame-wise models.

Mirrors CausalInferencePipeline.inference block by block (denoise -> KV-cache
refresh), but decodes each latent frame as soon as it exists, the way a live
stream would, and times every block separately.

Each frame-wise block is 1 latent frame = 4 pixel frames (1 for the first block).
Wan 2.1 plays at 16 fps, so real time (RTF <= 1) means every block, including
decode, must finish in <= 250 ms.

Run from anywhere:  python bench_fps.py --steps 2 --decoder taehv
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Causal-Forcing")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_out")
os.chdir(REPO)  # the repo loads weights through relative paths
sys.path.insert(0, REPO)

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.taehv import TAEHV

PLAYBACK_FPS = 16
MODELS = {
    1: ("configs/causal_forcing_dmd_framewise_1step.yaml", "checkpoints/causal-forcing++/framewise-1step.pt"),
    2: ("configs/causal_forcing_dmd_framewise_2step.yaml", "checkpoints/causal-forcing++/framewise-2step.pt"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, choices=sorted(MODELS), default=2)
parser.add_argument("--decoder", choices=["wan", "taehv", "none"], default="taehv")
parser.add_argument("--compile", action="store_true", help="torch.compile the transformer")
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--fp8", action="store_true", help="torchao FP8 dynamic quantization of linear layers")
parser.add_argument("--latent_frames", type=int, default=21, help="21 latent frames = 81 pixel frames (native max)")
parser.add_argument("--runs", type=int, default=2, help="timed runs after one warm-up run")
parser.add_argument("--prompt", default="A first-person drone flight over a misty pine forest at sunrise, "
                                        "the camera gliding steadily forward between the trees, cinematic lighting.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save", action="store_true", help="write the last timed run to bench_out/*.mp4")
args = parser.parse_args()
assert 1 <= args.latent_frames <= 21, "frame-wise models are trained for at most 21 latent frames (81 pixel frames)"

device = torch.device("cuda")
torch.set_grad_enabled(False)
torch._dynamo.config.cache_size_limit = 64  # KV length changes every block

# ---------------------------------------------------------------- load models
config_path, ckpt_path = MODELS[args.steps]
config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(config_path))
pipeline = CausalInferencePipeline(config, device=device)

state_dict = torch.load(ckpt_path, map_location="cpu")
key = "generator_ema" if "generator_ema" in state_dict else "generator"
gen_sd = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in state_dict[key].items()}
missing, unexpected = pipeline.generator.load_state_dict(gen_sd, strict=False)
assert not missing, f"checkpoint is missing {len(missing)} keys, e.g. {missing[:3]}"
print(f"Loaded {ckpt_path} [{key}] ({len(unexpected)} unexpected keys ignored)")
del state_dict, gen_sd

pipeline = pipeline.to(dtype=torch.bfloat16)
generator, text_encoder = pipeline.generator, pipeline.text_encoder

# Encode the prompt on the GPU, time a warm encode (= cost of a live prompt
# switch), then park the 11 GB text encoder on the CPU for the benchmark.
text_encoder.to(device)
conditional_dict = text_encoder(text_prompts=[args.prompt])
torch.cuda.synchronize()
t = time.perf_counter()
conditional_dict = text_encoder(text_prompts=[args.prompt])
torch.cuda.synchronize()
prompt_encode_ms = (time.perf_counter() - t) * 1000
text_encoder.to("cpu")
torch.cuda.empty_cache()

generator.to(device)
if args.fp8:
    from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
    quantize_(generator, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
if args.compile:
    generator.compile(mode=args.compile_mode)

if args.decoder == "wan":
    decoder = VAEDecoderWrapper()
    vae_sd = torch.load("wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", map_location="cpu")
    decoder.load_state_dict({k: v for k, v in vae_sd.items() if "decoder." in k or "conv2" in k})
    decoder = decoder.eval().requires_grad_(False).to(device=device, dtype=torch.float16)
elif args.decoder == "taehv":
    decoder = TAEHV(checkpoint_path="checkpoints/taew2_1.pth").eval().requires_grad_(False).to(device, torch.float16)


class StreamDecoder:
    """Turns one latent frame at a time into displayable uint8 frames on the CPU."""

    TAEHV_CONTEXT = 3  # latent frames of history re-decoded for temporal continuity

    def __init__(self):
        self.first = True
        self.wan_cache = [c.to(device=device, dtype=torch.float16) for c in ZERO_VAE_CACHE]
        self.taehv_history = None

    def __call__(self, latent):  # latent: [1, 1, 16, 60, 104]
        if args.decoder == "none":
            return None
        if args.decoder == "wan":
            pixels, self.wan_cache = decoder(latent.half(), *self.wan_cache)  # [-1, 1], 4 frames
            pixels = pixels * 0.5 + 0.5
        else:
            window = latent if self.taehv_history is None else torch.cat([self.taehv_history, latent], dim=1)
            self.taehv_history = window[:, -self.TAEHV_CONTEXT:]
            pixels = decoder.decode_video(window.half(), parallel=True)[:, -4:]  # [0, 1]
        if self.first:  # the first latent frame decodes to a single real frame
            pixels = pixels[:, 3:]
            self.first = False
        return (pixels[0].clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()  # T H W C


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def generate(seed):
    """One 81-frame stream. Returns per-block timings (ms) and the decoded frames."""
    rng = torch.Generator(device).manual_seed(seed)
    noise = torch.randn([1, args.latent_frames, 16, 60, 104], device=device, dtype=torch.bfloat16, generator=rng)
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    stream = StreamDecoder()
    blocks, frames = [], []

    for i in range(args.latent_frames):
        start = i * pipeline.frame_seq_length
        steps = pipeline.denoising_step_list
        if i == 0 and pipeline.denoising_step_list_first_chunk is not None:
            steps = pipeline.denoising_step_list_first_chunk

        t0 = sync_time()
        noisy = noise[:, i:i + 1]
        for j, current_timestep in enumerate(steps):
            timestep = torch.ones([1, 1], device=device, dtype=torch.int64) * current_timestep
            _, denoised = generator(noisy_image_or_video=noisy, conditional_dict=conditional_dict, timestep=timestep,
                                    kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                                    current_start=start)
            if j < len(steps) - 1:
                noisy = pipeline.scheduler.add_noise(
                    denoised.flatten(0, 1), torch.randn_like(denoised.flatten(0, 1)),
                    steps[j + 1] * torch.ones([1], device=device, dtype=torch.long),
                ).unflatten(0, denoised.shape[:2])
        t1 = sync_time()

        # Re-run on the clean frame so later frames attend to clean context.
        generator(noisy_image_or_video=denoised, conditional_dict=conditional_dict,
                  timestep=torch.ones_like(timestep) * config.context_noise,
                  kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache, current_start=start)
        t2 = sync_time()

        out = stream(denoised)
        t3 = sync_time()
        if out is not None:
            frames.append(out)
        blocks.append({"steps": len(steps), "denoise": (t1 - t0) * 1000, "kv_refresh": (t2 - t1) * 1000,
                       "decode": (t3 - t2) * 1000, "total": (t3 - t0) * 1000,
                       "frames": 1 if i == 0 else 4})
    return blocks, frames


# ---------------------------------------------------------------- benchmark
label = f"cf++ {args.steps}-step | 832x480 | decoder={args.decoder} | " \
        f"{'fp8' if args.fp8 else 'bf16'} | {'compile:' + args.compile_mode if args.compile else 'eager'}"
print(f"\n=== {label}")
t = time.perf_counter()
generate(args.seed)  # warm-up (and compilation, if enabled)
print(f"warm-up run: {time.perf_counter() - t:.1f}s")

torch.cuda.reset_peak_memory_stats()
runs = []
for r in range(args.runs):
    blocks, frames = generate(args.seed + 1 + r)
    runs.append(blocks)


def mean(xs):
    return sum(xs) / len(xs)


steady = [b for blocks in runs for b in blocks[1:]]  # skip block 0 (4-step first frame)
last = [blocks[-1] for blocks in runs]  # fullest KV cache = worst case
first = [blocks[0] for blocks in runs]
steady_ms = mean([b["total"] for b in steady])
fps_steady = 4000 / steady_ms
fps_worst = 4000 / mean([b["total"] for b in last])
fps_denoise_only = 4000 / mean([b["denoise"] + b["kv_refresh"] for b in steady])
total_frames = sum(b["frames"] for b in runs[0])
# Block i attends to i cached frames, so this is cost vs. context length. The
# longest context that still fits the real-time budget bounds a rolling window.
per_block = [mean([blocks[i]["total"] for blocks in runs]) for i in range(args.latent_frames)]
budget_ms = 4 * 1000 / PLAYBACK_FPS
max_context = next((i - 1 for i in range(1, len(per_block)) if per_block[i] > budget_ms), len(per_block) - 1)
fps_end_to_end = mean([total_frames * 1000 / sum(b["total"] for b in blocks) for blocks in runs])

result = {
    "label": label, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    "fps_steady": round(fps_steady, 2), "rtf_steady": round(PLAYBACK_FPS / fps_steady, 3),
    "fps_worst_block": round(fps_worst, 2), "rtf_worst_block": round(PLAYBACK_FPS / fps_worst, 3),
    "fps_end_to_end": round(fps_end_to_end, 2), "fps_without_decode": round(fps_denoise_only, 2),
    "first_frame_ms": round(mean([b["total"] for b in first]), 1),
    "block_ms": {k: round(mean([b[k] for b in steady]), 1) for k in ("denoise", "kv_refresh", "decode", "total")},
    "block_ms_first_vs_last": [round(mean([blocks[1]["total"] for blocks in runs]), 1),
                               round(mean([b["total"] for b in last]), 1)],
    "max_context_frames_under_budget": max_context,
    "block_ms_by_context": [round(ms, 1) for ms in per_block],
    "prompt_switch_encode_ms": round(prompt_encode_ms, 1),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
}
print(json.dumps(result, indent=2))
with open(os.path.join(os.path.dirname(REPO), "bench_results.jsonl"), "a") as f:
    f.write(json.dumps(result) + "\n")

if args.save and frames:
    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"cf{args.steps}step_{args.decoder}_{'fp8' if args.fp8 else 'bf16'}{'_compile' if args.compile else ''}.mp4"
    write_video(os.path.join(OUT_DIR, name), torch.cat(frames), fps=PLAYBACK_FPS)
    print(f"saved bench_out/{name}")
