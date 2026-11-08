"""
Endless Causal Forcing++ stream with a rolling attention window.

CausalStream produces one latent frame (4 pixel frames at 16 fps) per step().
Only the last `window` latent frames stay in the KV cache, so the cost per
step stops growing once the window is full, and the stream can run forever.
Prompts are swapped by resetting the cross-attention cache; the visual
context in the KV cache carries over, so the scene changes without a cut.

Run as a script for an endurance test:
    python stream.py --duration 180
which writes stream_out/<name>/{stream.mp4, timing.json, metrics.csv, contact.png}.
"""
import argparse
import csv
import json
import os
import queue
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "Causal-Forcing")
LAUNCH_DIR = os.getcwd()  # for resolving user-supplied paths after the chdir below
LORA_PRESETS = os.path.join(HERE, "loras", "presets.json")
os.chdir(REPO)  # the repo loads weights through relative paths
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import cv2
import imageio_ffmpeg
import torch
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from wan.modules.attention import attention
from wan.modules.causal_model import CausalWanSelfAttention
from wan.modules.model import rope_params
from demo_utils.taehv import TAEHV
import weights

PLAYBACK_FPS = 16
BUDGET_MS = 4 * 1000 / PLAYBACK_FPS  # one latent frame = 4 pixel frames
MAX_LATENT_FRAMES = 1024 * 64  # RoPE table length, ~4.5 h of video
MODELS = {
    1: ("configs/causal_forcing_dmd_framewise_1step.yaml", "checkpoints/causal-forcing++/framewise-1step.pt"),
    2: ("configs/causal_forcing_dmd_framewise_2step.yaml", "checkpoints/causal-forcing++/framewise-2step.pt"),
}
DEFAULT_SCHEDULE = [
    (0, "A first-person drone flight over a misty pine forest at sunrise, the camera gliding steadily "
        "forward between the trees, cinematic lighting."),
    (90, "A first-person drone flight low over a calm lake at sunrise, mist drifting on the water, "
         "mountains reflected on the surface, cinematic lighting."),
    (135, "A first-person drone flight along snowy mountain ridges in golden morning light, the camera "
          "gliding steadily forward, cinematic."),
]


_original_self_attn_forward = CausalWanSelfAttention.forward


def _ring_self_attn_forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None,
                            current_start=0, cache_start=None):
    """KV-cached self-attention over a ring buffer of frames.

    The repo's version shifts the whole cache left to evict a frame and reads
    indices back with .item(), and its RoPE offset is a Python int that changes
    every frame, which makes torch.compile recompile endlessly. Keys are stored
    already rotated by their absolute position and attention ignores key order,
    so the new frame can simply overwrite the oldest slot. CausalStream puts the
    frame's RoPE factors, write indices and filled length into kv_cache.
    """
    if kv_cache is None or "rope" not in kv_cache:
        return _original_self_attn_forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache,
                                           current_start, cache_start)
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)
    x = _ring_attention(q, k, v, kv_cache)
    return self.o(x.flatten(2))


@torch.compiler.disable
def _ring_attention(q, k, v, kv_cache):
    """RoPE, cache write and attention, kept out of torch.compile: inductor cannot generate
    complex-number kernels, and the attention length changes while the window fills."""
    b, s, n, _ = q.shape

    def rotate(t):  # same float64 math as causal_rope_apply
        t = torch.view_as_complex(t.to(torch.float64).reshape(b, s, n, -1, 2))
        return torch.view_as_real(t * kv_cache["rope"]).flatten(3).type_as(v)

    kv_cache["k"].index_copy_(1, kv_cache["write_idx"], rotate(k))
    kv_cache["v"].index_copy_(1, kv_cache["write_idx"], v)
    if kv_cache["read_idx"] is not None:  # re-cache: only the frames before this one
        keys, values = kv_cache["k"].index_select(1, kv_cache["read_idx"]), kv_cache["v"].index_select(1, kv_cache["read_idx"])
        return attention(rotate(q), keys, values)
    filled = kv_cache["filled"]
    return attention(rotate(q), kv_cache["k"][:, :filled], kv_cache["v"][:, :filled])


class CausalStream:
    """Endless frame-wise generation: step() returns the next 4 frames (1 for the very first step)."""

    TAEHV_CONTEXT = 3  # latent frames of history re-decoded for temporal continuity
    LATENT_H, LATENT_W = 30, 52  # tokens per frame after 2x2 patching of 60x104 latents

    def __init__(self, steps=1, window=12, sink=0, fp8=True, compile_mode="default", switch_mode="flush", keep=2,
                 guidance=0.0, guidance_frames=12, refresh_frames=0, loras=(), device="cuda"):
        self.device = torch.device(device)
        self.switch_mode, self.keep = switch_mode, keep
        self.guidance, self.guidance_frames = guidance, guidance_frames
        self.refresh_frames = refresh_frames  # regenerate the sink this often (0 = never)
        config_path, ckpt_path = MODELS[steps]
        config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(config_path))
        config.model_kwargs.local_attn_size = window + 1  # past frames + the frame being generated
        config.model_kwargs.sink_size = sink
        self.config = config

        pipeline = CausalInferencePipeline(config, device=self.device)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        key = "generator_ema" if "generator_ema" in state_dict else "generator"
        gen_sd = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in state_dict[key].items()}
        missing, _ = pipeline.generator.load_state_dict(gen_sd, strict=False)
        assert not missing, f"checkpoint is missing {len(missing)} keys, e.g. {missing[:3]}"
        del state_dict, gen_sd
        if loras:  # (path, scale) pairs, merged while the weights are still fp32 on the CPU
            weights.merge_into(pipeline.generator.model, loras,
                               base_path="wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
        self.pipeline = pipeline.to(dtype=torch.bfloat16)
        self.generator = pipeline.generator.to(self.device)
        self.text_encoder = pipeline.text_encoder  # stays on the CPU between encodes

        # Ring buffer layout: `sink` permanent slots, then a ring of the most recent frames.
        assert 0 <= sink < window, "the window needs room for recent frames besides the sink"
        CausalWanSelfAttention.forward = _ring_self_attn_forward
        self.slots, self.sink = window + 1, sink
        tokens = self.pipeline.frame_seq_length
        self.write_idx = [torch.arange(slot * tokens, (slot + 1) * tokens, device=self.device)
                          for slot in range(self.slots)]

        # Frame positions grow without bound, but Wan's RoPE table stops at 1024 latent
        # frames (~4 min). Attention only sees offsets inside the window, so a longer
        # table gives identical results and just postpones the limit.
        model = self.generator.model
        c = model.dim // model.num_heads // 2
        temporal = rope_params(MAX_LATENT_FRAMES, 2 * (c - 2 * (c // 3)))
        height, width = rope_params(1024, 2 * (c // 3)), rope_params(1024, 2 * (c // 3))
        h, w = self.LATENT_H, self.LATENT_W
        self.rope_temporal = temporal.to(self.device)
        self.rope_spatial = torch.cat([height[:h].view(h, 1, -1).expand(h, w, -1),
                                       width[:w].view(1, w, -1).expand(h, w, -1)], dim=-1).to(self.device)

        if fp8:
            from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
            quantize_(self.generator, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
        if compile_mode != "none":
            torch._dynamo.config.cache_size_limit = 64  # attention length grows until the window is full
            self.generator.compile(mode=compile_mode)
        self.decoder = TAEHV(checkpoint_path="checkpoints/taew2_1.pth").eval().requires_grad_(False) \
            .to(self.device, torch.float16)
        self.cond = None

    @torch.no_grad()
    def encode(self, prompts, offload=True):
        """Text embeddings for each prompt. Moves the 11 GB text encoder to the GPU, and back
        to the CPU afterwards unless offload=False (live use, where prompts keep coming)."""
        self.text_encoder.to(self.device)
        conds = [{k: v.detach() for k, v in self.text_encoder(text_prompts=[p]).items()} for p in prompts]
        if offload:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        return conds

    def set_prompt(self, cond):
        """Switch prompts. Mid-stream, `switch_mode` decides what happens to the cached frames:
        "flush" keeps only the sink and the last `keep` frames (Infinity-RoPE's KV flush), so
        the new prompt is not outvoted by a window full of the old scene; "recache" re-encodes
        the cached frames under the new prompt (LongLive); "resink" replaces the sink with a fresh
        opening frame for the new prompt; "none" just swaps the text. Modes combine with "+"."""
        if self.guidance and self.frame_index > 0:  # keep the old prompt and its text K/V to push away from
            self.old_cond, self.old_crossattn = self.cond, self.pipeline.crossattn_cache
            self.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=self.device)
            self.guide_left = self.guidance_frames
        self.cond = cond
        for cache in getattr(self.pipeline, "crossattn_cache", None) or []:
            cache["is_init"] = False  # recompute text K/V on the next forward
        if self.frame_index > 0:
            if "flush" in self.switch_mode:
                self._flush()
            if "recache" in self.switch_mode:
                self._recache()
            if "resink" in self.switch_mode and self.frame_index >= self.sink:
                self._new_sink()

    def reset(self, seed=0):
        self.pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=self.device)
        self.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=self.device)
        self.rng = torch.Generator(self.device).manual_seed(seed)
        self.frame_index = 0
        self.taehv_history = None
        self.sink_keys = None  # sink K per layer, rotated to their original positions 0..sink-1
        self.recent = []  # (slot, frame) of the cached non-sink frames, oldest first
        self.slot_latent = {}  # clean latent held in each slot, for re-caching
        self.guide_left = 0  # frames still guided away from the previous prompt
        self.sink_age = 0  # frames generated since the sink was last (re)generated

    def _assign_slot(self, i):
        """Slot for new frame i. Occupied slots always stay contiguous from 0, so attention can
        read a plain prefix of the cache; when the window is full the oldest recent frame goes."""
        if i < self.sink:
            return i
        if len(self.recent) == self.slots - self.sink:
            slot = self.recent.pop(0)[0]
        else:
            slot = self.sink + len(self.recent)
        self.recent.append((slot, i))
        return slot

    def _point_cache(self, position, slot, filled_slots=None, read_slots=None):
        """Tell every layer's ring attention the RoPE position and slot of the frame being run,
        and which slots it reads: the first `filled_slots`, or exactly `read_slots`."""
        h, w = self.LATENT_H, self.LATENT_W
        rope = torch.cat([self.rope_temporal[position].view(1, 1, -1).expand(h, w, -1), self.rope_spatial], dim=-1)
        state = {"rope": rope.reshape(h * w, 1, -1), "write_idx": self.write_idx[slot],
                 "filled": (filled_slots or 0) * self.pipeline.frame_seq_length,
                 "read_idx": None if read_slots is None else torch.cat([self.write_idx[s] for s in read_slots])}
        for cache in self.pipeline.kv_cache1:
            cache.update(state)

    def _point_cache_at_frame(self, i):
        slot = self._assign_slot(i)
        self._point_cache(i, slot, filled_slots=i + 1 if i < self.sink else self.sink + len(self.recent))
        self._place_sink()
        return slot

    def _place_sink(self):
        """Keep the sink directly in front of the oldest recent frame (see _slide_sink)."""
        if self.sink and self.recent and self.recent[0][1] > self.sink:
            self._slide_sink(self.pipeline.kv_cache1, shift=self.recent[0][1] - self.sink)

    def _flush(self):
        """Drop all recent frames except the last `keep`, moved to the front of the recent region."""
        kept = self.recent[-self.keep:] if self.keep else []
        tokens = self.pipeline.frame_seq_length
        for cache in self.pipeline.kv_cache1:
            saved = [(cache["k"][:, s * tokens:(s + 1) * tokens].clone(), cache["v"][:, s * tokens:(s + 1) * tokens].clone())
                     for s, _ in kept]
            for n, (k, v) in enumerate(saved):
                dst = self.sink + n
                cache["k"][:, dst * tokens:(dst + 1) * tokens] = k
                cache["v"][:, dst * tokens:(dst + 1) * tokens] = v
        latents = [self.slot_latent[s] for s, _ in kept]
        self.recent = [(self.sink + n, frame) for n, (_, frame) in enumerate(kept)]
        for (slot, _), latent in zip(self.recent, latents):
            self.slot_latent[slot] = latent

    @torch.no_grad()
    def _recache(self):
        """Re-run every cached frame, oldest first, under the current prompt, each attending only
        to the frames before it, and overwrite its K/V. Sink frames are rebuilt at their original
        positions (their pristine copy), then slid in front of the recent frames as usual."""
        p = self.pipeline
        context_t = torch.zeros([1, 1], device=self.device, dtype=torch.float32)  # same input as the refresh pass
        sink_frames = [(f, f) for f in range(min(self.sink, self.frame_index))]
        done = []
        for n, (slot, frame) in enumerate(sink_frames + self.recent):
            if n == len(sink_frames) and self.sink_keys is not None:  # sink rebuilt: new pristine copy
                self.sink_keys = None
                self._place_sink()
            self._point_cache(frame, slot, read_slots=done + [slot])
            self.generator(noisy_image_or_video=self.slot_latent[slot], conditional_dict=self.cond,
                           timestep=context_t, kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                           current_start=0)
            done.append(slot)

    def _slide_sink(self, caches, shift):
        """Deep Forcing's "deep sink": move the sink frames' time position to just before the
        recent frames, so attention sees one contiguous clip that still starts with the opening
        frames the model was trained to expect. Rotates from a pristine copy each time, so bf16
        rounding does not accumulate."""
        sink_tokens = self.sink * self.pipeline.frame_seq_length
        if self.sink_keys is None:  # sink frames are final once the ring first wraps
            self.sink_keys = [cache["k"][:, :sink_tokens].clone() for cache in caches]
        rotation = self.rope_temporal[shift]  # complex, one factor per temporal channel pair
        t = rotation.shape[-1]
        for cache, keys in zip(caches, self.sink_keys):
            time = torch.view_as_complex(keys[..., :2 * t].to(torch.float64).reshape(*keys.shape[:-1], t, 2))
            cache["k"][:, :sink_tokens, :, :2 * t] = torch.view_as_real(time * rotation).flatten(-2).to(keys.dtype)

    @torch.no_grad()
    def _denoise_and_cache(self, steps, guide=False):
        """Denoise one latent frame from noise at the slot and position set by _point_cache, then
        re-run it clean so it is cached as context for later frames. Returns the clean latent."""
        p = self.pipeline
        start = 0  # positions come from _point_cache; a changing int here would force recompiles
        noisy = torch.randn([1, 1, 16, 60, 104], device=self.device, dtype=torch.bfloat16, generator=self.rng)
        for j, current_timestep in enumerate(steps):
            timestep = torch.ones([1, 1], device=self.device, dtype=torch.int64) * current_timestep
            # Transition guidance, x0 = x0_new + g * (x0_new - x0_old), only on the first (layout-setting)
            # denoising step so it costs one extra pass per frame.
            guide_now = guide and j == 0
            if guide_now:
                _, old = self.generator(noisy_image_or_video=noisy, conditional_dict=self.old_cond, timestep=timestep,
                                        kv_cache=p.kv_cache1, crossattn_cache=self.old_crossattn, current_start=start)
            _, denoised = self.generator(noisy_image_or_video=noisy, conditional_dict=self.cond, timestep=timestep,
                                         kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                                         current_start=start)
            if guide_now:
                denoised = denoised + self.guidance * (denoised - old)
            if j < len(steps) - 1:
                noisy = p.scheduler.add_noise(
                    denoised.flatten(0, 1), torch.randn_like(denoised.flatten(0, 1)),
                    steps[j + 1] * torch.ones([1], device=self.device, dtype=torch.long),
                ).unflatten(0, denoised.shape[:2])

        # Re-run on the clean frame so later frames attend to clean context.
        self.generator(noisy_image_or_video=denoised, conditional_dict=self.cond,
                       timestep=torch.ones_like(timestep) * self.config.context_noise,
                       kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache, current_start=start)
        return denoised

    @torch.no_grad()
    def _new_sink(self):
        """Replace the sink with fresh opening frames for the current prompt, generated without
        context like the start of a new video. The stream's anchor then describes the new scene
        while the kept recent frames carry the motion, so the picture morphs instead of cutting."""
        p = self.pipeline
        for f in range(self.sink):
            self._point_cache(f, f, filled_slots=f + 1)
            first = f == 0 and p.denoising_step_list_first_chunk is not None
            self.slot_latent[f] = self._denoise_and_cache(
                p.denoising_step_list_first_chunk if first else p.denoising_step_list)
        self.sink_keys = None  # pristine copy is retaken on the next slide
        self.sink_age = 0

    @torch.no_grad()
    def step(self):
        """Generate, cache and decode the next latent frame. Returns uint8 frames [T, H, W, C] on the CPU."""
        p, i = self.pipeline, self.frame_index
        if self.refresh_frames and self.sink_age >= self.refresh_frames and i >= self.sink:
            self._new_sink()  # a fresh anchor for the same prompt; recent frames drift toward it
        slot = self._point_cache_at_frame(i)
        steps = p.denoising_step_list_first_chunk if i == 0 and p.denoising_step_list_first_chunk is not None \
            else p.denoising_step_list
        denoised = self._denoise_and_cache(steps, guide=bool(self.guide_left))
        self.slot_latent[slot] = denoised
        self.guide_left = max(0, self.guide_left - 1)
        self.sink_age += 1

        window = denoised if self.taehv_history is None else torch.cat([self.taehv_history, denoised], dim=1)
        self.taehv_history = window[:, -self.TAEHV_CONTEXT:]
        pixels = self.decoder.decode_video(window.half(), parallel=True)[:, -4:]  # [0, 1]
        if i == 0:  # the first latent frame decodes to a single real frame
            pixels = pixels[:, 3:]
        self.frame_index += 1
        return (pixels[0].clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()


class Recorder(threading.Thread):
    """Writes frames to MP4 and collects per-frame image statistics off the generation thread."""

    THUMB_EVERY_S = 10

    def __init__(self, path, width, height):
        super().__init__(daemon=True)
        self.frames = queue.Queue()
        self.writer = imageio_ffmpeg.write_frames(path, (width, height), fps=PLAYBACK_FPS, codec="libx264",
                                                  quality=7, ffmpeg_log_level="error")
        self.writer.send(None)
        self.stats, self.thumbs = [], []
        self.prev_gray = None

    def run(self):
        while (frame := self.frames.get()) is not None:
            self.writer.send(np.ascontiguousarray(frame))
            gray_full = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(frame, (208, 120), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
            n = len(self.stats)
            self.stats.append({
                "brightness": float(gray.mean()),
                "saturation": float(cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1].mean()),
                "sharpness": float(cv2.Laplacian(gray_full, cv2.CV_64F).var()),
                "motion": float(np.abs(gray - self.prev_gray).mean()) if self.prev_gray is not None else 0.0,
            })
            self.prev_gray = gray
            if n % (self.THUMB_EVERY_S * PLAYBACK_FPS) == 0:
                self.thumbs.append((n / PLAYBACK_FPS, cv2.resize(frame, (312, 180), interpolation=cv2.INTER_AREA)))
        self.writer.close()


def load_schedule(path):
    if path is None:
        return DEFAULT_SCHEDULE
    with open(os.path.join(LAUNCH_DIR, path)) as f:
        return [(float(t), p.strip()) for t, p in (line.split("\t", 1) for line in f if line.strip())]


def contact_sheet(thumbs, path, columns=6):
    tiles = []
    for t, img in thumbs:
        img = img.copy()
        cv2.putText(img, f"{int(t)}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % columns)
    rows = [np.concatenate(tiles[r:r + columns], axis=1) for r in range(0, len(tiles), columns)]
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=sorted(MODELS), default=2)
    parser.add_argument("--window", type=int, default=6, help="past latent frames kept in attention")
    parser.add_argument("--sink", type=int, default=1, help="first latent frames never evicted from the window")
    parser.add_argument("--duration", type=float, default=180, help="seconds of video to generate")
    parser.add_argument("--schedule", help="text file of '<seconds>\\t<prompt>' lines (default: built-in)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_fp8", action="store_true")
    parser.add_argument("--switch", default="flush+resink", choices=["none", "flush", "recache", "flush+recache", "resink", "flush+resink"],
                        help="what happens to the cached frames on a prompt switch (see CausalStream.set_prompt)")
    parser.add_argument("--keep", type=int, default=2, help="recent frames kept by a flush")
    parser.add_argument("--guidance", type=float, default=0.0,
                        help="after a switch, push predictions away from the old prompt by this factor")
    parser.add_argument("--guidance_s", type=float, default=3.0, help="seconds of guided frames after a switch")
    parser.add_argument("--refresh_s", type=float, default=8.0, help="regenerate the sink every N seconds (0 = off)")
    parser.add_argument("--lora", action="append", default=[],
                        help="preset name from loras/presets.json, or PATH[:SCALE] of a LoRA or full Wan 1.3B "
                             "fine-tune; repeatable. Without it the vanilla model runs.")
    parser.add_argument("--compile_mode", default="default",
                        choices=["none", "default", "max-autotune-no-cudagraphs", "reduce-overhead"])
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or f"cf{args.steps}step_w{args.window}_s{args.sink}_{int(args.duration)}s" \
        + f"_{args.switch}" + (f"_g{args.guidance:g}" if args.guidance else "") + (f"_r{args.refresh_s:g}" if args.refresh_s else "")
    out_dir = os.path.join(HERE, "stream_out", name)
    os.makedirs(out_dir, exist_ok=True)
    schedule = load_schedule(args.schedule)

    stream = CausalStream(steps=args.steps, window=args.window, sink=args.sink,
                          fp8=not args.no_fp8, compile_mode=args.compile_mode,
                          switch_mode=args.switch, keep=args.keep, guidance=args.guidance,
                          guidance_frames=round(args.guidance_s * PLAYBACK_FPS / 4),
                          refresh_frames=round(args.refresh_s * PLAYBACK_FPS / 4),
                          loras=weights.parse_specs(args.lora, LORA_PRESETS, LAUNCH_DIR))
    conds = stream.encode([p for _, p in schedule])

    # Warm-up past the point where the window is full, so every compiled shape exists.
    t = time.perf_counter()
    stream.reset(seed=args.seed + 1000)
    stream.set_prompt(conds[0])
    for _ in range(args.window + args.sink + 8):
        stream.step()
    stream.set_prompt(conds[-1])  # exercise the re-cache path before timing
    stream.step()
    print(f"warm-up: {time.perf_counter() - t:.1f}s")

    recorder = Recorder(os.path.join(out_dir, "stream.mp4"), 832, 480)
    recorder.start()
    stream.reset(seed=args.seed)
    stream.set_prompt(conds[0])
    torch.cuda.reset_peak_memory_stats()

    total_frames = int(args.duration * PLAYBACK_FPS)
    next_prompt, switches = 1, []
    block_ms, available_at = [], []  # per block; per pixel frame (s since start)
    t_start = time.perf_counter()
    try:
        while len(available_at) < total_frames:
            video_s = len(available_at) / PLAYBACK_FPS
            if next_prompt < len(schedule) and video_s >= schedule[next_prompt][0]:
                t0 = time.perf_counter()
                stream.set_prompt(conds[next_prompt])
                torch.cuda.synchronize()
                switches.append({"video_s": round(video_s, 2), "block": len(block_ms), "prompt": schedule[next_prompt][1],
                                 "switch_ms": round((time.perf_counter() - t0) * 1000, 1)})
                print(f"[{video_s:6.1f}s] prompt -> {schedule[next_prompt][1][:70]}")
                next_prompt += 1

            t0 = time.perf_counter()
            frames = stream.step()
            t1 = time.perf_counter()
            block_ms.append((t1 - t0) * 1000)
            for frame in frames.numpy():
                recorder.frames.put(frame)
                available_at.append(t1 - t_start)

            if len(block_ms) % 40 == 0:  # every 10 s of video
                recent = block_ms[-40:]
                print(f"[{len(available_at) / PLAYBACK_FPS:6.1f}s] {4000 / np.mean(recent):5.1f} fps | "
                      f"worst block {max(recent):5.0f} ms | wall {t1 - t_start:6.1f}s")
    except KeyboardInterrupt:
        print("interrupted, writing what we have")
    finally:
        recorder.frames.put(None)
        recorder.join()

    # Playback starting at frame 0 with startup delay D never stalls iff D >= max_n(available_n - n/16).
    available = np.array(available_at)
    playback_lag = available - np.arange(len(available)) / PLAYBACK_FPS
    steady = np.array(block_ms[1:])
    per_10s = [round(4000 / np.mean(block_ms[k:k + 40]), 2) for k in range(1, len(block_ms), 40)]
    for s in switches:
        s["block_ms"] = round(block_ms[s["block"]], 1)
    summary = {
        "config": {"steps": args.steps, "window": args.window, "sink": args.sink, "fp8": not args.no_fp8,
                   "compile": args.compile_mode, "switch": args.switch, "keep": args.keep,
                   "guidance": args.guidance, "guidance_s": args.guidance_s, "refresh_s": args.refresh_s, "lora": args.lora, "seed": args.seed},
        "video_seconds": round(len(available) / PLAYBACK_FPS, 1),
        "wall_seconds": round(float(available[-1]), 1),
        "fps_mean": round(4000 / steady.mean(), 2),
        "rtf_mean": round(PLAYBACK_FPS * steady.mean() / 4000, 3),
        "block_ms": {"p50": round(float(np.percentile(steady, 50)), 1),
                     "p95": round(float(np.percentile(steady, 95)), 1),
                     "p99": round(float(np.percentile(steady, 99)), 1),
                     "max": round(float(steady.max()), 1)},
        "blocks_over_budget": int((steady > BUDGET_MS).sum()),
        "blocks_total": len(steady),
        "startup_delay_for_zero_stalls_s": round(float(playback_lag.max()), 3),
        "fps_per_10s": per_10s,
        "prompt_switches": switches,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "fps_per_10s"}, indent=2))
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump({**summary, "block_ms_all": [round(x, 1) for x in block_ms]}, f, indent=1)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["second", "brightness", "saturation", "sharpness", "motion"])
        for s in range(len(recorder.stats) // PLAYBACK_FPS):
            chunk = recorder.stats[s * PLAYBACK_FPS:(s + 1) * PLAYBACK_FPS]
            writer.writerow([s] + [round(np.mean([c[k] for c in chunk]), 2)
                                   for k in ("brightness", "saturation", "sharpness", "motion")])
    contact_sheet(recorder.thumbs, os.path.join(out_dir, "contact.png"))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
