"""
Community fine-tunes and LoRAs for the stream, merged into the generator at load time.

Add-ons for Wan 2.1 T2V 1.3B are trained on the base model, while the stream runs
Causal Forcing++ weights distilled from that same base. Each add-on is turned into a
weight delta and added on top, before FP8 quantization and compilation, so it costs
nothing per frame:
  - LoRA files (lora_A/lora_B, lora_down/lora_up with optional alpha, diff/diff_b):
    delta = up @ down * alpha / rank
  - full fine-tuned checkpoints: delta = fine-tune - Wan base
Every tensor in a file must map onto a model parameter of the same shape, or loading stops.
"""
import json
import os
import re

import torch
from safetensors import safe_open

PREFIXES = ("model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.", "model.")
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight", ".alpha",
                 ".diff", ".diff_b")
# diffusers WanTransformer3DModel names -> original Wan names used by this model
DIFFUSERS_TO_WAN = [
    (r"^condition_embedder\.text_embedder\.linear_1\.", "text_embedding.0."),
    (r"^condition_embedder\.text_embedder\.linear_2\.", "text_embedding.2."),
    (r"^condition_embedder\.time_embedder\.linear_1\.", "time_embedding.0."),
    (r"^condition_embedder\.time_embedder\.linear_2\.", "time_embedding.2."),
    (r"^condition_embedder\.time_proj\.", "time_projection.1."),
    (r"^proj_out\.", "head.head."),
    (r"^scale_shift_table$", "head.modulation"),
    (r"\.attn1\.", ".self_attn."),
    (r"\.attn2\.", ".cross_attn."),
    (r"\.to_q\.", ".q."), (r"\.to_k\.", ".k."), (r"\.to_v\.", ".v."), (r"\.to_out\.0\.", ".o."),
    (r"\.ffn\.net\.0\.proj\.", ".ffn.0."),
    (r"\.ffn\.net\.2\.", ".ffn.2."),
    (r"\.norm2\.", ".norm3."),  # diffusers norm2 is the cross-attention norm, Wan's norm3
    (r"\.scale_shift_table$", ".modulation"),
]


def parse_specs(values, presets_path, base_dir):
    """--lora values -> [(path, scale)]. Each value is a preset name from presets_path (whose
    paths are relative to that file) or PATH[:SCALE] relative to base_dir."""
    presets = {}
    if os.path.exists(presets_path):
        with open(presets_path) as f:
            presets = json.load(f)
    specs = []
    for value in values:
        if value in presets:
            specs += parse_specs(presets[value], presets_path, os.path.dirname(os.path.abspath(presets_path)))
            continue
        match = re.fullmatch(r"(.+?)(?::([-+]?\d*\.?\d+))?", value)
        path = os.path.normpath(os.path.join(base_dir, match.group(1)))
        assert os.path.exists(path), f"--lora {value}: no preset by that name and no file at {path}"
        specs.append((path, float(match.group(2) or 1.0)))
    return specs


def _to_wan_name(key, params, aliases):
    for prefix in PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("lora_unet_"):  # kohya: dots flattened to underscores
        return aliases.get(key[len("lora_unet_"):])
    if key not in params and not any(key + s in params for s in (".weight", ".bias")):
        for pattern, repl in DIFFUSERS_TO_WAN:
            key = re.sub(pattern, repl, key)
    return key


@torch.no_grad()
def merge_into(model, specs, base_path):
    """Add each (path, scale) add-on into `model` (a CausalWanModel, fp32 weights) in place."""
    params = dict(model.named_parameters())
    aliases = {name.rsplit(".", 1)[0].replace(".", "_"): name.rsplit(".", 1)[0] for name in params}
    for path, scale in specs:
        changed, rel = 0, []
        with safe_open(path, "pt") as f:
            keys = list(f.keys())
            is_lora = any(k.endswith(LORA_SUFFIXES) for k in keys)
            deltas = _lora_deltas(f, keys, params, aliases) if is_lora else _finetune_deltas(f, keys, params, aliases, base_path)
            for name, delta in deltas:
                p = params[name]
                assert delta.shape == p.shape, f"{os.path.basename(path)}: {name} is {tuple(delta.shape)}, model has {tuple(p.shape)}"
                rel.append((delta.norm() / p.float().norm().clamp_min(1e-12)).item())
                p.copy_((p.float() + scale * delta.to(p.device)).to(p.dtype))
                changed += 1
        rel.sort()
        print(f"merged {os.path.basename(path)} ({'LoRA' if is_lora else 'fine-tune delta'}, scale {scale:g}): "
              f"{changed} tensors, relative change median {rel[len(rel) // 2]:.2%}, max {rel[-1]:.2%}")


def _lora_deltas(f, keys, params, aliases):
    modules = {}
    for key in keys:
        suffix = next((s for s in LORA_SUFFIXES if key.endswith(s)), None)
        assert suffix, f"unrecognised LoRA tensor {key}"
        modules.setdefault(key[:-len(suffix)], {})[suffix] = f.get_tensor(key).float()
    for module, parts in modules.items():
        name = _to_wan_name(module, params, aliases)
        assert name is not None, f"no model module for LoRA {module}"
        if ".diff" in parts:
            yield name + ".weight", parts[".diff"]
        if ".diff_b" in parts:
            yield name + ".bias", parts[".diff_b"]
        down = parts.get(".lora_A.weight", parts.get(".lora_down.weight"))
        up = parts.get(".lora_B.weight", parts.get(".lora_up.weight"))
        if down is None and up is None:
            continue
        assert down is not None and up is not None, f"LoRA {module} has only one of up/down"
        rank = down.shape[0]
        alpha = parts[".alpha"].item() if ".alpha" in parts else rank
        target = params[name + ".weight"]
        yield name + ".weight", (up.flatten(1) @ down.flatten(1)).reshape(target.shape) * (alpha / rank)


def _finetune_deltas(f, keys, params, aliases, base_path):
    with safe_open(base_path, "pt") as base:
        base_keys = set(base.keys())
        for key in keys:
            name = _to_wan_name(key, params, aliases)
            assert name in params and name in base_keys, f"fine-tune tensor {key} has no matching base/model weight"
            yield name, f.get_tensor(key).float() - base.get_tensor(name).float()
