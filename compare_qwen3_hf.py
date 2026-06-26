"""Compare the local Qwen3 implementation against HuggingFace Qwen3ForCausalLM.

Loads both models with the same weights, runs a forward pass on randomly generated
token ids, and reports logits / hidden-state differences.

Hidden-state alignment (HF uses output_hidden_states=True):
  - embeddings:     local tok_emb(input_ids)  vs  hf_hidden_states[0]
  - decoder layer:  local_hidden_states[i]    vs  hf_hidden_states[i + 1]
                    for i = 0 .. n_layers - 2
  - final norm:     local final_norm(hs[-1])  vs  hf_hidden_states[-1]
    (HF does not expose the last decoder block output separately; the final
     tuple entry is already post-RMSNorm.)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from dataclasses import dataclass

import torch
from transformers import Qwen3ForCausalLM

from evaluator import seed_everything
from qwen3 import (
    QWEN3_CONFIG_4B,
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen,
)

# Local configs keyed by HuggingFace repo id (extend as needed).
MODEL_CONFIGS: dict[str, dict] = {
    "Qwen/Qwen3-4B": QWEN3_CONFIG_4B,
    "Qwen/Qwen3-4B-Thinking-2507": {
        **QWEN3_CONFIG_4B,
        "rope_base": 5_000_000.0,
        "context_length": 40_960,
    },
}


@dataclass
class TensorDiff:
    max_abs: float
    mean_abs: float
    cosine: float

    def __str__(self) -> str:
        return (
            f"max_abs={self.max_abs:.6e}  mean_abs={self.mean_abs:.6e}  "
            f"cosine={self.cosine:.8f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local Qwen3Model vs HuggingFace Qwen3ForCausalLM."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace repo id or local snapshot directory.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Cache directory for weights (defaults to ./<repo-name>).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Inference dtype for both models.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="HF attention backend (eager matches the local SDPA path most closely).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run one model at a time to reduce peak GPU memory.",
    )
    parser.add_argument(
        "--vocab_min",
        type=int,
        default=1,
        help="Minimum token id for random inputs (inclusive).",
    )
    parser.add_argument(
        "--vocab_max",
        type=int,
        default=1000,
        help="Maximum token id for random inputs (exclusive).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance for logits / hidden-state max abs diff. "
        "Defaults to 1e-2 (float32) or 1.0 (bfloat16).",
    )
    parser.add_argument(
        "--cosine_atol",
        type=float,
        default=0.9999,
        help="Minimum cosine similarity required for hidden-state comparisons.",
    )
    return parser.parse_args()


def resolve_paths(model_name_or_path: str, local_dir: str | None) -> tuple[str, str]:
    if os.path.isdir(model_name_or_path):
        snapshot_dir = model_name_or_path
        repo_id = _repo_id_from_snapshot(snapshot_dir)
    else:
        repo_id = model_name_or_path
        snapshot_dir = local_dir or f"./{repo_id.split('/')[-1]}"
    return repo_id, snapshot_dir


def _repo_id_from_snapshot(snapshot_dir: str) -> str:
    config_path = os.path.join(snapshot_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        for key in ("_name_or_path", "name_or_path"):
            if key in cfg and cfg[key]:
                return cfg[key]
    return os.path.basename(os.path.normpath(snapshot_dir))


def torch_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def build_local_config(repo_id: str, dtype: torch.dtype) -> dict:
    if repo_id not in MODEL_CONFIGS:
        known = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(
            f"No local config registered for {repo_id!r}. Known repos: {known}"
        )
    return {**MODEL_CONFIGS[repo_id], "dtype": dtype}


def load_local_model(
    repo_id: str, snapshot_dir: str, cfg: dict, device: torch.device
) -> Qwen3Model:
    print(f"Loading local Qwen3Model from {snapshot_dir} ...")
    weights = download_from_huggingface_from_snapshots(repo_id, snapshot_dir)
    model = Qwen3Model(cfg)
    load_weights_into_qwen(model, cfg, weights)
    del weights
    model.to(device)
    model.eval()
    return model


def load_hf_model(
    snapshot_dir: str,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str,
) -> Qwen3ForCausalLM:
    print(f"Loading HuggingFace Qwen3ForCausalLM from {snapshot_dir} ...")
    kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if device.type == "cuda":
        kwargs["device_map"] = device
    else:
        kwargs["device_map"] = "cpu"

    model = Qwen3ForCausalLM.from_pretrained(snapshot_dir, **kwargs)
    model.eval()
    return model


def random_input_ids(
    batch_size: int,
    seq_len: int,
    vocab_min: int,
    vocab_max: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        vocab_min,
        vocab_max,
        (batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> TensorDiff:
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    diff = (a32 - b32).abs()
    denom = a32.norm() * b32.norm()
    cosine = float((a32 @ b32 / denom).item()) if denom > 0 else 1.0
    return TensorDiff(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        cosine=cosine,
    )


def argmax_match_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item())


@torch.no_grad()
def run_local(
    model: Qwen3Model, input_ids: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        logits, _, hidden_states = model(input_ids, return_cache=True)
    embeddings = model.tok_emb(input_ids)
    final_hidden = model.final_norm(hidden_states[-1])
    return logits, hidden_states, embeddings, final_hidden


@torch.no_grad()
def run_hf(
    model: Qwen3ForCausalLM, input_ids: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    outputs = model(input_ids, output_hidden_states=True)
    return outputs.logits, outputs.hidden_states


def compare_hidden_states(
    local_hs: list[torch.Tensor],
    hf_hs: tuple[torch.Tensor, ...],
    local_embeddings: torch.Tensor,
    local_final: torch.Tensor,
) -> list[tuple[str, TensorDiff]]:
    results: list[tuple[str, TensorDiff]] = []

    results.append(("embeddings", tensor_diff(local_embeddings, hf_hs[0])))

    # HF hidden_states[0] is the embedding; hidden_states[i + 1] is the output
    # of decoder layer i for i < n_layers - 1. The last tuple entry is the
    # post-final-norm state, not the raw last decoder block output.
    n_layers = len(local_hs)
    for layer_idx in range(n_layers - 1):
        hf_idx = layer_idx + 1
        results.append(
            (
                f"decoder_layer_{layer_idx}",
                tensor_diff(local_hs[layer_idx], hf_hs[hf_idx]),
            )
        )

    results.append(("final_norm", tensor_diff(local_final, hf_hs[-1])))
    return results


def print_report(
    logits_diff: TensorDiff,
    logits_argmax: float,
    hidden_diffs: list[tuple[str, TensorDiff]],
    atol: float,
    cosine_atol: float,
) -> bool:
    print("\n=== Logits ===")
    print(logits_diff)
    print(f"argmax match rate: {logits_argmax:.4f}")

    print("\n=== Hidden states ===")
    logits_ok = logits_diff.max_abs <= atol and logits_argmax == 1.0
    hidden_ok = True
    for name, diff in hidden_diffs:
        ok = diff.max_abs <= atol and diff.cosine >= cosine_atol
        flag = "OK" if ok else "DIFF"
        print(f"  [{flag}] {name:>20}: {diff}")
        hidden_ok = hidden_ok and ok

    print("\n=== Summary ===")
    if logits_ok and hidden_ok:
        print(
            f"PASS: argmax tokens match, logits max_abs <= {atol:g}, "
            f"hidden states cosine >= {cosine_atol:g}."
        )
    elif logits_ok:
        print(
            f"PASS (logits): argmax tokens match and logits max_abs <= {atol:g}. "
            f"Hidden states show larger abs diffs (common in bfloat16); "
            f"check cosine similarities above."
        )
    else:
        print(
            f"FAIL: argmax mismatch and/or logits max_abs > {atol:g}. "
            "Inspect hidden-state diffs for the first diverging layer."
        )
    return logits_ok


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype)
    atol = args.atol if args.atol is not None else (1.0 if dtype == torch.bfloat16 else 1e-2)
    repo_id, snapshot_dir = resolve_paths(args.model_name_or_path, args.local_dir)
    cfg = build_local_config(repo_id, dtype)

    print(f"repo_id={repo_id}")
    print(f"snapshot_dir={snapshot_dir}")
    print(f"device={device}  dtype={dtype}  atol={atol:g}  sequential={args.sequential}")

    input_ids = random_input_ids(
        args.batch_size,
        args.seq_len,
        args.vocab_min,
        args.vocab_max,
        device,
    )
    print(f"random input_ids shape={tuple(input_ids.shape)}  seed={args.seed}")

    if args.sequential:
        local_model = load_local_model(repo_id, snapshot_dir, cfg, device)
        local_logits, local_hs, local_emb, local_final = run_local(
            local_model, input_ids
        )
        local_logits = local_logits.cpu()
        local_hs = [h.cpu() for h in local_hs]
        local_emb = local_emb.cpu()
        local_final = local_final.cpu()
        del local_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        hf_model = load_hf_model(snapshot_dir, dtype, device, args.attn_implementation)
        hf_logits, hf_hs = run_hf(hf_model, input_ids)
        hf_logits = hf_logits.cpu()
        hf_hs = tuple(h.cpu() for h in hf_hs)
        del hf_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        local_model = load_local_model(repo_id, snapshot_dir, cfg, device)
        hf_model = load_hf_model(snapshot_dir, dtype, device, args.attn_implementation)

        local_logits, local_hs, local_emb, local_final = run_local(
            local_model, input_ids
        )
        hf_logits, hf_hs = run_hf(hf_model, input_ids)

        del local_model, hf_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logits_diff = tensor_diff(local_logits, hf_logits)
    hidden_diffs = compare_hidden_states(local_hs, hf_hs, local_emb, local_final)
    ok = print_report(
        logits_diff,
        argmax_match_rate(local_logits, hf_logits),
        hidden_diffs,
        atol,
        args.cosine_atol,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
