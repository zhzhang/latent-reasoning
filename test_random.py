"""Quick shape smoke test for Qwen3Model.generate with random tiny configs."""

import random
import sys

from utils import ensure_libcuda_on_path

ensure_libcuda_on_path()

import torch

from qwen3 import Qwen3Model


def random_tiny_config(seed: int | None = None) -> dict:
    rng = random.Random(seed)

    n_heads = rng.choice([2, 4, 8])
    n_kv_groups = rng.choice([d for d in (1, 2, 4) if n_heads % d == 0])
    head_dim = rng.choice([16, 32, 64])
    emb_dim = n_heads * head_dim
    n_layers = rng.randint(1, 3)
    context_length = rng.randint(32, 128)
    vocab_size = rng.randint(256, 1024)

    return {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "emb_dim": emb_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "hidden_dim": emb_dim * rng.choice([2, 4]),
        "head_dim": head_dim,
        "qk_norm": rng.choice([True, False]),
        "n_kv_groups": n_kv_groups,
        "rope_base": 10_000.0,
        "dtype": torch.float32,
    }


def left_padded_batch(
    batch_size: int,
    seq_lens: list[int],
    vocab_size: int,
    pad_id: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(seq_lens)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

    for i, length in enumerate(seq_lens):
        offset = max_len - length
        input_ids[i, offset:] = torch.randint(
            1, vocab_size, (length,), device=device
        )
        attention_mask[i, offset:] = 1

    return input_ids, attention_mask


def assert_generate_shapes(
    model: Qwen3Model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> None:
    batch_size, prompt_len = input_ids.shape
    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )

    assert out.ndim == 2, f"expected 2D output, got shape {tuple(out.shape)}"
    assert out.shape[0] == batch_size, (
        f"batch dim mismatch: {out.shape[0]} vs {batch_size}"
    )
    assert out.shape[1] >= prompt_len, (
        f"output shorter than prompt: {out.shape[1]} vs {prompt_len}"
    )
    assert out.shape[1] <= prompt_len + max_new_tokens, (
        f"generated too many tokens: {out.shape[1] - prompt_len} > {max_new_tokens}"
    )
    assert out.dtype == torch.long, f"expected long tokens, got {out.dtype}"


def run_case(name: str, cfg: dict, seed: int) -> None:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = Qwen3Model(cfg).to(device)
    model.eval()

    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    eos_token_id = vocab_size - 1
    max_new_tokens = min(8, context_length // 4)

    # Single sequence, no padding.
    prompt_len = random.randint(4, min(16, context_length - max_new_tokens))
    input_ids = torch.randint(
        1, vocab_size, (1, prompt_len), dtype=torch.long, device=device
    )
    assert_generate_shapes(model, input_ids, None, max_new_tokens)

    # Batched left-padded prompts.
    batch_size = random.randint(2, 4)
    seq_lens = [
        random.randint(4, min(16, context_length - max_new_tokens))
        for _ in range(batch_size)
    ]
    input_ids, attention_mask = left_padded_batch(
        batch_size, seq_lens, vocab_size, device=device
    )
    assert_generate_shapes(model, input_ids, attention_mask, max_new_tokens)

    # Early stop when every row emits eos on the first decode step.
    input_ids = torch.full((2, 4), 1, dtype=torch.long, device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=16,
        eos_token_id=eos_token_id,
    )
    assert out.shape[0] == 2
    assert out.shape[1] >= 4

    print(f"  ok  {name}  cfg={cfg['n_layers']}L/{cfg['emb_dim']}d ctx={context_length}")


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(1_000_000)
    n_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"test_random.py  seed={seed}  cases={n_cases}")
    for i in range(n_cases):
        cfg = random_tiny_config(seed + i)
        run_case(f"case_{i}", cfg, seed + i)

    print("all shape checks passed")


if __name__ == "__main__":
    main()
