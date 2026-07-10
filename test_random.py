"""Quick shape smoke test for Qwen3Model.generate with random tiny configs.

=============================================================================
KEY ENVIRONMENT DIFFERENCES: local vs Modal
=============================================================================

This script used to run entirely on your laptop (`python test_random.py`).
It now runs the *same* shape checks inside a Modal container in the cloud.

What changes when you leave the local process:

1. **Process location**
   - Local: one Python process on your machine, using your venv / system packages.
   - Modal: `modal run` starts a thin local client; the real work runs in a
     remote Debian container that Modal builds from the `image = ...` definition
     below. Your laptop only orchestrates and prints returned logs/results.

2. **Dependencies**
   - Local: whatever you installed with `uv sync` / `pip` into this repo's env
     (see `pyproject.toml`).
   - Modal: *only* what is declared on `modal.Image` is available remotely.
     Installing packages locally does **not** put them in the container. That is
     why torch (and requests, which `qwen3` imports at module load) are listed
     on the Image explicitly.

3. **Local source modules**
   - Local: `from qwen3 import ...` works because the repo root is on
     `sys.path` when you run the script from this directory.
   - Modal: the container starts with a clean filesystem. Sibling modules are
     not present unless we mount them with `Image.add_local_python_source(...)`.
     Without that, `import qwen3` would fail inside the remote Function.

4. **Hardware / CUDA**
   - Local: this smoke test already forced `device="cpu"`. No GPU is required.
   - Modal: we likewise request no GPU (`@app.function` has no `gpu=`). The
     container has no NVIDIA driver, which is fine for this CPU-only test.

5. **CLI / entrypoint**
   - Local: `if __name__ == "__main__": main()` and `sys.argv` parsing.
   - Modal: `@app.local_entrypoint()` marks the function Modal invokes on your
     machine when you run `modal run test_random.py`. Typed parameters become
     CLI flags automatically (`--seed`, `--n-cases`). The entrypoint then calls
     `.remote(...)` to execute the heavy function in the cloud.

6. **How to run**
   - One-time: `pip install modal` (or `uv add modal`) and `modal setup`.
   - Then: `modal run test_random.py`
   - Optional flags: `modal run test_random.py --seed 42 --n-cases 5`

See https://modal.com/docs for Apps, Images, and Functions.
"""

import random

# ---------------------------------------------------------------------------
# CHANGE FROM LOCAL: import `modal` and define an App + Image.
#
# Locally there is no App object — you just run Python. On Modal, an App is the
# unit that groups Functions and owns the ephemeral run created by `modal run`.
# ---------------------------------------------------------------------------
import modal

# App name shows up in the Modal dashboard / CLI. Pick something stable so
# re-runs update the same logical app rather than littering unnamed stubs.
app = modal.App("latent-reasoning-test-random")

# CHANGE FROM LOCAL: declare the remote environment in code (no Dockerfile/YAML).
#
# `debian_slim` ≈ a minimal Debian + the requested Python. We pin 3.12 to match
# this repo's `.python-version`.
#
# `uv_pip_install` installs into the *container* image at build time. We only
# pull what this smoke test needs (not the full `pyproject.toml` stack of audio
# / eval extras), which keeps the image small and builds fast.
#
# `add_local_python_source` copies our local `qwen3.py` / `utils.py` into the
# container's import path (`/root`). Default `copy=False` mounts them at
# container start so editing those files does not force a full image rebuild.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        # Runtime deps used by the remote Function. Sibling modules pull these
        # in at import time even though this smoke test never hits the network
        # or shows a progress bar: `qwen3` imports `requests`, and `utils`
        # imports both `requests` and `tqdm`.
        "torch",
        "requests",
        "tqdm",
    )
    .add_local_python_source("qwen3", "utils")
)


def random_tiny_config(seed: int | None = None) -> dict:
    rng = random.Random(seed)

    n_heads = rng.choice([2, 4, 8])
    n_kv_groups = rng.choice([d for d in (1, 2, 4) if n_heads % d == 0])
    head_dim = rng.choice([16, 32, 64])
    emb_dim = n_heads * head_dim
    n_layers = rng.randint(1, 3)
    context_length = rng.randint(32, 128)
    vocab_size = rng.randint(256, 1024)

    # CHANGE FROM LOCAL: `torch` is imported inside the remote Function (see
    # `run_shape_smoke_tests`) so this helper receives `dtype` as a value that
    # was created only after torch was available in the container. Callers pass
    # `torch.float32` in; we keep the field name identical to the original cfg.
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
        # Placeholder; overwritten with torch.float32 once torch is imported
        # inside the Modal Function. Keeping the key preserves cfg shape.
        "dtype": None,
    }


def left_padded_batch(
    batch_size: int,
    seq_lens: list[int],
    vocab_size: int,
    torch_module,
    pad_id: int = 0,
    device: str = "cpu",
) -> tuple:
    """Build a left-padded batch.

    CHANGE FROM LOCAL: we take `torch_module` as an argument instead of using a
    global `import torch`. That lets the module stay importable on a machine
    that only has the Modal client installed (no local torch), while the remote
    Function passes in the container's torch.
    """
    max_len = max(seq_lens)
    input_ids = torch_module.full(
        (batch_size, max_len), pad_id, dtype=torch_module.long, device=device
    )
    attention_mask = torch_module.zeros(
        batch_size, max_len, dtype=torch_module.long, device=device
    )

    for i, length in enumerate(seq_lens):
        offset = max_len - length
        input_ids[i, offset:] = torch_module.randint(
            1, vocab_size, (length,), device=device
        )
        attention_mask[i, offset:] = 1

    return input_ids, attention_mask


def assert_generate_shapes(
    model,
    input_ids,
    attention_mask,
    max_new_tokens: int,
    torch_module,
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
    assert out.dtype == torch_module.long, (
        f"expected long tokens, got {out.dtype}"
    )


def run_case(name: str, cfg: dict, seed: int, torch_module, Qwen3Model) -> None:
    torch_module.manual_seed(seed)
    # Unchanged intent vs local: force CPU. On Modal we also did not request a
    # GPU on the Function, so CUDA would not be available anyway.
    device = torch_module.device("cpu")
    model = Qwen3Model(cfg).to(device)
    model.eval()

    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    eos_token_id = vocab_size - 1
    max_new_tokens = min(8, context_length // 4)

    # Single sequence, no padding.
    prompt_len = random.randint(4, min(16, context_length - max_new_tokens))
    input_ids = torch_module.randint(
        1, vocab_size, (1, prompt_len), dtype=torch_module.long, device=device
    )
    assert_generate_shapes(model, input_ids, None, max_new_tokens, torch_module)

    # Batched left-padded prompts.
    batch_size = random.randint(2, 4)
    seq_lens = [
        random.randint(4, min(16, context_length - max_new_tokens))
        for _ in range(batch_size)
    ]
    input_ids, attention_mask = left_padded_batch(
        batch_size, seq_lens, vocab_size, torch_module, device=device
    )
    assert_generate_shapes(
        model, input_ids, attention_mask, max_new_tokens, torch_module
    )

    # Early stop when every row emits eos on the first decode step.
    input_ids = torch_module.full((2, 4), 1, dtype=torch_module.long, device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=16,
        eos_token_id=eos_token_id,
    )
    assert out.shape[0] == 2
    assert out.shape[1] >= 4

    print(
        f"  ok  {name}  cfg={cfg['n_layers']}L/{cfg['emb_dim']}d "
        f"ctx={context_length}"
    )


# ---------------------------------------------------------------------------
# CHANGE FROM LOCAL: the body that used to live in `main()` (and the top-level
# torch / qwen3 imports) now runs inside a Modal Function.
#
# `@app.function(image=image)` means:
#   - Modal builds/pulls `image`,
#   - starts a container,
#   - deserializes arguments,
#   - executes this function there,
#   - streams stdout back to your terminal,
#   - returns the Python return value to the caller via `.remote()`.
#
# No `gpu=` → CPU instance (cheapest / enough for tiny random models).
# ---------------------------------------------------------------------------
@app.function(image=image)
def run_shape_smoke_tests(seed: int, n_cases: int) -> str:
    # CHANGE FROM LOCAL: import heavy / Image-only packages *inside* the
    # Function. If these lived at module top-level, `modal run` would also need
    # them installed on the *local* client machine just to parse the file.
    # Importing here means only the container (which has them via `image`)
    # needs torch / our model code.
    import torch
    from qwen3 import Qwen3Model

    print(f"test_random.py (modal)  seed={seed}  cases={n_cases}")
    for i in range(n_cases):
        cfg = random_tiny_config(seed + i)
        # Fill dtype now that torch exists in this process.
        cfg["dtype"] = torch.float32
        run_case(f"case_{i}", cfg, seed + i, torch, Qwen3Model)

    message = "all shape checks passed"
    print(message)
    # Returning a string lets the local entrypoint print a clear success signal
    # even if log streaming is interleaved with other Modal client output.
    return message


# ---------------------------------------------------------------------------
# CHANGE FROM LOCAL: replace `if __name__ == "__main__": main()` + sys.argv.
#
# `@app.local_entrypoint()` runs on your laptop when you invoke:
#     modal run test_random.py
# Typed args become CLI options: --seed, --n-cases.
#
# The entrypoint's job is orchestration only: pick defaults, then call
# `.remote(...)` so the smoke test executes in Modal's cloud container.
# Use `.local(...)` instead if you ever want to debug the Function body in-process
# without a remote container (still requires local torch + qwen3 imports).
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(seed: int = -1, n_cases: int = 3) -> None:
    # Preserve the original "random seed if omitted" behavior. Modal's CLI
    # always passes the default when the flag is absent, so we use -1 as the
    # sentinel (sys.argv absence is no longer observable here).
    if seed < 0:
        seed = random.randrange(1_000_000)

    # CHANGE FROM LOCAL: `main()` used to *be* the test. Now it dispatches to
    # the remote Function. Blocking call: waits for the container to finish and
    # returns the Function's return value.
    result = run_shape_smoke_tests.remote(seed, n_cases)
    print(result)
