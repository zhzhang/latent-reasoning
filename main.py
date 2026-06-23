import argparse
import glob
import os

import torch
from transformers import AutoTokenizer

from evaluator import (
    clean_answer,
    extract_answer_from_output,
    is_correct,
    seed_everything,
)
from qwen3 import (
    QWEN3_CONFIG_4B,
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen,
)
from utils import download_url, load_jsonl

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "2909d34ef28520753df82a2234c357259d254aa8/grade_school_math/data/test.jsonl"
)

# Qwen3-4B-Thinking-2507 differs from the vanilla Qwen3-4B config: it uses a
# larger RoPE base (theta) and a much longer native context window.
QWEN3_4B_THINKING_CONFIG = {
    **QWEN3_CONFIG_4B,
    "rope_base": 5_000_000.0,
    "context_length": 40_960,  # plenty for GSM8K; full model supports 262_144
}

# Final-answer instruction recommended by the Qwen3 model card for math tasks.
MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Token id for `` in Qwen3 thinking models.
THINK_END_TOKEN_ID = 151668


def ensure_libcuda_on_path():
    """Make libcuda.so discoverable so torch.compile's Triton backend can build.

    On some setups libcuda.so.1 exists but is not in the linker cache, which makes
    Triton fail with "libcuda.so cannot found!". Triton re-reads LD_LIBRARY_PATH
    from the environment at compile time, so prepending the right directory here is
    enough (no need to relaunch the process).
    """
    for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"):
        if glob.glob(os.path.join(d, "libcuda.so*")):
            current = os.environ.get("LD_LIBRARY_PATH", "")
            if d not in current.split(":"):
                os.environ["LD_LIBRARY_PATH"] = (d + ":" + current).rstrip(":")
            return d
    return None


def load_model_and_tokenizer(repo_id, local_dir, device, compile_model=True):
    print(f"Loading tokenizer from {repo_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    print(f"Downloading / loading weights for {repo_id} ...")
    weights = download_from_huggingface_from_snapshots(repo_id, local_dir)

    # Weights are stored in bfloat16; the config dtype keeps the whole model in
    # half precision (RMSNorm internally upcasts to fp32 for numerical stability).
    model = Qwen3Model(QWEN3_4B_THINKING_CONFIG)
    load_weights_into_qwen(model, QWEN3_4B_THINKING_CONFIG, weights)
    del weights
    model.to(device)
    model.eval()

    if compile_model and device == "cuda":
        ensure_libcuda_on_path()
        print("Compiling model with torch.compile (first step will be slow) ...")
        # Compile forward (not the whole module): generate() is a Python loop that
        # calls self.forward each step, so the compiled graph must live on forward.
        # dynamic=True: the KV cache grows by one token per decode step, so we
        # avoid recompiling on every new sequence length.
        model.forward = torch.compile(model.forward, dynamic=True)

    return model, tokenizer


def build_chat_prompt(tokenizer, question):
    messages = [{"role": "user", "content": f"{question}\n{MATH_INSTRUCTION}"}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def split_completion_tokens(new_tokens):
    """Split generated token ids into reasoning and answer portions."""
    try:
        think_end = len(new_tokens) - new_tokens[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        think_end = 0
    return new_tokens[:think_end], new_tokens[think_end:]


@torch.no_grad()
def generate_completion(model, tokenizer, prompt, device, gen_kwargs):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    new_tokens = output_ids[0][input_ids.shape[1]:].tolist()
    reasoning_tokens, answer_tokens = split_completion_tokens(new_tokens)
    reasoning = tokenizer.decode(reasoning_tokens, skip_special_tokens=True).strip("\n")
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip("\n")
    return reasoning, answer_text, len(reasoning_tokens)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="Qwen/Qwen3-4B-Thinking-2507"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./Qwen3-4B-Thinking-2507",
        help="Where to cache the downloaded model snapshot.",
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of GSM8K test questions to evaluate (-1 for all).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile (compilation has a one-time warmup cost).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Allow TF32 for the fp32 matmuls (e.g. RMSNorm upcast) and speed up SDPA.
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    test_filepath = os.path.join(args.data_root, "gsm8k_test.jsonl")
    if not os.path.exists(test_filepath):
        download_url(GSM8K_TEST_URL, args.data_root)
        os.rename(os.path.join(args.data_root, "test.jsonl"), test_filepath)

    list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")
    if args.num_samples and args.num_samples > 0:
        list_data_dict = list_data_dict[: args.num_samples]

    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, args.local_dir, device, compile_model=not args.no_compile
    )

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    answers = []
    for sample in list_data_dict:
        prompt = build_chat_prompt(tokenizer, sample["instruction"])
        reasoning, answer_text, num_reasoning_tokens = generate_completion(
            model, tokenizer, prompt, device, gen_kwargs
        )
        model_answer = clean_answer(answer_text)
        is_cor = is_correct(model_answer, sample["output"])
        answers.append(is_cor)

        print(
            f'Question: {sample["instruction"]}\n\n'
            f'Gold: {extract_answer_from_output(sample["output"])}\n\n'
            f"Reasoning ({num_reasoning_tokens} tokens):\n{reasoning}\n\n"
            f"Model Answer: {model_answer}\n\n"
            f"Is correct: {is_cor}\n"
            f"Running accuracy: {sum(answers)}/{len(answers)} = "
            f"{sum(answers) / len(answers):.4f}\n"
            + "=" * 80
        )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {sum(answers) / len(answers) if answers else 0.0}.",
            file=f,
        )

    print(
        f"\nFinal: {sum(answers)}/{len(answers)} correct "
        f"(accuracy {sum(answers) / len(answers) if answers else 0.0:.4f})"
    )


if __name__ == "__main__":
    main()
