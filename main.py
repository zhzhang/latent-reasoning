import argparse
import os

from utils import download_url, load_jsonl

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


def load_model_and_tokenizer(repo_id, local_dir, device, compile_model=True):
    print(f"Loading tokenizer from {repo_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    # Decoder-only models must be left-padded for correct batched generation:
    # the last token of every row has to be a real token so the next-token logits
    # line up across the batch.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Downloading / loading weights for {repo_id} ...")
    weights = download_from_huggingface_from_snapshots(repo_id, local_dir)

    # Weights are stored in bfloat16; the config dtype keeps the whole model in
    # half precision (RMSNorm internally upcasts to fp32 for numerical stability).
    model = Qwen3Model(QWEN3_CONFIG_4B)
    load_weights_into_qwen(model, QWEN3_CONFIG_4B, weights)
    del weights
    model.to(device)
    model.eval()

    if compile_model and device == "cuda":
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
def generate_batch(model, tokenizer, prompts, device, gen_kwargs):
    """Generate completions for a batch of prompts (left-padded).

    Returns a list of (reasoning, answer_text, num_reasoning_tokens) tuples,
    one per prompt and in the same order.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )

    # With left padding the generated tokens start at the same column for every row.
    prompt_len = input_ids.shape[1]
    results = []
    for i in range(output_ids.shape[0]):
        new_tokens = output_ids[i][prompt_len:].tolist()
        print(tokenizer.decode(new_tokens))
        reasoning_tokens, answer_tokens = split_completion_tokens(new_tokens)
        reasoning = tokenizer.decode(reasoning_tokens, skip_special_tokens=True).strip("\n")
        answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip("\n")
        results.append((reasoning, answer_text, len(reasoning_tokens)))
    return results


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
        default=64,
        help="Number of GSM8K test questions to evaluate (-1 for all).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of questions to generate for in parallel.",
    )
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

    print("Loading model and tokenizer...")
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
    for start in range(0, len(list_data_dict), args.batch_size):
        batch = list_data_dict[start : start + args.batch_size]
        prompts = [build_chat_prompt(tokenizer, s["instruction"]) for s in batch]
        outputs = generate_batch(model, tokenizer, prompts, device, gen_kwargs)

        for sample, (reasoning, answer_text, num_reasoning_tokens) in zip(batch, outputs):
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
