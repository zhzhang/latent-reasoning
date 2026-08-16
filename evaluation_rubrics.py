import argparse
import json
import os
import re
import logging
import time
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Callable, TypedDict, AsyncGenerator
from collections import defaultdict
from dataclasses import dataclass, field

from openai import APIStatusError, APITimeoutError, APIConnectionError, AsyncOpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

EVALUATE_SYS_PROMPT = "You are a impartial and meticulous AI evaluator. Your task is to assess a model's response to a question by evaluating its reasoning and its final answer. You must strictly follow the provided ground truth information and the detailed scoring rubric."
MODEL_NAME = "gpt-4o-2024-11-20"
ANTHROPIC_MODEL_NAME = "claude-sonnet-4-5"
NUM_RATERS = 5
SEED = 42
LENGTH_LIMIT = 10000

DEFAULT_API_MAX_RETRIES = 20
DEFAULT_API_RETRY_INTERVAL = 1.0
DEFAULT_MAX_WORKERS = 80
DEFAULT_QPS = 8.0
DEFAULT_TIMEOUT = 120.0

# ======== Utils ========

class ScorerAPIError(Exception):
    pass


class ResponseValidationError(Exception):
    pass


@dataclass
class Metrics:
    correct: int = 0
    accum_score: float = 0.0
    total: int = 0


@dataclass
class EvaluationResult:
    record_id: str
    exception: Optional[str] = None
    correct: bool = False
    score: float = 0.0
    raw_responses: List[str] = field(default_factory=list)
    rubric_results: List[Dict[str, Any]] = field(default_factory=list)


class InputItem(TypedDict):
    id: str
    question: str
    answer: str
    thinking: str
    cue: List[str]
    choices: List[str]
    rubric: List[Dict[str, Any]]
    thinking_prediction: str
    answer_prediction: str
    modality: str
    category: str


class EvaluatedItem(InputItem):
    new: bool
    score: float
    correct: bool
    raw_responses: List[str]


class AsyncRateLimiter:
    def __init__(self, qps: float):
        self.base_qps = float(qps)
        self.current_qps = float(qps)
        self.lock = asyncio.Lock()
        self.next_time = time.monotonic()

    @property
    def interval(self):
        return 1.0 / self.current_qps

    def degrade(self):
        self.current_qps = max(self.current_qps / 2.0, self.base_qps / 16.0)
        logger.warning(f"[RateLimiter] Rate limit encountered. Reducing QPS to {self.current_qps:.2f}.")

    def restore(self):
        if self.current_qps < self.base_qps:
            self.current_qps = min(self.current_qps + 0.5, self.base_qps)
            logger.warning(f"[RateLimiter] Recovering. Increasing QPS to {self.current_qps:.2f}.")

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                wait_time = self.next_time - now
                await asyncio.sleep(wait_time)
                now = time.monotonic()

            self.next_time = now + self.interval


class BaseScorer:
    """Shared retry / rate-limit loop for LLM rubric judges."""

    def __init__(
        self,
        api_max_retries: int,
        api_retry_interval: float,
        max_workers: int,
        qps: float,
        timeout: float,
        model_name: str,
    ):
        self.api_max_retries = api_max_retries
        self.api_retry_interval = api_retry_interval
        self.max_workers = max_workers
        self.qps = qps
        self.timeout = timeout
        self.model_name = model_name
        self.rate_limiter = AsyncRateLimiter(qps=qps)

    async def _complete(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        raise NotImplementedError

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in (429, 418):
            return True
        return "rate" in str(exc).lower()

    async def call(
        self,
        log_id: str,
        parser: Callable,
        user_prompt: str,
        system_prompt: str = EVALUATE_SYS_PROMPT,
        **complete_kwargs,
    ) -> Tuple[Any, str]:
        for attempt in range(1, self.api_max_retries + 1):
            try:
                await self.rate_limiter.acquire()
                response_str = await self._complete(
                    system_prompt, user_prompt, **complete_kwargs
                )
                self.rate_limiter.restore()
            except (APITimeoutError, APIConnectionError) as e:
                self.rate_limiter.degrade()
                logger.warning(f"{log_id} (api_attempt={attempt}) Transient error: {e}")
                continue
            except Exception as e:
                if self._is_rate_limit_error(e):
                    self.rate_limiter.degrade()
                    logger.warning(f"{log_id} (api_attempt={attempt}) Rate limit hit: {e}")
                    continue
                if isinstance(e, APIStatusError):
                    raise ScorerAPIError(f"{log_id} OpenAI API returned an error: {e}") from e
                raise ScorerAPIError(f"{log_id} Unexpected exception during API call: {e}") from e

            if response_str is None:
                logger.warning(f"{log_id} (api_attempt={attempt}) Empty response content (None).")
                continue

            try:
                return parser(response_str), response_str
            except Exception as e:
                logger.warning(f"{log_id} (api_attempt={attempt}) Failed to parse evaluator response: {e}")

            await asyncio.sleep(self.api_retry_interval)

        raise ScorerAPIError(f"{log_id} Max retries ({self.api_max_retries}) exceeded.")


class OpenAIScorer(BaseScorer):
    """OpenAI judge. Interactive ``_complete`` is unused; batch via ``evaluate_records_openai_batch``."""

    def __init__(
        self,
        api_max_retries: int,
        api_retry_interval: float,
        max_workers: int,
        qps: float,
        timeout: float,
        model_name: str = MODEL_NAME,
    ):
        if not (os.environ.get("OPENAI_API_KEY") or "").strip():
            raise ValueError(
                "Environment variable OPENAI_API_KEY is not set. Please export it before running."
            )
        super().__init__(
            api_max_retries=api_max_retries,
            api_retry_interval=api_retry_interval,
            max_workers=max_workers,
            qps=qps,
            timeout=timeout,
            model_name=model_name,
        )
        self.client = AsyncOpenAI()

    async def _complete(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> Optional[str]:
        del kwargs
        raise RuntimeError(
            "OpenAI interactive scoring is disabled; use evaluate_records_openai_batch "
            "(Batch API, ~50% cheaper)."
        )

class AnthropicScorer(BaseScorer):
    def __init__(
        self,
        api_max_retries: int,
        api_retry_interval: float,
        max_workers: int,
        qps: float,
        timeout: float,
        model_name: str = ANTHROPIC_MODEL_NAME,
        max_tokens: int = 4096,
    ):
        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            raise ValueError(
                "Environment variable ANTHROPIC_API_KEY is not set. Please export it before running."
            )
        super().__init__(
            api_max_retries=api_max_retries,
            api_retry_interval=api_retry_interval,
            max_workers=max_workers,
            qps=qps,
            timeout=timeout,
            model_name=model_name,
        )
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic()
        self.max_tokens = max_tokens

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        if super()._is_rate_limit_error(exc):
            return True
        rate_limit = getattr(self._anthropic, "RateLimitError", None)
        return bool(rate_limit and isinstance(exc, rate_limit))

    async def _complete(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> Optional[str]:
        del kwargs
        response = await self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        parts = []
        for block in response.content or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None


# Back-compat alias used by StreamingRubricGrader / CLI.
class Scorer(OpenAIScorer):
    pass


def clean(string: str):
    """
    Removes leading/trailing whitespace from a multi-line string block and from each line within it.
    This is useful for cleaning up indented triple-quoted strings.
    """
    return "\n".join(line.lstrip() for line in string.strip().splitlines())


# ======== Evaluation ========

def create_evaluation_user_prompt(
    question: str,
    answer: str,
    thinking: str,
    cue,
    thinking_prediction: str,
    answer_prediction: str,
    rubric,
) -> str:
    """
    Constructs the detailed user prompt for the LLM-based evaluator.
    It provides all necessary context, the model's submission, the evaluation rubric,
    and a strict JSON output format for the LLM to follow.
    """
    formatted_cues = ", ".join(cue)  # Not used

    rubric_sections = []
    for item in rubric:
        section = f"""
        ### Criterion: {item['name']}
        - Scoring Rule: {item['scoring_point']}
        - Score Options: {item['choices']}
        - Evaluate: Carefully read the model's reasoning and determine if it meets this criterion.
        """
        rubric_sections.append(clean(section))

    rubric_string = "\n".join(rubric_sections)

    json_format_sections = []
    for i, item in enumerate(rubric):
        comma = "," if i < len(rubric) - 1 else ""
        section = f"""
        {{
        "name": "{item['name']}",
        "score": [Assign a score from {item['choices']}],
        "justification": "Your justification for this score, citing evidence from the model's reasoning. (less than 50 words)"
        }}{comma}"""
        json_format_sections.append(clean(section))

    json_format_string = "\n".join(json_format_sections)

    prompt = f"""
    ## Ground Truth & Problem Context

    - Question: {question}
    - Correct Answer: {answer}
    - Ideal Reasoning (for reference): {thinking}

    ## Evaluation Rubric & Scoring

    For each point in the rubric, provide a score and a brief justification based on the model's output.

    {rubric_string}

    ## Required Output Format

    Provide your complete evaluation in a single, valid JSON object. Do not add any text before or after the JSON. The JSON structure must be as follows:

    [
    {json_format_string}
    ]
    
    ## Model's Submission to Evaluate

    - Model's Reasoning:
    {thinking_prediction} The answer is {answer_prediction}.
    """
    return clean(prompt)


def parse_evaluator_response(log_id, raw_response, rubric):
    """Parse a judge response and return (normalized total, per-criterion details)."""
    match = re.search(r'\[\s*\{.*\}\s*\]', raw_response, re.DOTALL)
    content = match.group(0) if match else raw_response

    try:
        eval_result = json.loads(content)
    except json.JSONDecodeError as e:
        raise ResponseValidationError(f"{log_id} Invalid JSON in evaluator output: {str(e)}") from e

    if not isinstance(eval_result, list):
        raise ResponseValidationError(f"{log_id} Evaluator output must be a JSON list.")

    if len(eval_result) != len(rubric):
        raise ResponseValidationError(
            f"{log_id} Rubric length mismatch: expected {len(rubric)} items, got {len(eval_result)}."
        )

    criteria = []
    total_prediction_score = 0
    total_max_score = 0

    for i, (res_item, rub_item) in enumerate(zip(eval_result, rubric)):
        if not all(key in res_item for key in ["name", "score"]):
            raise ResponseValidationError(f"{log_id} Evaluator item {i} must include 'name' and 'score'.")

        expected_name = rub_item["name"]
        actual_name = res_item["name"]
        if actual_name != expected_name:
            logger.warning(
                f"{log_id} Rubric name mismatch at index {i}: expected '{expected_name}', got '{actual_name}'."
            )

        score = res_item["score"]
        allowed_choices: List[int] = rub_item["choices"]

        if not isinstance(score, (int, float)):
            raise ResponseValidationError(
                f"{log_id} Score for '{expected_name}' must be numeric, got {type(score).__name__}."
            )

        if score not in allowed_choices:
            raise ResponseValidationError(
                f"{log_id} Invalid score for section '{expected_name}': {score}. "
                f"Allowed choices are: {allowed_choices}."
            )

        criteria.append(
            {
                "name": expected_name,
                "score": score,
                "justification": str(res_item.get("justification") or ""),
            }
        )
        total_prediction_score += score
        total_max_score += max(allowed_choices)

    if total_max_score == 0:
        return 0.0, criteria

    return total_prediction_score / total_max_score, criteria


def validate_and_score(log_id, raw_response, rubric):
    score, _ = parse_evaluator_response(log_id, raw_response, rubric)
    return score


def criterion_passed(score, rub_item):
    return score == max(rub_item["choices"])


def build_failed_rubric_results(rubric):
    return [
        {
            "name": item["name"],
            "score": 0,
            "justification": "",
            "pass": False,
            "rater_scores": [],
            "rater_passes": [],
        }
        for item in rubric
    ]


def build_single_rater_rubric_results(rubric, criteria: List[Dict[str, Any]]):
    """Per-criterion results for a single judge decision (no multi-rater vote)."""
    results = []
    for rub_item, crit in zip(rubric, criteria):
        score = crit["score"]
        results.append(
            {
                "name": rub_item["name"],
                "score": score,
                "justification": crit.get("justification") or "",
                "pass": criterion_passed(score, rub_item),
            }
        )
    return results


def aggregate_rubric_results(rubric, rater_criterion_details):
    """Aggregate per-rater criterion scores into pass/fail booleans per rubric point."""
    results = []
    for index, rub_item in enumerate(rubric):
        scores = []
        for rater_details in rater_criterion_details:
            entry = rater_details[index]
            scores.append(entry["score"] if isinstance(entry, dict) else entry)
        rater_passes = [criterion_passed(score, rub_item) for score in scores]
        pass_count = sum(rater_passes)
        results.append(
            {
                "name": rub_item["name"],
                "pass": pass_count > len(rater_passes) / 2,
                "rater_scores": scores,
                "rater_passes": rater_passes,
            }
        )
    return results


def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))

    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)

    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)

    return cond1 and cond2


def evaluation_result_from_raw(
    record_id: str,
    rubric: List[Dict[str, Any]],
    raw_response: str,
    *,
    correct: Optional[bool] = None,
) -> EvaluationResult:
    """Parse one judge reply into an EvaluationResult (single-rater).

    ``correct`` records answer string-match status when provided; it does not
    gate whether the rubric judge ran.
    """
    log_id = f"(record_id={record_id})"
    score, criteria = parse_evaluator_response(log_id, raw_response, rubric)
    return EvaluationResult(
        record_id,
        score=score,
        correct=bool(correct) if correct is not None else True,
        raw_responses=[raw_response],
        rubric_results=build_single_rater_rubric_results(rubric, criteria),
    )


def failed_string_match_result(
    record_id: str,
    rubric: List[Dict[str, Any]],
) -> EvaluationResult:
    """Legacy helper: zero score without an LLM call (no longer used by default)."""
    return EvaluationResult(
        record_id,
        score=0.0,
        correct=False,
        rubric_results=build_failed_rubric_results(rubric),
    )


async def evaluate_one_record(
    scorer: BaseScorer,
    semaphore,
    record_id: str,
    question: str,
    answer: str,
    thinking: str,
    cue: List[str],
    choices: List[str],
    rubric: List[Dict[str, Any]],
    thinking_prediction: str,
    answer_prediction: str,
    *,
    num_raters: int = NUM_RATERS,
) -> EvaluationResult:
    async with semaphore:
        answer_correct = string_match(answer, answer_prediction, choices)

        if len(answer_prediction) >= LENGTH_LIMIT:
            logger.warning(f"(record_id={record_id}) 'answer_prediction' is too long (length={len(answer_prediction)} limit={LENGTH_LIMIT}). Skipped.")

        if len(thinking_prediction) >= LENGTH_LIMIT:
            logger.warning(f"(record_id={record_id}) 'thinking_prediction' is too long (length={len(thinking_prediction)} limit={LENGTH_LIMIT}). Skipped")

        user_prompt = create_evaluation_user_prompt(
            question, answer, thinking, cue,
            thinking_prediction, answer_prediction, rubric
        )

        scores = []
        raw_responses = []
        rater_criterion_details = []
        for rater_id in range(num_raters):
            log_id = f"(record_id={record_id} rater_id={rater_id})"

            try:
                parser_func = lambda res, log_id=log_id: parse_evaluator_response(log_id, res, rubric)
                (score, criteria), raw_response = await scorer.call(
                    log_id, parser_func, user_prompt, EVALUATE_SYS_PROMPT
                )
            except Exception as e:
                logger.exception(
                    f"{log_id} Scoring failed for this sample. It will be skipped in this run and counted as NOT EVALUATED. "
                    f"Please re-run the evaluation after this run completes to obtain a complete result set."
                )
                return EvaluationResult(record_id, exception=str(e), raw_responses=raw_responses)

            scores.append(score)
            raw_responses.append(raw_response)
            rater_criterion_details.append(criteria)

        if num_raters == 1:
            return EvaluationResult(
                record_id,
                score=scores[0],
                correct=answer_correct,
                raw_responses=raw_responses,
                rubric_results=build_single_rater_rubric_results(
                    rubric, rater_criterion_details[0]
                ),
            )

        sorted_scores = sorted(scores)
        if len(sorted_scores) >= 3:
            trimmed_scores = sorted_scores[1:-1]
        else:
            trimmed_scores = sorted_scores
        avg_score = sum(trimmed_scores) / len(trimmed_scores)

        return EvaluationResult(
            record_id,
            score=avg_score,
            correct=answer_correct,
            raw_responses=raw_responses,
            rubric_results=aggregate_rubric_results(rubric, rater_criterion_details),
        )


def _rater_custom_id(record_id: str, rater_id: int) -> str:
    return f"{record_id}__rater{rater_id}"


def evaluate_records_openai_batch(
    pending: List[InputItem],
    *,
    model_name: str = MODEL_NAME,
    num_raters: int = NUM_RATERS,
    work_dir: Optional[str] = None,
    poll_interval_s: float = 30.0,
) -> Dict[str, EvaluationResult]:
    """Score pending rubric items via the OpenAI Batch API (50% cheaper)."""
    from pathlib import Path

    from api_batch import openai_batch_chat_text, openai_chat_request, run_openai_chat_batch

    if not pending:
        return {}

    requests = []
    owners: Dict[str, Tuple[str, int]] = {}
    for item in pending:
        user_prompt = create_evaluation_user_prompt(
            item["question"],
            item["answer"],
            item["thinking"],
            item["cue"],
            item["thinking_prediction"],
            item["answer_prediction"],
            item["rubric"],
        )
        for rater_id in range(num_raters):
            custom_id = _rater_custom_id(item["id"], rater_id)
            owners[custom_id] = (item["id"], rater_id)
            requests.append(
                openai_chat_request(
                    custom_id=custom_id,
                    model=model_name,
                    temperature=0.0,
                    seed=SEED,
                    messages=[
                        {"role": "system", "content": EVALUATE_SYS_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            )

    batch_dir = Path(work_dir) if work_dir else Path("outputs") / "openai-rubrics-batch"
    print(
        f"[openai-batch] rubric scoring {len(pending)} items "
        f"× {num_raters} raters = {len(requests)} requests -> {batch_dir}"
    )
    raw_by_id = run_openai_chat_batch(
        requests,
        work_dir=batch_dir,
        display_name="rubrics",
        poll_interval_s=poll_interval_s,
    )

    by_record: Dict[str, Dict[int, str]] = {}
    for custom_id, (record_id, rater_id) in owners.items():
        text = openai_batch_chat_text(raw_by_id.get(custom_id) or {})
        if text:
            by_record.setdefault(record_id, {})[rater_id] = text

    results: Dict[str, EvaluationResult] = {}
    for item in pending:
        record_id = item["id"]
        answer_correct = string_match(
            item["answer"], item["answer_prediction"], item["choices"]
        )
        rater_map = by_record.get(record_id) or {}
        scores = []
        raw_responses = []
        rater_criterion_details = []
        failed = False
        for rater_id in range(num_raters):
            raw = rater_map.get(rater_id)
            if not raw:
                failed = True
                results[record_id] = EvaluationResult(
                    record_id,
                    exception=f"missing batch result for rater {rater_id}",
                    raw_responses=raw_responses,
                )
                break
            log_id = f"(record_id={record_id} rater_id={rater_id})"
            try:
                score, criteria = parse_evaluator_response(log_id, raw, item["rubric"])
            except Exception as exc:  # noqa: BLE001
                failed = True
                results[record_id] = EvaluationResult(
                    record_id,
                    exception=str(exc),
                    raw_responses=raw_responses + [raw],
                )
                break
            scores.append(score)
            raw_responses.append(raw)
            rater_criterion_details.append(criteria)
        if failed:
            continue
        if num_raters == 1:
            results[record_id] = EvaluationResult(
                record_id,
                score=scores[0],
                correct=answer_correct,
                raw_responses=raw_responses,
                rubric_results=build_single_rater_rubric_results(
                    item["rubric"], rater_criterion_details[0]
                ),
            )
        else:
            sorted_scores = sorted(scores)
            trimmed = (
                sorted_scores[1:-1] if len(sorted_scores) >= 3 else sorted_scores
            )
            results[record_id] = EvaluationResult(
                record_id,
                score=sum(trimmed) / len(trimmed),
                correct=answer_correct,
                raw_responses=raw_responses,
                rubric_results=aggregate_rubric_results(
                    item["rubric"], rater_criterion_details
                ),
            )
    return results


async def run_evaluation(
    input_data: List[InputItem],
    existing_output_data: List[EvaluatedItem],
    scorer: BaseScorer,
    *,
    num_raters: int = NUM_RATERS,
    batch_work_dir: Optional[str] = None,
    poll_interval_s: float = 30.0,
) -> AsyncGenerator[EvaluatedItem, None]:
    existing_ids = set()
    for existing_output_item in existing_output_data:
        keys = set(existing_output_item.keys())
        if not EvaluatedItem.__required_keys__ <= keys:
            logger.error(f"Existing output data item is missing required keys: Got {keys}, expected {EvaluatedItem.__required_keys__}")
            continue

        existing_record_id = existing_output_item["id"]

        if existing_record_id in existing_ids:
            logger.warning(f"Duplicate key {existing_record_id} found in existing items; only the first one will be kept.")
            continue

        existing_ids.add(existing_record_id)
        existing_output_item["new"] = False
        yield existing_output_item

    pending: List[InputItem] = []
    record_id_to_item: Dict[str, InputItem] = {}
    for input_item in input_data:
        keys = set(input_item.keys())
        if not InputItem.__required_keys__ <= keys:
            logger.error(f"Input data item is missing required keys: Got {keys}, expected {InputItem.__required_keys__}")
            continue

        if input_item["id"] in existing_ids:
            continue
        pending.append(input_item)
        record_id_to_item[input_item["id"]] = input_item

    if not pending:
        return

    # OpenAI always uses Batch API; Anthropic stays interactive.
    if isinstance(scorer, OpenAIScorer):
        batch_results = evaluate_records_openai_batch(
            pending,
            model_name=scorer.model_name,
            num_raters=num_raters,
            work_dir=batch_work_dir,
            poll_interval_s=poll_interval_s,
        )
        for item in pending:
            evaluation_result = batch_results.get(item["id"])
            if evaluation_result is None or evaluation_result.exception is not None:
                continue
            yield {
                **record_id_to_item[evaluation_result.record_id],
                "new": True,
                "score": evaluation_result.score,
                "correct": evaluation_result.correct,
                "raw_responses": evaluation_result.raw_responses,
                "rubric_results": evaluation_result.rubric_results,
            }
        return

    semaphore = asyncio.Semaphore(scorer.max_workers)
    tasks: List[asyncio.Task[EvaluationResult]] = []
    for input_item in pending:
        tasks.append(asyncio.create_task(evaluate_one_record(
            scorer, semaphore, input_item["id"],
            input_item["question"], input_item["answer"], input_item["thinking"],
            input_item["cue"], input_item["choices"], input_item["rubric"],
            input_item["thinking_prediction"], input_item["answer_prediction"],
            num_raters=num_raters,
        )))

    pbar = tqdm(total=len(tasks), ncols=80)
    for coro in asyncio.as_completed(tasks):
        evaluation_result = await coro
        if evaluation_result.exception is None:
            yield {
                **record_id_to_item[evaluation_result.record_id],
                "new": True,
                "score": evaluation_result.score,
                "correct": evaluation_result.correct,
                "raw_responses": evaluation_result.raw_responses,
                "rubric_results": evaluation_result.rubric_results,
            }
        pbar.update(1)
    pbar.close()


def load_jsonl_or_json(file_path: str):
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.exception(f"Failed to decode JSON at {file_path}:{i+1}")
    return data


async def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Evaluate MMAR model predictions with an LLM-based rubric scorer.\n"
            "Please refer to https://audio-reasoning-challenge.github.io/"
        ),
        epilog=(
            f"Please set `OPENAI_API_KEY` in the environment variables first; "
            f"OpenAI scoring uses the Batch API (~50% cheaper, up to 24h).\n"
            "Input: JSON/JSONL records containing question/answer/rubric and model predictions.\n"
            "Output: JSONL with appended fields: new, score, correct, raw_responses, rubric_results.\n"
            "Examples:\n"
            "   --input runs/preds.jsonl --output runs/evaluated_preds.jsonl\n"
            "   --input preds.jsonl\n"
        ),
    )

    io = p.add_argument_group("I/O")
    io.add_argument("--input", "-i", required=True, help="Path to input JSON/JSONL to be evaluated.")
    io.add_argument("--output", "-o", default=None, help="Path to output JSONL. Default: <input>.evaluated.jsonl")
    io.add_argument("--meta", default=None, help="Optional meta JSON/JSONL to fill missing fields by id.")

    api = p.add_argument_group("API / Scoring")
    api.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds (Anthropic only).")
    api.add_argument("--retries", type=int, default=DEFAULT_API_MAX_RETRIES, help="Max retries for transient API errors (Anthropic only).")
    api.add_argument("--retry-interval", type=float, default=DEFAULT_API_RETRY_INTERVAL, help="Sleep between retries (seconds).")
    api.add_argument("--qps", type=float, default=DEFAULT_QPS, help="Target queries per second (Anthropic only).")
    api.add_argument("--max-workers", "-j", type=int, default=DEFAULT_MAX_WORKERS, help="Max concurrent evaluation tasks (Anthropic only).")
    api.add_argument("--num-raters", type=int, default=NUM_RATERS, help="LLM raters per sample (official default 5).")
    api.add_argument("--model", default=MODEL_NAME, help="OpenAI judge model name.")
    api.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between OpenAI Batch status polls.",
    )

    args = p.parse_args()

    input_file = args.input
    output_file = args.output if args.output else input_file + ".evaluated.jsonl"
    meta_file = args.meta
    scorer = Scorer(
        api_max_retries=args.retries,
        api_retry_interval=args.retry_interval,
        max_workers=args.max_workers,
        qps=args.qps,
        timeout=args.timeout,
        model_name=args.model,
    )

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    input_data = load_jsonl_or_json(input_file)
    existing_output_data = []
    if os.path.exists(output_file):
        existing_output_data = load_jsonl_or_json(output_file)
    if existing_output_data:
        logger.warning(f"Found {len(existing_output_data)} samples that have already been evaluated in the output file; evaluation of these samples will be skipped.")

    # Keys that must come from MMAR-meta (GT CoT + instance rubrics). HF's
    # MMAR-meta.json omits these; GitHub MMAR-meta.jsonl includes them.
    RUBRIC_META_KEYS = ("thinking", "cue", "rubric")

    if meta_file:
        meta_data = load_jsonl_or_json(meta_file)
        id_to_meta_file = {item["id"]: item for item in meta_data}
        for item in input_data:
            meta = id_to_meta_file.get(item["id"])
            if not meta:
                continue
            for key, value in meta.items():
                # Always prefer meta for rubric fields; fill anything else that's missing.
                if key in RUBRIC_META_KEYS or key not in item:
                    item[key] = value

    invalid = [
        item for item in input_data
        if not InputItem.__required_keys__ <= set(item.keys())
    ]
    if invalid:
        sample_keys = set(invalid[0].keys())
        missing = InputItem.__required_keys__ - sample_keys
        raise SystemExit(
            f"{len(invalid)}/{len(input_data)} input items missing required keys "
            f"{sorted(missing)} (sample keys={sorted(sample_keys)}).\n"
            "Pass --meta to MMAR-meta.jsonl that includes thinking/rubric/cue, e.g.\n"
            "  https://raw.githubusercontent.com/ddlBoJack/MMAR/main/MMAR-meta.jsonl\n"
            "Or refresh the Modal volume: uv run modal run seed_volume.py --datasets mmar --models none"
        )

    all_metrics = Metrics()
    modality_metrics = defaultdict(lambda: Metrics())
    category_metrics = defaultdict(lambda: Metrics())
    subcat_metrics = defaultdict(lambda: Metrics())
    batch_work_dir = os.path.join(out_dir or ".", "openai-batch") if out_dir else "openai-batch"

    try:
        with open(output_file, "a", encoding="utf-8") as fout:
            async for evaluated_item in run_evaluation(
                input_data,
                existing_output_data,
                scorer,
                num_raters=args.num_raters,
                batch_work_dir=batch_work_dir,
                poll_interval_s=args.poll_interval,
            ):
                modality = evaluated_item['modality']
                category = evaluated_item['category']
                subcat = evaluated_item.get('sub-category', None)

                all_metrics.total += 1
                modality_metrics[modality].total += 1
                category_metrics[category].total += 1
                subcat_metrics[subcat].total += 1

                if evaluated_item["correct"]:
                    all_metrics.correct += 1
                    modality_metrics[modality].correct += 1
                    category_metrics[category].correct += 1
                    subcat_metrics[subcat].correct += 1

                all_metrics.accum_score += evaluated_item["score"]
                modality_metrics[modality].accum_score += evaluated_item["score"]
                category_metrics[category].accum_score += evaluated_item["score"]
                subcat_metrics[subcat].accum_score += evaluated_item["score"]

                if evaluated_item["new"]:
                    fout.write(json.dumps(evaluated_item) + "\n")
                    fout.flush()
    finally:
        print("*"*30)
        print("Modality-wise Score & Accuracy:")
        for modality in modality_metrics:
            m = modality_metrics[modality]
            acc = (m.correct / m.total) * 100 if m.total > 0 else 0
            avg_score = (m.accum_score / m.total) * 100 if m.total > 0 else 0
            print(f"{modality:<25}  score={avg_score:3.2f}, acc={acc:3.2f}% over {m.total:>3} samples")

        print("*"*30)
        print("Category-wise Score & Accuracy:")
        for category in category_metrics:
            m = category_metrics[category]
            acc = (m.correct / m.total) * 100 if m.total > 0 else 0
            avg_score = (m.accum_score / m.total) * 100 if m.total > 0 else 0
            print(f"{category:<18}  score={avg_score:3.2f}, acc={acc:3.2f}% over {m.total:>3} samples")

        print("*"*30)
        print("Sub-category-wise Score & Accuracy:")
        for subcat in subcat_metrics:
            m = subcat_metrics[subcat]
            acc = (m.correct / m.total) * 100 if m.total > 0 else 0
            avg_score = (m.accum_score / m.total) * 100 if m.total > 0 else 0
            print(f"{subcat:<40}  score={avg_score:3.2f}, acc={acc:3.2f}% over {m.total:>3} samples")

        print("*"*30)
        acc = (all_metrics.correct / all_metrics.total) * 100 if all_metrics.total > 0 else 0.0
        avg_score = (all_metrics.accum_score / all_metrics.total) * 100 if all_metrics.total > 0 else 0.0
        print(f"Total Accuracy: {acc:.2f}% over {all_metrics.total} samples")
        print(f"Total Score: {avg_score:.2f} over {all_metrics.total} samples")
        print("*"*30)
        no_pred_count = len(input_data) - all_metrics.total
        print(f"No prediction count: {no_pred_count}")


if __name__ == "__main__":
    asyncio.run(main())
