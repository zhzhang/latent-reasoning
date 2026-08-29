"""Reference-free judge with optional tool access. RQ4 condition switch.

Both conditions are text-only. The Messages API has no audio content block, so the
judge never receives the clip directly; in the tools condition the four local
analysis tools are its only route to the audio.
"""
from __future__ import annotations
import re
from audio_tools import TOOLS, SCHEMAS

MODEL = "claude-sonnet-5"  # match the oracle run exactly
MAX_TOOL_TURNS = 6
# 1200 truncated the verdict on shots that called 3-4 tools: thinking bills against
# this budget and the tool results are bulky, so the turn ended before "VERDICT:".
# 4000 still truncated one shot in 25. Raised rather than trimming tool output, which
# must stay byte-identical across every arm of the experiment.
MAX_TOKENS = 6000
# claude-sonnet-5 rejects sampling parameters, so this arm runs on the model's
# default sampling and is NOT deterministic. Recorded as null in the manifest.
TEMPERATURE = None
N_SAMPLES = 1  # one verdict per shot; the oracle run used 3

SYSTEM = """You independently verify whether another model's answer to a question about an audio clip is correct.

You are NOT given a reference answer.

Rules:
- Do not assume the answer is correct. Verify each factual claim it makes.
- An answer fails if any hard factual claim is wrong, even when the main point is right.
- An answer fails if it hedges across several possibilities and any of them is wrong.
- Wording differences that a reasonable listener would accept are fine.
- Assume the answering model can hear perfectly, so do not excuse errors on difficulty.

Finish with exactly:
VERDICT: PASS or FAIL
CONFIDENCE: 0.0 to 1.0
REASON: one sentence"""

NO_TOOLS_SYSTEM = SYSTEM + """

You cannot hear the clip and have no tools for inspecting it. Judge from the question
and the answer text alone."""

TOOL_SYSTEM = SYSTEM + """

You cannot hear the clip. The analysis tools below are your only access to it. Call the
ones relevant to the question before deciding, and ground your verdict in what they
return rather than in what sounds plausible."""


def _user_text(question: str, answer: str) -> str:
    return (f"QUESTION ASKED OF THE OTHER MODEL:\n{question}\n\n"
            f"ITS ANSWER:\n{answer}\n\nIs this answer correct?")


def parse(text: str) -> dict:
    v = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.I)
    c = re.search(r"CONFIDENCE:\s*([0-9.]+)", text, re.I)
    r = re.search(r"REASON:\s*(.+)", text, re.I)
    return {
        "verdict": v.group(1).upper() if v else None,
        "pass": (v.group(1).upper() == "PASS") if v else None,
        "confidence": float(c.group(1)) if c else None,
        "reason": r.group(1).strip() if r else None,
        "parsed": v is not None,
    }


def judge(client, audio_path: str, question: str, answer: str,
          use_tools: bool) -> dict:
    """One verdict. use_tools toggles the RQ4 intervention.

    audio_path is never sent to the API; it is only the handle the local tools
    analyse, and is unused when use_tools is False.
    """
    messages = [{"role": "user",
                 "content": [{"type": "text", "text": _user_text(question, answer)}]}]

    # No temperature/top_p/top_k: claude-sonnet-5 returns 400 on sampling params.
    kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS,
                  system=TOOL_SYSTEM if use_tools else NO_TOOLS_SYSTEM)
    if use_tools:
        kwargs["tools"] = SCHEMAS

    calls = []
    # Input is re-sent every turn, so summing across turns is the billed total.
    tok_in = tok_out = turns = 0
    for _ in range(MAX_TOOL_TURNS if use_tools else 1):
        resp = client.messages.create(messages=messages, **kwargs)
        turns += 1
        u = getattr(resp, "usage", None)
        tok_in += getattr(u, "input_tokens", 0) or 0
        tok_out += getattr(u, "output_tokens", 0) or 0
        if resp.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for blk in resp.content:
            if blk.type != "tool_use":
                continue
            calls.append(blk.name)
            try:
                out = TOOLS[blk.name](audio_path)
                err = False
            except Exception as exc:
                out, err = f"tool failed: {exc}", True
            results.append({"type": "tool_result", "tool_use_id": blk.id,
                            "content": out, "is_error": err})
        messages.append({"role": "user", "content": results})

    text = "".join(b.text for b in resp.content if b.type == "text")
    out = parse(text)
    out.update(raw=text, tool_calls=calls, n_tool_calls=len(calls),
               stop_reason=resp.stop_reason, api_turns=turns,
               input_tokens=tok_in, output_tokens=tok_out)
    return out
