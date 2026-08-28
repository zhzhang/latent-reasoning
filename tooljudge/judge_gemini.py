"""Reference-free judge on Gemini. Same interface as judge.py, plus audio.

Gemini accepts audio parts, so this module runs the two arms Claude cannot:
audio_only and audio_tools. Prompts, the verdict parser, and the tool
implementations are imported from judge.py and audio_tools.py rather than
duplicated, so the only difference between the Claude and Gemini arms is the
provider and whether the clip is attached.

Reads GEMINI_API_KEY from the environment.
"""
from __future__ import annotations
import functools, mimetypes, os
from pathlib import Path
from google.genai import types
from audio_tools import TOOLS, SCHEMAS
from judge import (SYSTEM, NO_TOOLS_SYSTEM, TOOL_SYSTEM, parse, _user_text,
                   MAX_TOOL_TURNS)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
SUPPORTS_AUDIO = True
# Deliberately not raised with judge.MAX_TOKENS: Gemini never truncated at 4000
# (25/25 finish_reason STOP), so the cap stays where it was measured.
MAX_OUTPUT_TOKENS = 4000

# Gemini still accepts sampling parameters, unlike claude-sonnet-5, so the Gemini
# arms stay deterministic. Deliberately not imported from judge.py, whose
# TEMPERATURE is now None.
TEMPERATURE = 0

# Inline the clip below this; hand anything larger to the Files API. The MMAR
# corpus tops out near 11 MB, so the upload path is a safety net, not the norm.
INLINE_MAX_BYTES = 15 * 1024 * 1024

# judge.py's TOOL_SYSTEM tells the judge it cannot hear the clip, which is true
# for Claude and for the no-audio Gemini arms but not when the audio is attached.
AUDIO_TOOL_SYSTEM = SYSTEM + """

You can hear the clip, and you also have analysis tools for it. Call the ones relevant
to the question before deciding, and prefer what they return over your impression of
the audio where the two disagree."""


def make_client():
    """Gemini client from GEMINI_API_KEY."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY")
    from google import genai
    return genai.Client(api_key=key)


def system_for(use_tools: bool, send_audio: bool) -> str:
    if send_audio:
        return AUDIO_TOOL_SYSTEM if use_tools else SYSTEM
    return TOOL_SYSTEM if use_tools else NO_TOOLS_SYSTEM


def gemini_schemas():
    """audio_tools.SCHEMAS in Gemini function-declaration form.

    Both take plain JSON Schema for parameters, so the conversion is a rename:
    input_schema -> parameters_json_schema.
    """
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=s["name"],
            description=s["description"],
            parameters_json_schema=s["input_schema"],
        ) for s in SCHEMAS
    ])]


@functools.lru_cache(maxsize=512)
def _inline_part(path: str):
    mt = mimetypes.guess_type(path)[0] or "audio/wav"
    return types.Part.from_bytes(data=Path(path).read_bytes(), mime_type=mt)


_UPLOADED: dict[str, tuple[str, str]] = {}


def _audio_part(client, path: str):
    """Cached by audio path, so three shots of one question upload it once."""
    if os.path.getsize(path) <= INLINE_MAX_BYTES:
        return _inline_part(path)
    if path not in _UPLOADED:
        f = client.files.upload(file=path)
        _UPLOADED[path] = (f.uri, f.mime_type or
                           mimetypes.guess_type(path)[0] or "audio/wav")
    uri, mt = _UPLOADED[path]
    return types.Part.from_uri(file_uri=uri, mime_type=mt)


def _parts(resp):
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return []
    content = getattr(cands[0], "content", None)
    return getattr(content, "parts", None) or []


def judge(client, audio_path: str, question: str, answer: str,
          use_tools: bool, send_audio: bool = True) -> dict:
    """One verdict. Returns the same dict shape as judge.judge()."""
    parts = []
    if send_audio:
        parts.append(_audio_part(client, audio_path))
    parts.append(types.Part.from_text(text=_user_text(question, answer)))
    contents = [types.Content(role="user", parts=parts)]

    config = types.GenerateContentConfig(
        system_instruction=system_for(use_tools, send_audio),
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        # We drive the loop ourselves, as judge.py does, so the schemas stay the
        # single source of truth and tool_calls is ours to record.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tools=gemini_schemas() if use_tools else None,
    )

    calls: list[str] = []
    resp = None
    # Input is re-sent every turn, so summing across turns is the billed total.
    tok_in = tok_out = turns = 0
    tok_audio = tok_text = 0
    for _ in range(MAX_TOOL_TURNS if use_tools else 1):
        resp = client.models.generate_content(
            model=MODEL, contents=contents, config=config)
        turns += 1
        u = getattr(resp, "usage_metadata", None)
        tok_in += getattr(u, "prompt_token_count", 0) or 0
        # thoughts are billed as output but reported separately
        tok_out += ((getattr(u, "candidates_token_count", 0) or 0)
                    + (getattr(u, "thoughts_token_count", 0) or 0))
        # Per-modality split of the prompt, so the audio share of the bill is a
        # measurement rather than an inference.
        for d in (getattr(u, "prompt_tokens_details", None) or []):
            mod = str(getattr(d.modality, "value", d.modality) or "").upper()
            n = getattr(d, "token_count", 0) or 0
            if mod == "AUDIO":
                tok_audio += n
            elif mod == "TEXT":
                tok_text += n
        fcs = [p.function_call for p in _parts(resp) if p.function_call]
        if not fcs:
            break
        contents.append(resp.candidates[0].content)
        results = []
        for fc in fcs:
            calls.append(fc.name)
            fn = TOOLS.get(fc.name)
            if fn is None:
                out = f"tool failed: no such tool {fc.name!r}"
            else:
                try:
                    out = fn(audio_path)
                except Exception as exc:
                    out = f"tool failed: {exc}"
            results.append(types.Part.from_function_response(
                name=fc.name, response={"output": out}))
        contents.append(types.Content(role="user", parts=results))

    text = "".join(p.text for p in _parts(resp) if p.text)
    # Gemini's finish_reason is the analogue of Anthropic's stop_reason; MAX_TOKENS
    # here means the same thing as Anthropic's "max_tokens".
    cands = getattr(resp, "candidates", None) or []
    fr = getattr(cands[0], "finish_reason", None) if cands else None
    out = parse(text)
    out.update(raw=text, tool_calls=calls, n_tool_calls=len(calls),
               stop_reason=getattr(fr, "value", fr), api_turns=turns,
               input_tokens=tok_in, output_tokens=tok_out,
               audio_prompt_tokens=tok_audio, text_prompt_tokens=tok_text)
    return out
