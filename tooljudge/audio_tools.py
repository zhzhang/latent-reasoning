"""Perception tools the judge can call. All CPU-only."""
from __future__ import annotations
import functools, threading, numpy as np, librosa
# librosa lazy-loads submodules: a bare `import librosa` leaves
# librosa.feature.rhythm unresolvable, so tempo() would raise AttributeError and the
# try/except below would silently report nan. Import it explicitly.
import librosa.feature.rhythm

SR = 16000
_WHISPER = None
# Guards both the lazy WhisperModel construction and the decode. lru_cache does not
# serialise concurrent misses, and faster-whisper is not documented as thread-safe.
_WHISPER_LOCK = threading.Lock()


def _load(path: str):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y, sr


@functools.lru_cache(maxsize=512)
def audio_overview(path: str) -> str:
    """Duration, loudness envelope, silence regions. Cheap orientation call."""
    y, sr = _load(path)
    dur = len(y) / sr
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    db = librosa.amplitude_to_db(rms, ref=np.max)
    quiet = db < -35
    spans, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = t[i]
        elif not q and start is not None:
            if t[i] - start > 0.25:
                spans.append((round(start, 2), round(t[i], 2)))
            start = None
    if start is not None and t[-1] - start > 0.25:
        spans.append((round(start, 2), round(t[-1], 2)))
    step = max(1, len(db) // 12)
    curve = [(round(t[i], 2), round(float(db[i]), 1)) for i in range(0, len(db), step)]
    return (
        f"duration={dur:.2f}s sample_rate={sr}\n"
        f"loudness_dB_over_time={curve}\n"
        f"silent_spans={spans if spans else 'none'}"
    )


@functools.lru_cache(maxsize=512)
def transcribe(path: str) -> str:
    """Speech transcript with timestamps. Empty if no speech."""
    global _WHISPER
    # transcribe() returns a lazy generator, so decoding happens on iteration:
    # the comprehension over segs must stay inside the lock, not just the load.
    with _WHISPER_LOCK:
        if _WHISPER is None:
            from faster_whisper import WhisperModel
            _WHISPER = WhisperModel("base", device="cpu", compute_type="int8")
        segs, info = _WHISPER.transcribe(path, vad_filter=True)
        lines = [f"[{s.start:.2f}-{s.end:.2f}] {s.text.strip()}" for s in segs]
    if not lines:
        return "no speech detected"
    return f"language={info.language} (p={info.language_probability:.2f})\n" + "\n".join(lines)


@functools.lru_cache(maxsize=512)
def tempo_and_pitch_over_time(path: str) -> str:
    """Windowed tempo and median pitch. Use for speed, pitch-shift, or
    playback-rate questions: both rising together means the clip was sped up."""
    y, sr = _load(path)
    dur = len(y) / sr
    n = max(2, min(6, int(dur // 1.5)))
    edges = np.linspace(0, len(y), n + 1).astype(int)
    rows = []
    for i in range(n):
        seg = y[edges[i]:edges[i + 1]]
        if len(seg) < sr // 4:
            continue
        try:
            # librosa.beat.tempo is a deprecated alias, removed at librosa 1.0.
            tempo = float(librosa.feature.rhythm.tempo(y=seg, sr=sr)[0])
        except Exception:
            tempo = float("nan")
        f0 = librosa.yin(seg, fmin=50, fmax=2000, sr=sr)
        f0 = f0[np.isfinite(f0)]
        pitch = float(np.median(f0)) if len(f0) else float("nan")
        cent = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
        rows.append(
            f"[{edges[i]/sr:.2f}-{edges[i+1]/sr:.2f}s] "
            f"tempo={tempo:.1f}bpm median_pitch={pitch:.1f}Hz centroid={cent:.0f}Hz"
        )
    return "\n".join(rows) + (
        "\nInterpretation: tempo and pitch rising together indicates faster playback; "
        "both falling indicates slower playback. Tempo alone changing is a musical change."
    )


@functools.lru_cache(maxsize=512)
def event_timeline(path: str) -> str:
    """Onset times and per-onset spectral character. Use for counting events,
    ordering them, or identifying when the character of the sound changes."""
    y, sr = _load(path)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(onsets) == 0:
        return "no discrete onsets detected"
    rows = []
    for t0 in onsets[:40]:
        a = int(t0 * sr)
        b = min(len(y), a + int(0.3 * sr))
        seg = y[a:b]
        if len(seg) < 256:
            continue
        cent = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
        flat = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
        rows.append(f"t={t0:.2f}s centroid={cent:.0f}Hz flatness={flat:.3f}")
    return f"n_onsets={len(onsets)}\n" + "\n".join(rows) + (
        "\nHigh flatness means noise-like (percussion, applause, noise); "
        "low flatness means tonal (voice, melodic instrument)."
    )


TOOLS = {
    "audio_overview": audio_overview,
    "transcribe": transcribe,
    "tempo_and_pitch_over_time": tempo_and_pitch_over_time,
    "event_timeline": event_timeline,
}

SCHEMAS = [
    {"name": "audio_overview",
     "description": "Duration, loudness over time, and silent spans. Call this first to orient.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "transcribe",
     "description": "Transcribe speech with timestamps. Returns 'no speech detected' for non-speech clips.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "tempo_and_pitch_over_time",
     "description": "Windowed tempo and pitch. Use for playback speed, pitch shift, or tempo change questions.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "event_timeline",
     "description": "Onset times with spectral character. Use for counting, ordering, or characterising sound events.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
]
