"""
Clip Picker — self-hosted server
---------------------------------
Upload a video, describe what you want, get real clips back.

Pipeline:
  1. faster-whisper transcribes the video's audio (choose your own model size).
  2. The transcript + your request go to a local Ollama model, which reasons
     about which moments actually match and returns timestamps.
  3. ffmpeg cuts those exact moments into real, downloadable video files.

Everything above runs locally — faster-whisper and Ollama are both models
that run on your own machine, so there is no per-request API cost and no
data ever leaves your computer.

Run:
  pip install -r requirements.txt
  python app.py
  -> http://localhost:8000

Config (environment variables, all optional):
  WHISPER_MODEL     tiny | base | small | medium | large-v3   (default: small)
  WHISPER_DEVICE    cpu | cuda                                (default: cpu)
  WHISPER_COMPUTE   int8 | float16 | float32                  (default: int8)
  OLLAMA_URL        base URL of your Ollama server             (default: http://localhost:11434)
  OLLAMA_MODEL      model name pulled in Ollama                (default: llama3.1)
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel, BatchedInferencePipeline

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
# VAD (voice-activity detection) skips silent/non-speech stretches instead of
# wasting model time transcribing dead air. On by default.
WHISPER_VAD = os.environ.get("WHISPER_VAD", "1") not in ("0", "false", "False")
# Batched inference processes multiple audio chunks at once instead of
# strictly sequentially. Biggest win on GPU; can also help on multi-core CPU.
WHISPER_BATCHED = os.environ.get("WHISPER_BATCHED", "0") not in ("0", "false", "False")
WHISPER_BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
# Re-transcribing the exact same file (by content hash) is skipped entirely.
TRANSCRIPT_CACHE = os.environ.get("TRANSCRIPT_CACHE", "1") not in ("0", "false", "False")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
CACHE_DIR = BASE_DIR / "cache" / "transcripts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Clip Picker")

print(f"[clip-picker] loading whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE})...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
if WHISPER_BATCHED:
    whisper_model = BatchedInferencePipeline(model=whisper_model)
    print(f"[clip-picker] batched inference enabled (batch_size={WHISPER_BATCH_SIZE}).")
print(f"[clip-picker] whisper model ready. (vad_filter={WHISPER_VAD}, cache={TRANSCRIPT_CACHE})")

# In-memory job store. Fine for a single-user, self-hosted tool.
# Restarting the server clears it (files on disk under jobs/ stay until you delete them).
jobs = {}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def merge_segments_into_sentences(segments, max_chunk_seconds: float = 20.0):
    """Whisper's raw segments are often mid-sentence fragments. Merging them
    into full sentences (splitting on sentence-ending punctuation, or after
    max_chunk_seconds if a sentence runs long) gives the LLM cleaner, more
    meaningful units to reason about — it tends to pick noticeably better
    clip boundaries from this than from raw fragments."""
    merged = []
    buf, buf_start, buf_end = [], None, None
    for seg in segments:
        if buf_start is None:
            buf_start = seg["start"]
        buf.append(seg["text"])
        buf_end = seg["end"]
        text_so_far = " ".join(buf).strip()
        if re.search(r'[.!?]["\')]?$', text_so_far) or (buf_end - buf_start) > max_chunk_seconds:
            merged.append({"start": round(buf_start, 2), "end": round(buf_end, 2), "text": text_so_far})
            buf, buf_start = [], None
    if buf:
        merged.append({"start": round(buf_start, 2), "end": round(buf_end, 2), "text": " ".join(buf).strip()})
    return merged


def fit_clip_length(start: float, end: float, target_length, duration: float):
    """Re-center a picked moment to hit a specific target length (seconds),
    clamped to the video's bounds. Used only when the person set a fixed
    clip length instead of AUTO."""
    if not target_length:
        return start, end
    center = (start + end) / 2.0
    half = target_length / 2.0
    new_start, new_end = center - half, center + half
    if new_start < 0:
        new_end += -new_start
        new_start = 0.0
    if new_end > duration:
        new_start -= (new_end - duration)
        new_end = duration
    new_start = max(0.0, new_start)
    return round(new_start, 2), round(new_end, 2)


def hash_file(path: Path) -> str:
    """Content hash of the uploaded file, used as the transcript cache key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_audio(video_path: Path, out_path: Path) -> None:
    """Pull mono 16kHz audio out of the video with ffmpeg before handing it to
    Whisper. Whisper resamples to this format internally anyway, so decoding
    it once with ffmpeg (fast, hardware-accelerated where available) instead
    of leaving Whisper to demux the whole video container is generally
    quicker, especially on large video files with heavy video streams."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-800:]}")


def load_cached_transcript(file_hash: str):
    cache_path = CACHE_DIR / f"{file_hash}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_cached_transcript(file_hash: str, segments, duration: float) -> None:
    cache_path = CACHE_DIR / f"{file_hash}.json"
    try:
        cache_path.write_text(json.dumps({"segments": segments, "duration": duration}))
    except OSError:
        pass


def transcribe_audio(audio_path: Path):
    """Run faster-whisper with VAD filtering and (optionally) batched
    inference, and return plain segment dicts + duration."""
    kwargs = {"beam_size": 5, "vad_filter": WHISPER_VAD}
    if WHISPER_VAD:
        kwargs["vad_parameters"] = {"min_silence_duration_ms": 500}
    if WHISPER_BATCHED:
        kwargs["batch_size"] = WHISPER_BATCH_SIZE

    segments_iter, info = whisper_model.transcribe(str(audio_path), **kwargs)
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments_iter
    ]
    return segments, info.duration


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not ffmpeg_available():
        raise HTTPException(500, "ffmpeg is not installed / not on PATH. Install it and restart the server.")

    t0 = time.monotonic()
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = job_dir / f"source{suffix}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_hash = hash_file(video_path) if TRANSCRIPT_CACHE else None
    cached = load_cached_transcript(file_hash) if file_hash else None

    if cached:
        segments = cached["segments"]
        duration = cached["duration"]
    else:
        audio_path = job_dir / "audio.wav"
        try:
            extract_audio(video_path, audio_path)
            segments, duration = transcribe_audio(audio_path)
        except Exception as e:
            raise HTTPException(500, f"Transcription failed: {e}")
        finally:
            # Only the source video is needed after this point.
            audio_path.unlink(missing_ok=True)

        if file_hash:
            save_cached_transcript(file_hash, segments, duration)

    jobs[job_id] = {
        "video_path": str(video_path),
        "segments": segments,
        "duration": duration,
    }

    elapsed = round(time.monotonic() - t0, 2)
    return {
        "job_id": job_id,
        "duration": duration,
        "segment_count": len(segments),
        "cached": bool(cached),
        "elapsed_seconds": elapsed,
    }


@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found (server may have restarted).")
    return FileResponse(job["video_path"])


@app.post("/api/find-clips")
async def find_clips(
    job_id: str = Form(...),
    query: str = Form(...),
    count: int = Form(3),
    target_length: str = Form("auto"),
):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found (server may have restarted).")
    segments = job["segments"]
    if not segments:
        raise HTTPException(400, "No speech was found in this video to search over.")

    t0 = time.monotonic()

    # "auto" (or anything non-numeric) lets the model judge each clip's
    # natural length itself. A number pins every clip to roughly that length.
    length_value = None
    if target_length and target_length.strip().lower() != "auto":
        try:
            length_value = max(1.0, float(target_length))
        except ValueError:
            length_value = None

    # Sentence-level chunks reason better than raw (often mid-sentence) Whisper segments.
    sentences = merge_segments_into_sentences(segments)
    transcript_lines = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in sentences)

    length_instruction = (
        f"Prefer moments that work as standalone clips (roughly 2 to 45 seconds; merge "
        f"adjacent lines if a moment needs more than one)."
        if length_value is None else
        f"Each clip should be about {length_value:.0f} seconds long — center it on the "
        f"single best moment, merging adjacent lines if needed to fill that length."
    )

    # Ask for more candidates than needed, each scored, so pick-quality doesn't
    # depend on the model getting the final N (and their overlap) exactly right
    # in one shot — that non-overlap step is instead handled deterministically
    # below. More candidates + explicit scoring consistently surfaces better
    # picks than asking the model for the final answer directly.
    candidate_count = max(count * 3, 8)
    prompt = f"""You are selecting the best moments from a video transcript for a highlight reel.

Transcript (timestamps in seconds, from the actual video; text has been joined into full sentences):
{transcript_lines}

Viewer's request: "{query}"

Think briefly about which moments genuinely fit the request — not just ones that mention \
related words, but ones that actually deliver what the viewer is asking for. Then list up to \
{candidate_count} candidate moments, even ones you're only somewhat confident about. Use only \
real timestamps from the transcript above — never invent times. {length_instruction}

For each candidate, give a "score" from 1-10 for how well it truly matches the request (10 = \
perfect fit, 1 = weak/tenuous) and a one-line "reason".

After your thinking, output a JSON array as the LAST thing in your response, with no markdown \
fences after it, in exactly this form:
[{{"start": 12.3, "end": 18.1, "score": 8, "reason": "one short phrase"}}]"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                # Low temperature: this is a selection/judgment task, not a
                # creative one, so we want the model's most confident picks
                # rather than added randomness.
                "options": {"temperature": 0.2},
            },
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            502,
            f"Couldn't reach Ollama at {OLLAMA_URL} (model '{OLLAMA_MODEL}'). "
            f"Is Ollama running and has the model been pulled? ({e})",
        )

    # The model may "think out loud" before the JSON, so take the LAST
    # [...]-looking block in its response rather than the first.
    start_idx, end_idx = raw.rfind("["), raw.rfind("]")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise HTTPException(502, "The model didn't return a usable list. Try again, or try a different OLLAMA_MODEL.")

    try:
        candidates = json.loads(raw[start_idx:end_idx + 1])
    except json.JSONDecodeError:
        raise HTTPException(502, "Couldn't parse the model's response as JSON.")

    duration = job["duration"]
    scored = []
    for c in candidates:
        try:
            start = max(0.0, float(c["start"]))
            end = min(duration, float(c["end"]))
            score = float(c.get("score", 5))
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            scored.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "score": score,
                "reason": str(c.get("reason", ""))[:200],
            })

    if not scored:
        raise HTTPException(502, "The model didn't return any valid clips. Try rephrasing your request.")

    # Greedily take the highest-scoring candidates, skipping anything that
    # overlaps a clip already picked — deterministic, so pick quality doesn't
    # depend on the model itself reasoning correctly about overlaps. When a
    # fixed target length is set, each candidate is re-centered to that
    # length before the overlap check.
    scored.sort(key=lambda c: c["score"], reverse=True)
    clips = []
    for c in scored:
        if len(clips) >= count:
            break
        start, end = fit_clip_length(c["start"], c["end"], length_value, duration)
        if end <= start:
            continue
        if any(not (end <= taken["start"] or start >= taken["end"]) for taken in clips):
            continue
        clips.append({"start": round(start, 2), "end": round(end, 2), "reason": c["reason"]})
    clips.sort(key=lambda c: c["start"])

    elapsed = round(time.monotonic() - t0, 2)
    return {"clips": clips, "elapsed_seconds": elapsed}


@app.post("/api/export")
async def export_clip(job_id: str = Form(...), start: float = Form(...), end: float = Form(...)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found (server may have restarted).")
    if end <= start:
        raise HTTPException(400, "end must be after start.")

    job_dir = Path(job["video_path"]).parent
    out_name = f"clip_{start:.2f}-{end:.2f}.mp4"
    out_path = job_dir / out_name
    duration = end - start

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", job["video_path"],
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        raise HTTPException(500, f"ffmpeg failed: {result.stderr[-800:]}")

    return FileResponse(out_path, filename=out_name, media_type="video/mp4")


@app.get("/api/health")
def health():
    return JSONResponse({
        "ffmpeg": ffmpeg_available(),
        "whisper_model": WHISPER_MODEL_SIZE,
        "whisper_device": WHISPER_DEVICE,
        "whisper_vad": WHISPER_VAD,
        "whisper_batched": WHISPER_BATCHED,
        "whisper_batch_size": WHISPER_BATCH_SIZE if WHISPER_BATCHED else None,
        "transcript_cache": TRANSCRIPT_CACHE,
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
    })


app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
