"""
Clip Picker — self-hosted server
---------------------------------
Upload a video, describe what you want, get real clips back.

Pipeline:
  1. faster-whisper transcribes the video's audio (choose your own model size).
  2. The transcript + your request go to a local Ollama model, which reasons
     about which moments actually match and returns timestamps.
  3. ffmpeg cuts those exact moments into real, downloadable video files.

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

import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Clip Picker")

print(f"[clip-picker] loading whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE})...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print("[clip-picker] whisper model ready.")

# In-memory job store. Fine for a single-user, self-hosted tool.
# Restarting the server clears it (files on disk under jobs/ stay until you delete them).
jobs = {}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not ffmpeg_available():
        raise HTTPException(500, "ffmpeg is not installed / not on PATH. Install it and restart the server.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = job_dir / f"source{suffix}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        segments_iter, info = whisper_model.transcribe(str(video_path), beam_size=5)
        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments_iter
        ]
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

    jobs[job_id] = {
        "video_path": str(video_path),
        "segments": segments,
        "duration": info.duration,
    }

    return {"job_id": job_id, "duration": info.duration, "segment_count": len(segments)}


@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found (server may have restarted).")
    return FileResponse(job["video_path"])


@app.post("/api/find-clips")
async def find_clips(job_id: str = Form(...), query: str = Form(...), count: int = Form(3)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found (server may have restarted).")
    segments = job["segments"]
    if not segments:
        raise HTTPException(400, "No speech was found in this video to search over.")

    transcript_lines = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
    prompt = f"""You are selecting the best moments from a video transcript for a highlight reel.

Transcript (timestamps in seconds, from the actual video):
{transcript_lines}

Viewer's request: "{query}"

Pick the {count} best non-overlapping moments that match the request. Use the real timestamps \
from the transcript above — don't invent times. Prefer moments that work as standalone clips \
(roughly 2 to 45 seconds; merge adjacent lines if a moment needs more than one).

Respond with ONLY a JSON array, no other text, no markdown fences, in exactly this form:
[{{"start": 12.3, "end": 18.1, "reason": "one short phrase"}}]"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
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

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise HTTPException(502, "The model didn't return a usable list. Try again, or try a different OLLAMA_MODEL.")

    try:
        picked = json.loads(match.group(0))
    except json.JSONDecodeError:
        raise HTTPException(502, "Couldn't parse the model's response as JSON.")

    duration = job["duration"]
    clips = []
    for p in picked:
        try:
            start = max(0.0, float(p["start"]))
            end = min(duration, float(p["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            clips.append({"start": round(start, 2), "end": round(end, 2), "reason": str(p.get("reason", ""))[:200]})

    if not clips:
        raise HTTPException(502, "The model didn't return any valid clips. Try rephrasing your request.")

    return {"clips": clips}


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
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
    })


app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
