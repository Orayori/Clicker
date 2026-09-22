# Clip Picker — self-hosted

Upload a video, describe what you want, get real clips back. Runs entirely on
your own machine: transcription (Whisper), reasoning about what to pick
(a local LLM via Ollama), and cutting (ffmpeg) all happen server-side. Nothing
leaves your computer, and there's no API key or per-use cost.

## What you need installed

1. **Python 3.10+**
2. **ffmpeg** — must be on your PATH.
   - Windows: `winget install ffmpeg` (or download from ffmpeg.org and add to PATH)
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (or your distro's equivalent)
3. **Ollama** — https://ollama.com — install it, then pull a model:
   ```
   ollama pull llama3.1
   ```
   (Any chat-capable model works — `llama3.1`, `qwen2.5`, `mistral`, etc.
   Bigger models reason about "intriguing" or "funny" moments noticeably
   better than small ones, if your machine can run them.)

## Setup

```bash
cd clip-picker-server
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:8000** in your browser.

The first run downloads the Whisper model (size depends on config below) —
that's a one-time download, cached afterward.

## Using a better AI model

Everything is controlled by environment variables — set them before running
`python app.py`:

| Variable          | Default                  | Notes |
|-------------------|---------------------------|-------|
| `WHISPER_MODEL`    | `small`                  | `tiny`, `base`, `small`, `medium`, `large-v3`. Bigger = more accurate transcripts, slower. |
| `WHISPER_DEVICE`   | `cpu`                    | Set to `cuda` if you have an NVIDIA GPU — much faster, lets you comfortably run `medium`/`large-v3`. |
| `WHISPER_COMPUTE`  | `int8`                   | `int8` (fastest on CPU), `float16` (best on GPU), `float32`. |
| `OLLAMA_URL`       | `http://localhost:11434` | Change if Ollama runs elsewhere (another machine, a container). |
| `OLLAMA_MODEL`     | `llama3.1`                | Any model you've pulled with `ollama pull`. |
| `WHISPER_VAD`      | `1`                       | Voice-activity detection: skips silent/non-speech stretches instead of transcribing them. Free speed-up; set to `0` to disable. |
| `WHISPER_BATCHED`  | `0`                       | Batched inference: transcribes multiple audio chunks at once instead of strictly one-by-one. Biggest win on GPU, can help on multi-core CPU too. Set to `1` to enable. |
| `WHISPER_BATCH_SIZE` | `8`                     | Chunk batch size when `WHISPER_BATCHED=1`. Higher uses more memory/VRAM but can be faster. |
| `TRANSCRIPT_CACHE` | `1`                       | Skips re-transcribing a video you've already uploaded before (matched by file content, not filename). Set to `0` to disable. |

### Why uploads are faster now

- Audio is extracted to a small mono 16kHz WAV with `ffmpeg` before transcription, instead of letting Whisper demux the full video container itself — faster for large video files, especially ones with heavy/high-bitrate video streams.
- `WHISPER_VAD=1` (default) skips silent stretches instead of spending model time on them.
- `WHISPER_BATCHED=1` processes several audio chunks in parallel rather than one after another — most effective on GPU.
- Re-uploading the exact same video file reuses the cached transcript instantly instead of re-running Whisper.

Example, running a bigger transcription model on a GPU with a stronger LLM:

```bash
# macOS/Linux
WHISPER_MODEL=medium WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16 OLLAMA_MODEL=llama3.1:70b python app.py

# Windows (PowerShell)
$env:WHISPER_MODEL="medium"; $env:WHISPER_DEVICE="cuda"; $env:WHISPER_COMPUTE="float16"; $env:OLLAMA_MODEL="llama3.1:70b"; python app.py
```

## Running it as an actual always-on website

To keep it running and reachable beyond your own machine:

- **On your LAN**: it already binds to `0.0.0.0:8000`, so `http://<your-computer's-LAN-IP>:8000` works from other devices on your network as-is.
- **Keep it running in the background**: use `pm2`, a systemd service, or Windows Task Scheduler / NSSM to run `python app.py` persistently.
- **Exposed to the internet**: put it behind a reverse proxy (Caddy or nginx) with HTTPS, and add authentication in front of it (this app has none built in — anyone who can reach it can upload videos and use your models). A Cloudflare Tunnel or Tailscale Funnel is an easy way to get a public HTTPS URL without opening ports.

## Notes

- Uploaded videos and cut clips are stored under `jobs/<job-id>/`. Nothing is
  cleaned up automatically — delete old job folders yourself periodically.
- If Ollama isn't running, or the model name doesn't match a pulled model,
  "Find clips" will fail with a clear error rather than hanging silently.
- `/api/health` shows current config and whether ffmpeg was found — useful
  for a quick sanity check after setup.

## How clip picking works, and how to improve it further

"Find clips" now asks the model to score several candidate moments (more
than you asked for) rather than commit to the final answer in one shot, then
picks the highest-scoring ones that don't overlap in code. This tends to
produce noticeably better picks than asking for the final N directly, and
runs at `temperature=0.2` so picks are consistent rather than random. The
transcript it reasons over is also merged into full sentences first, since
raw Whisper segments are often mid-sentence fragments.

Everything here is local (faster-whisper + Ollama), so none of this costs
anything to run or tune. If you want even better judgment:

- **Use a bigger Ollama model.** This is the single biggest lever —
  `ollama pull qwen2.5:32b` (or `llama3.1:70b` if your machine can run it)
  reasons about "funny" or "intriguing" moments noticeably better than an
  8B model. Set `OLLAMA_MODEL` to match.
- **Be specific in your request.** "The funniest moment" gives the model
  less to work with than "the moment where he trips over the dog and everyone
  laughs."
