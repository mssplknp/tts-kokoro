# Kokoro TTS Server

A self-hosted FastAPI service that turns **complete sentences** into raw
`s16le` PCM audio for the LiveKit voice agent. Synthesis runs on
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (default: Hindi), a
small high-quality neural TTS that runs well on Apple Silicon (MPS) and CPU.

This is a drop-in replacement for the reference `app/tts-server` (Indic Parler)
implementation: it serves the same `POST /v1/tts/stream` contract and outputs
`44100 Hz` mono PCM by default, so the existing LiveKit clients work unchanged.

---

## Why 44100 Hz?

Kokoro natively generates at **24000 Hz**. The LiveKit clients hardcode
`sample_rate=44100` (see `app/livekit_agent/tts/audio_chunker.py`,
`app/livekit_agent/tts/tts_pipeline.py` and
`app/livekit_agent/adapters/external_tts.py`), so the server **resamples
24000 → 44100** before streaming to stay drop-in compatible. If you update the
clients instead, set `SAMPLE_RATE=24000` in `.env` to emit native-rate audio
with no resampling cost.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # optional; defaults work out of the box
python run.py                 # dev server with auto-reload
```

Or run it directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

> Keep `--workers 1`. A single worker owns the model; the server serializes
> inference internally (see *Concurrency* below).

First boot downloads the model (~330 MB) and warms it up, so the very first
request pays no compile/download cost. Everything else is cached in
`~/.cache/huggingface`.

---

## API

### `POST /v1/tts/stream`

Synthesize one sentence (or short batch) and **stream raw `s16le` mono PCM**.

Request (`text` is the only required field):

```json
{
  "text": "नमस्ते, मैं किसान मित्र हूँ।",
  "language": "hi",
  "voice": "hm_psi",
  "speed": 1.0,
  "sample_rate": 44100,
  "request_id": "speech_abc123"
}
```

| Field         | Type    | Notes                                          |
|---------------|---------|------------------------------------------------|
| `text`        | string  | required, non-empty after strip                |
| `language`    | string  | `hi`/`h` (this server's configured language)   |
| `voice`       | string  | one of the configured language's voices        |
| `speed`       | float   | `0 < speed <= 3.0` (default `SPEED`)           |
| `sample_rate` | int     | `8000..192000`; resampled on the fly           |
| `request_id`  | string  | echoed back in `X-Request-Id`                  |

Response: `200 application/octet-stream` (raw PCM), chunked, first bytes as
soon as the first sub-unit is synthesized.

Headers:

| Header             | Value        |
|--------------------|--------------|
| `X-Sample-Rate`    | `44100`      |
| `X-Channels`       | `1`          |
| `X-Sample-Format`  | `s16le`      |
| `X-Request-Id`     | request id   |

### `POST /v1/tts/synthesize`

Non-streaming convenience: same body, returns one complete PCM buffer.

### `GET /v1/tts/health`

Liveness + capacity so the agent can gate on readiness:

```json
{
  "status": "ok",
  "model": "Kokoro-82M",
  "device": "mps",
  "language": "h",
  "native_sample_rate": 24000,
  "sample_rate": 44100,
  "default_voice": "hm_psi",
  "voices": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
  "limit": 4,
  "active": 0,
  "waiting": 0
}
```

### `GET /v1/tts/voices`

Available voices for the configured language.

### Errors

| Status | Condition                                                     |
|--------|---------------------------------------------------------------|
| 400    | empty/missing `text`, bad `language`/`voice`/`speed`/`sample_rate` |
| 429    | concurrency limit reached; wait timed out (retryable)         |
| 500    | provider failure; on `/stream` this truncates the body mid-flight |
| 503    | provider not ready (still warming up)                         |

---

## How it works

### Unit of text

The agent (custom `VoiceSession` pipeline or `AgentSession` + `ExternalTTS`)
sends **one complete sentence per request**. The server splits each sentence
into smaller sub-units at natural pause points (sentence-final
`. ! ? ।`, then clauses on `, ; :` when long enough), capped at ~200 chars.
Each sub-unit is one Kokoro synthesis call, which means:

- **Low time-to-first-byte** — a sub-unit of a few words synthesizes in
  well under a second on MPS; it streams as soon as it is ready.
- **Prompt cancellation** — when the client disconnects, generation stops at
  the next sub-unit boundary instead of finishing the whole sentence.

### Concurrency & ordering

- A `threading.Lock` serializes actual model inference because MPS (and the
  espeak-ng G2P layer) are **not thread-safe**. The lock is taken per
  sub-unit, so concurrent requests interleave fairly instead of blocking each
  other — this is the "generation lock" the agent contract calls for.
- Each sub-unit runs off the event loop via `asyncio.to_thread`; the loop
  never blocks on inference.
- A `ConcurrencyLimiter` bounds in-flight requests (`CONCURRENCY_LIMIT`, default
  4). Extra requests wait up to `QUEUE_WAIT_TIMEOUT` seconds, then get **429**
  so the agent can retry.

### Disconnects

`StreamingResponse` cancels the response task when the client goes away; the
generator's `finally` releases the concurrency slot and stops pulling further
sub-units, freeing the model promptly.

### Warm-up

On startup the server synthesizes `WARMUP_TEXT` with the default voice so the
MPS kernels, the resampler and the voice weights are loaded before the first
live request.

---

## Configuration

All settings are environment variables (see `.env.example`):

| Variable                | Default                  | Meaning                              |
|-------------------------|--------------------------|--------------------------------------|
| `HOST` / `PORT`         | `0.0.0.0` / `8001`       | bind address                         |
| `MODEL_REPO_ID`         | `hexgrad/Kokoro-82M`     | Hugging Face checkpoint              |
| `LANG_CODE`             | `h`                      | `a` (en-us), `b` (en-gb), `h` (hi)…  |
| `DEFAULT_VOICE`         | `hm_psi`                 | default voice                        |
| `DEVICE`                | `auto`                   | `auto` / `cpu` / `mps` / `cuda`      |
| `SPEED`                 | `1.0`                    | default speaking rate                |
| `SAMPLE_RATE`           | `44100`                  | output rate (client contract)        |
| `NATIVE_SAMPLE_RATE`    | `24000`                  | Kokoro's native rate                 |
| `FRAME_SIZE`            | `1764`                   | 20 ms frame @ 44.1 kHz mono s16le    |
| `CONCURRENCY_LIMIT`     | `4`                      | max in-flight requests               |
| `QUEUE_WAIT_TIMEOUT`    | `10.0`                   | seconds before overload → 429        |
| `WARMUP_TEXT`           | `नमस्ते, मैं किसान मित्र हूँ।` | boot-time warm-up phrase        |
| `LOG_LEVEL`             | `INFO`                   | logging level                        |

---

## Development

```bash
pip install -r requirements-dev.txt
pytest                    # unit + API tests (no model required)
```

Run a live smoke test against a running server:

```bash
python scripts/smoke_client.py "नमस्ते किसान भाई, आपकी गेहूँ की फसल कैसी चल रही है?" out.wav
```

## Project layout

```
app/
  main.py          application factory, lifespan, error handlers
  config.py        pydantic-settings configuration
  middleware.py    request-id correlation
  api/tts.py       /stream /synthesize /health /voices
  schemas/tts.py   TTSRequest model + validation
  providers/
    base.py        TTSProvider interface
    kokoro.py      Kokoro backend (streaming, lock, warm-up)
  core/
    text_split.py  sentence/clause sub-unit splitting
    audio.py       float → s16le PCM, 20 ms frame alignment
    resampler.py   torchaudio windowed-sinc resampler
    concurrency.py bounded request limiter (overload → 429)
experiments/       voice-testing scratch scripts
scripts/           smoke client
tests/             unit + API tests (fake provider)
```
