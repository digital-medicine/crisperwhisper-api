# CrisperWhisper API Docker

A FastAPI microservice for automatic speech recognition with word-level timestamps, built on [Nyra Health's CrisperWhisper](https://huggingface.co/nyrahealth/CrisperWhisper) — a fine-tuned variant of OpenAI's Whisper optimized for precise word boundaries and pause handling.

The service accepts audio in any format (WAV, MP3, M4A, OGG, FLAC, WebM, etc.), transcribes it, and returns word-level timestamps in WebVTT subtitle format.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with drivers installed (+ [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- A [HuggingFace](https://huggingface.co/) account with access to the `nyrahealth/CrisperWhisper` model

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd CrisperWhisper
   ```

2. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   HF_TOKEN=hf_your_huggingface_token
   ACCESS_TOKEN=your_secret_api_key
   DEBUG=false
   ```

   | Variable | Required | Default | Description |
   |----------|----------|---------|-------------|
   | `HF_TOKEN` | Yes | — | HuggingFace token to download the model |
   | `ACCESS_TOKEN` | No | (empty) | Bearer token for API auth. If empty, auth is disabled |
   | `DEBUG` | No | `false` | Set to `true` to enable a browser-based test UI at `GET /` |
   | `MODEL_ID` | No | `nyrahealth/CrisperWhisper` | HuggingFace model ID to load |

3. **Build and run**

   ```bash
   docker compose up --build
   ```

   On first launch the model weights are downloaded from HuggingFace and cached in a Docker volume (`hf-cache`), so subsequent starts are much faster.

## API

### `POST /transcribe`

Transcribe audio and return WebVTT subtitles with word-level timestamps.

**Authentication:** Include `Authorization: Bearer <ACCESS_TOKEN>` if `ACCESS_TOKEN` is set.

#### Option A — File upload (multipart/form-data)

```bash
curl -X POST \
  -H "Authorization: Bearer your_secret_api_key" \
  -F "audio=@recording.mp3" \
  http://localhost:8000/transcribe
```

#### Option B — Remote URL (application/json)

```bash
curl -X POST \
  -H "Authorization: Bearer your_secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/audio.wav"}' \
  http://localhost:8000/transcribe
```

#### Response

```
WEBVTT

0:00:00.000 --> 0:00:00.540
Hello

0:00:00.540 --> 0:00:01.080
world
```

### `GET /`

Returns `"CrisperWhisper API is running."` by default. When `DEBUG=true`, serves an interactive HTML form for testing transcription in the browser.

## How It Works

1. Audio is received (file upload or URL download)
2. Converted to 16 kHz mono WAV via `ffmpeg`
3. Fed into the CrisperWhisper ASR pipeline (30 s chunks, word-level timestamps)
4. Pause durations between words are redistributed for natural subtitle timing
5. Timestamps are formatted as WebVTT and returned

## Project Structure

```
├── server.py                  # FastAPI app, /transcribe endpoint, model loading
├── utils.py                   # Pause adjustment for word-level timestamps
├── Dockerfile                 # CUDA-enabled container image
├── docker-compose.yml         # Service definition with GPU reservation
└── requirements-docker.txt    # Python dependencies
```
