import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

import httpx
import torch
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from utils import adjust_pauses_for_hf_pipeline_output

def convert_to_wav(input_path: str, output_path: str):
    """Convert any audio format to 16kHz mono WAV using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        check=True,
        capture_output=True,
    )


ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
MODEL_ID = os.environ.get("MODEL_ID", "nyrahealth/CrisperWhisper")

pipe = None


def timestamps_to_vtt(timestamps: List[Dict[str, Union[str, Any]]]) -> str:
    """Convert timestamps to VTT format."""
    vtt_content: str = "WEBVTT\n\n"
    for word in timestamps:
        start_time, end_time = word["timestamp"]
        start_time_str = f"{int(start_time // 3600)}:{int(start_time // 60 % 60):02d}:{start_time % 60:06.3f}"
        end_time_str = f"{int(end_time // 3600)}:{int(end_time // 60 % 60):02d}:{end_time % 60:06.3f}"
        vtt_content += f"{start_time_str} --> {end_time_str}\n{word['text']}\n\n"
    return vtt_content


def load_pipeline():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model {MODEL_ID} on {device}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    model.generation_config.median_filter_width = 3
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Model loaded.")
    return asr_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    pipe = load_pipeline()
    yield


app = FastAPI(lifespan=lifespan)


def verify_token(authorization: str | None):
    if not ACCESS_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization[7:] != ACCESS_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid access token")


@app.post("/transcribe")
async def transcribe(
    request: Request,
    audio: UploadFile | None = File(None),
    authorization: str | None = Header(None),
):
    verify_token(authorization)

    audio_bytes: bytes | None = None

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type and audio is not None:
        audio_bytes = await audio.read()
    elif "application/json" in content_type:
        body = await request.json()
        audio_url = body.get("audio_url")
        if not audio_url:
            raise HTTPException(status_code=400, detail="Missing audio_url in JSON body")
        async with httpx.AsyncClient() as client:
            resp = await client.get(audio_url, follow_redirects=True, timeout=120)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download audio from URL (status {resp.status_code})")
            audio_bytes = resp.content
    else:
        raise HTTPException(status_code=400, detail="Send multipart/form-data with an audio file or application/json with audio_url")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data received")

    # Write to temp file for the pipeline
    suffix = ".wav"
    if audio and audio.filename:
        ext = os.path.splitext(audio.filename)[1]
        if ext:
            suffix = ext

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as wav_tmp:
            convert_to_wav(tmp.name, wav_tmp.name)
            result = pipe(wav_tmp.name, return_timestamps="word")

    result = adjust_pauses_for_hf_pipeline_output(result)
    vtt = timestamps_to_vtt(result["chunks"])
    return PlainTextResponse(content=vtt, media_type="text/vtt")


DEBUG_HTML = """<!DOCTYPE html>
<html>
<head><title>CrisperWhisper</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
  label { display: block; margin-top: 16px; font-weight: 600; }
  input, button, textarea { margin-top: 4px; font-size: 14px; }
  input[type=text], input[type=file] { width: 100%; padding: 8px; box-sizing: border-box; }
  button { padding: 8px 20px; cursor: pointer; }
  pre { background: #f4f4f4; padding: 16px; white-space: pre-wrap; overflow-x: auto; min-height: 60px; }
</style>
</head>
<body>
<h1>CrisperWhisper Transcribe</h1>
<form id="form">
  <label>Audio File</label>
  <input type="file" id="audioFile" accept="audio/*">

  <label>OR Audio URL</label>
  <input type="text" id="audioUrl" placeholder="https://example.com/audio.wav">

  <label>Access Token</label>
  <input type="text" id="token" placeholder="Bearer token">

  <button type="submit" style="margin-top:16px">Transcribe</button>
</form>

<h2>Result</h2>
<pre id="result">—</pre>

<script>
document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const resultEl = document.getElementById('result');
  resultEl.textContent = 'Processing...';
  const token = document.getElementById('token').value;
  const file = document.getElementById('audioFile').files[0];
  const url = document.getElementById('audioUrl').value;
  const headers = {};
  if (token) headers['Authorization'] = 'Bearer ' + token;

  try {
    let resp;
    if (file) {
      const fd = new FormData();
      fd.append('audio', file);
      resp = await fetch('/transcribe', { method: 'POST', headers, body: fd });
    } else if (url) {
      headers['Content-Type'] = 'application/json';
      resp = await fetch('/transcribe', { method: 'POST', headers, body: JSON.stringify({ audio_url: url }) });
    } else {
      resultEl.textContent = 'Please provide an audio file or URL.';
      return;
    }
    if (!resp.ok) {
      const err = await resp.text();
      resultEl.textContent = 'Error ' + resp.status + ': ' + err;
      return;
    }
    resultEl.textContent = await resp.text();
  } catch (err) {
    resultEl.textContent = 'Error: ' + err.message;
  }
});
</script>
</body>
</html>"""


@app.get("/")
async def index():
    if not DEBUG:
        return PlainTextResponse(content="CrisperWhisper API is running.", media_type="text/plain")
    return HTMLResponse(content=DEBUG_HTML)
