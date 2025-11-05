#!/Users/markbennett/github/mac-live-transcribe/transcribe_compare.py
import asyncio
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets
import tkinter as tk
from tkinter import ttk

# Simple .env loader (no extra dependency)
def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

# Load env before reading config
load_env()

LOCAL_BACKEND = os.environ.get("LOCAL_BACKEND", "faster").lower()  # faster | mlx

# Conditionally import backends
WhisperModel = None
MLXWhisper = None
if LOCAL_BACKEND == "faster":
    from faster_whisper import WhisperModel as FW_WhisperModel  # type: ignore
    WhisperModel = FW_WhisperModel
elif LOCAL_BACKEND == "mlx":
    try:
        import mlx_whisper as MLXWhisper  # type: ignore
    except Exception:
        MLXWhisper = None

# Config
SAMPLE_RATE = 16000
FRAME_MS = 20  # 20 ms frames
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
CHANNELS = 1
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-2-general")

# Mic level visualization config (dBFS mapping)
MIC_LEVEL_DB_MIN = float(os.environ.get("MIC_LEVEL_DB_MIN", "-60"))
MIC_LEVEL_DB_MAX = float(os.environ.get("MIC_LEVEL_DB_MAX", "-12"))
MIC_LEVEL_SMOOTHING = float(os.environ.get("MIC_LEVEL_SMOOTHING", "0.2"))  # 0..1

# Local VAD settings
VAD_DB_SILENCE = float(os.environ.get("VAD_DB_SILENCE", "-45"))
VAD_MIN_SIL_MS = int(os.environ.get("VAD_MIN_SIL_MS", "500"))
UTTERANCE_MIN_MS = int(os.environ.get("UTTERANCE_MIN_MS", "500"))
UTTERANCE_MAX_MS = int(os.environ.get("UTTERANCE_MAX_MS", "7000"))

# Whisper performance knobs (faster-whisper)
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", str(max(4, (os.cpu_count() or 8) - 1))))
WHISPER_NUM_WORKERS = int(os.environ.get("WHISPER_NUM_WORKERS", "1"))
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
WHISPER_BEST_OF = int(os.environ.get("WHISPER_BEST_OF", "1"))
WHISPER_TEMPERATURE = float(os.environ.get("WHISPER_TEMPERATURE", "0.0"))

from urllib.parse import urlencode

@dataclass
class SharedState:
    mic_level_percent: float = 0.0  # 0..100
    local_status: str = ""
    local_lines: list[str] = field(default_factory=list)
    cloud_lines: list[str] = field(default_factory=list)
    is_running: bool = True


def audio_stream_generator(audio_q: "queue.Queue[np.ndarray]", state: SharedState) -> None:
    level_ema = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal level_ema
        mono = indata[:, 0] if indata.ndim == 2 else indata

        # Normalize to -1..1 float and compute RMS
        audio_norm = mono.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio_norm * audio_norm) + 1e-12))
        # Convert to dBFS and map to 0..1 using configured range
        dbfs = 20.0 * np.log10(rms + 1e-12)
        span = max(6.0, MIC_LEVEL_DB_MAX - MIC_LEVEL_DB_MIN)
        level = (dbfs - MIC_LEVEL_DB_MIN) / span
        level = 0.0 if np.isnan(level) else float(np.clip(level, 0.0, 1.0))

        # Smoothing
        alpha = float(np.clip(MIC_LEVEL_SMOOTHING, 0.0, 1.0))
        level_ema = (1 - alpha) * level_ema + alpha * level
        state.mic_level_percent = max(0.0, min(100.0, level_ema * 100.0))

        if state.is_running:
            try:
                audio_q.put_nowait((mono.copy(), time.time()))
            except queue.Full:
                pass

    with sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        while True:
            time.sleep(0.1)


def transcribe_with_faster(model, audio_float: np.ndarray, language: Optional[str]) -> str:
    segments, _ = model.transcribe(
        audio_float,
        language=language,
        beam_size=WHISPER_BEAM_SIZE,
        best_of=WHISPER_BEST_OF,
        temperature=WHISPER_TEMPERATURE,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    return " ".join(s.text for s in segments).strip()


def transcribe_with_mlx(mlx_whisper, audio_float: np.ndarray, language: Optional[str]) -> str:
    # mlx_whisper expects 16 kHz float32 audio in [-1,1]
    # API: mlx_whisper.transcribe(audio_array, path_or_hf_repo="mlx-community/whisper-tiny", language="en")
    name = os.environ.get("MLX_MODEL", os.environ.get("WHISPER_MODEL", "small"))
    repo = name if "/" in name else f"mlx-community/whisper-{name}"
    # Ensure contiguous float32
    audio_float = np.ascontiguousarray(audio_float.astype(np.float32))
    try:
        out = mlx_whisper.transcribe(
            audio_float,
            path_or_hf_repo=repo,
            language=(language or "en"),
            fp16=True,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            verbose=False,
        )
    except Exception:
        # fallback to tiny repo
        out = mlx_whisper.transcribe(
            audio_float,
            path_or_hf_repo="mlx-community/whisper-tiny",
            language=(language or "en"),
            fp16=True,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            verbose=False,
        )
    text = (out.get("text") or "").strip()
    return text


def run_local_whisper(audio_q: "queue.Queue[np.ndarray]", state: SharedState, ui_queue: "queue.Queue[tuple]") -> None:
    # Indicate model loading (first run may download once)
    backend = LOCAL_BACKEND
    state.local_status = f"Local[{backend}]: loading model (first run may download)"
    ui_queue.put_nowait(("status", None))

    language = os.environ.get("WHISPER_LANGUAGE", "en") or None

    # Initialize backend
    model = None
    if LOCAL_BACKEND == "faster":
        model_size = os.environ.get("WHISPER_MODEL", "small")
        device = os.environ.get("WHISPER_DEVICE", "cpu")
        compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=WHISPER_CPU_THREADS,
            num_workers=WHISPER_NUM_WORKERS,
        )
    elif LOCAL_BACKEND == "mlx":
        if MLXWhisper is None:
            state.local_status = "Local[mlx]: mlx-whisper not installed"
            ui_queue.put_nowait(("status", None))
            return
        # MLX downloads model on first run automatically via HF
        pass

    state.local_status = f"Local[{backend}]: model ready"
    ui_queue.put_nowait(("status", None))

    # Utterance buffer
    max_samples = int((UTTERANCE_MAX_MS / 1000.0) * SAMPLE_RATE)
    utterance = np.zeros(0, dtype=np.int16)
    silence_ms = 0

    while True:
        chunk, _ = audio_q.get()
        samples = chunk.astype(np.int16)
        # Append to utterance buffer with cap
        if len(utterance) + len(samples) > max_samples:
            audio_to_decode = utterance
            utterance = np.zeros(0, dtype=np.int16)
            silence_ms = 0
            if len(audio_to_decode) >= int((UTTERANCE_MIN_MS / 1000.0) * SAMPLE_RATE):
                audio_float = (audio_to_decode.astype(np.float32) / 32768.0)
                if LOCAL_BACKEND == "faster":
                    text = transcribe_with_faster(model, audio_float, language)
                else:
                    text = transcribe_with_mlx(MLXWhisper, audio_float, language)
                if text:
                    state.local_lines.append(text)
                    ui_queue.put_nowait(("local_line", text))
            continue
        utterance = np.concatenate((utterance, samples))

        # VAD on this chunk
        audio_norm = samples.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio_norm * audio_norm) + 1e-12))
        dbfs = 20.0 * np.log10(rms + 1e-12)
        if dbfs < VAD_DB_SILENCE:
            silence_ms += FRAME_MS
        else:
            silence_ms = 0

        # If sufficient silence and enough speech collected, finalize
        if silence_ms >= VAD_MIN_SIL_MS and len(utterance) >= int((UTTERANCE_MIN_MS / 1000.0) * SAMPLE_RATE):
            # Drop trailing silence by trimming last silence_ms worth of samples
            trim_samples = int((silence_ms / 1000.0) * SAMPLE_RATE)
            audio_to_decode = utterance[:-trim_samples] if trim_samples < len(utterance) else utterance
            utterance = np.zeros(0, dtype=np.int16)
            silence_ms = 0

            audio_float = (audio_to_decode.astype(np.float32) / 32768.0)
            if LOCAL_BACKEND == "faster":
                text = transcribe_with_faster(model, audio_float, language)
            else:
                text = transcribe_with_mlx(MLXWhisper, audio_float, language)
            if text:
                state.local_lines.append(text)
                ui_queue.put_nowait(("local_line", text))


# Deepgram section unchanged except small refactor
from urllib.parse import urlencode

async def run_deepgram_ws(state: SharedState, ui_queue: "queue.Queue[tuple]") -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        ui_queue.put_nowait(("cloud_line", "[Deepgram] Set DEEPGRAM_API_KEY in .env to see cloud transcript"))
        return

    headers = {
        "Authorization": f"Token {api_key}",
    }
    params = {
        "model": DEEPGRAM_MODEL,
        "encoding": "linear16",
        "sample_rate": str(SAMPLE_RATE),
        "channels": str(CHANNELS),
        "punctuate": "true",
        "interim_results": "true",
        "smart_format": "true",
        "endpointing": "500",
    }

    uri = DEEPGRAM_URL + "?" + urlencode(params)

    async with websockets.connect(uri, additional_headers=headers, ping_interval=20, ping_timeout=20) as ws:
        loop = asyncio.get_event_loop()
        audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

        def dg_callback(indata, frames, time_info, status):
            mono = indata[:, 0] if indata.ndim == 2 else indata
            audio_bytes = mono.astype(np.int16).tobytes()
            if state.is_running:
                try:
                    audio_q.put_nowait(audio_bytes)
                except queue.Full:
                    pass

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=dg_callback,
        )
        stream.start()

        cloud_last_text = ""

        async def sender():
            try:
                while True:
                    data = await loop.run_in_executor(None, audio_q.get)
                    await ws.send(data)
            except Exception:
                pass

        async def receiver():
            nonlocal cloud_last_text
            async for message in ws:
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    continue
                transcript = ""
                is_final = False
                if "channel" in msg and "alternatives" in msg.get("channel", {}):
                    alts = msg["channel"]["alternatives"]
                    if alts:
                        alt0 = alts[0]
                        transcript = alt0.get("transcript", "")
                        is_final = bool(alt0.get("is_final") or msg.get("is_final"))
                elif "results" in msg:
                    results = msg.get("results", {})
                    channels = results.get("channels", [])
                    if channels and channels[0].get("alternatives"):
                        alt0 = channels[0]["alternatives"][0]
                        transcript = alt0.get("transcript", "")
                        is_final = bool(alt0.get("is_final") or results.get("is_final") or msg.get("is_final"))

                if not transcript:
                    continue

                # Append logic: when final or ends with punctuation
                should_commit = is_final or transcript.endswith((".", "?", "!"))
                if should_commit:
                    if transcript.startswith(cloud_last_text):
                        new_suffix = transcript[len(cloud_last_text):].strip()
                    else:
                        new_suffix = transcript
                    if new_suffix:
                        state.cloud_lines.append(new_suffix)
                        ui_queue.put_nowait(("cloud_line", new_suffix))
                    cloud_last_text = transcript

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        await asyncio.gather(sender_task, receiver_task)


def ui_loop(state: SharedState) -> None:
    root = tk.Tk()
    backend = LOCAL_BACKEND
    root.title(f"Live Transcription: Local ({backend}) vs Cloud (Deepgram)")

    container = tk.Frame(root)
    container.pack(fill=tk.BOTH, expand=True)

    # Mic level and status at top
    top_frame = tk.Frame(container)
    top_frame.pack(fill=tk.X, padx=8, pady=6)

    tk.Label(top_frame, text="Mic level").pack(side=tk.LEFT)
    mic_bar = ttk.Progressbar(top_frame, orient=tk.HORIZONTAL, length=300, mode="determinate", maximum=100)
    mic_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    status_label = tk.Label(top_frame, text=f"Local[{backend}]: starting...", anchor="w")
    status_label.pack(side=tk.LEFT, padx=8)

    def toggle_run():
        state.is_running = not state.is_running
        btn.config(text=("Pause" if state.is_running else "Transcribe"))

    btn = ttk.Button(top_frame, text="Pause", command=toggle_run)
    btn.pack(side=tk.RIGHT)

    columns = tk.Frame(container)
    columns.pack(fill=tk.BOTH, expand=True)

    left = tk.Frame(columns)
    right = tk.Frame(columns)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    lbl_local = tk.Label(left, text=f"Local ({backend})")
    lbl_local.pack(anchor="w", padx=8)
    txt_local = tk.Text(left, wrap=tk.WORD, height=30)
    txt_local.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
    txt_local.config(state=tk.DISABLED)

    lbl_cloud = tk.Label(right, text="Cloud (Deepgram)")
    lbl_cloud.pack(anchor="w", padx=8)
    txt_cloud = tk.Text(right, wrap=tk.WORD, height=30)
    txt_cloud.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
    txt_cloud.config(state=tk.DISABLED)

    ui_queue: "queue.Queue[tuple]" = queue.Queue()

    def append_line(widget: tk.Text, line: str):
        widget.config(state=tk.NORMAL)
        if widget.index('end-1c') != '1.0':
            widget.insert(tk.END, "\n")
        widget.insert(tk.END, line)
        widget.see(tk.END)
        widget.config(state=tk.DISABLED)

    def poll():
        # Update mic bar
        mic_bar["value"] = state.mic_level_percent
        # Update status
        if state.local_status:
            status_label.config(text=state.local_status)
        try:
            while True:
                who, payload = ui_queue.get_nowait()
                if who == "local_line" and isinstance(payload, str):
                    append_line(txt_local, payload)
                elif who == "cloud_line" and isinstance(payload, str):
                    append_line(txt_cloud, payload)
                elif who == "status":
                    status_label.config(text=state.local_status)
        except queue.Empty:
            pass
        root.after(50, poll)

    root.after(50, poll)

    # Start workers
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)

    t_audio = threading.Thread(target=audio_stream_generator, args=(audio_q, state), daemon=True)
    t_audio.start()

    t_local = threading.Thread(target=run_local_whisper, args=(audio_q, state, ui_queue), daemon=True)
    t_local.start()

    def start_asyncio_deepgram():
        asyncio.run(run_deepgram_ws(state, ui_queue))

    t_cloud = threading.Thread(target=start_asyncio_deepgram, daemon=True)
    t_cloud.start()

    root.mainloop()


if __name__ == "__main__":
    shared = SharedState()
    ui_loop(shared)
