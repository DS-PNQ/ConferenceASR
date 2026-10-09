"""Nemotron-3-Diarization streaming worker (transformers >= 5.19).

The backend pins transformers 4.57 for Hy-MT; this model needs 5.19 and
NeMo 3.0.0 cannot build it (RoPE encoder). So diarizer.py spawns this script
with the backend's own Python and ``../diardeps`` (transformers 5.19 +
huggingface_hub 1.x + tokenizers 0.23, --no-deps) first on PYTHONPATH.
Stand-alone on purpose: imports nothing from `app`.

argv: model_dir chunk,right_context dtype   e.g. "D:/CONFERENCE ASR/Nemotron-3-Diarization" 3,1 bfloat16
Fully offline: model_dir must hold config.json, processor_config.json and
model.safetensors (scripts/download_models.py puts them there once).

Protocol (binary). In: op byte + uint32 n, then
  b"A": n float32 samples @16 kHz appended to the stream
  b"R": reset (new meeting); n = epoch echoed on later output
Out: b"READY <json>\\n" once loaded (or b"ERROR <msg>\\n" and exit), then per chunk
  b"P" + uint32 epoch, start_frame, n_frames, n_spk + n_frames*n_spk float32
  speaker probabilities, one frame per 10 ms counted from the last reset.
"""
import json
import queue
import struct
import sys
import threading
import warnings

warnings.filterwarnings("ignore")


def main():
    model_id, sizes, dtype_name = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.stdout.buffer
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        chunk, right = (int(v) for v in sizes.split(","))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, dtype_name) if dev == "cuda" else torch.float32
        # Offline by design: a local folder only (the parent also sets HF_HUB_OFFLINE).
        proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForAudioFrameClassification.from_pretrained(
            model_id, dtype=dtype, local_files_only=True).to(dev).eval()
        proc.streaming_modes = dict(proc.streaming_modes, app=[chunk, right])
        proc.set_streaming_mode("app")
        torch.set_num_threads(2)  # GPU does the work; leave cores for ASR
    except Exception as e:  # noqa: BLE001 — parent falls back to NeMo Titanet
        out.write(b"ERROR " + str(e).replace("\n", " ").encode() + b"\n")
        out.flush()
        return

    first_n = proc.num_samples_first_audio_chunk
    later_n = proc.num_samples_per_audio_chunk
    step = proc.num_mel_frames_per_step
    out.write(b"READY " + json.dumps({
        "device": dev, "dtype": str(dtype).replace("torch.", ""),
        "latency_ms": proc.streaming_latency_ms, "chunk_ms": step * 10}).encode() + b"\n")
    out.flush()

    # Reader thread: the parent's writes never wait on a GPU step.
    inbox: queue.Queue = queue.Queue()

    def reader():
        inp = sys.stdin.buffer
        while True:
            head = inp.read(5)
            if len(head) < 5:
                inbox.put(None)
                return
            op, n = head[:1], struct.unpack("<I", head[1:])[0]
            inbox.put((op, n, inp.read(n * 4) if op == b"A" else b""))

    threading.Thread(target=reader, daemon=True).start()

    epoch, buf, buf0, mel, cache = 0, np.zeros(0, np.float32), 0, 0, None
    while True:
        msg = inbox.get()
        if msg is None:
            return  # parent closed the pipe
        op, n, payload = msg
        if op == b"R":
            epoch, buf, buf0, mel, cache = n, np.zeros(0, np.float32), 0, 0, None
            continue
        buf = np.concatenate([buf, np.frombuffer(payload, dtype=np.float32)])
        while True:  # every chunk the buffer now completes
            first = mel == 0
            start = 0 if first else proc.audio_chunk_start(mel)
            need = first_n if first else later_n
            if start + need > buf0 + buf.size:
                break
            x = buf[start - buf0: start - buf0 + need]
            inputs = proc(x, sampling_rate=16000, is_streaming=True,
                          is_first_audio_chunk=first).to(dev, dtype=dtype)
            with torch.inference_mode():
                o = model(**inputs, speaker_cache=cache)
            cache = o.speaker_cache
            p = o.logits[0].float().sigmoid().cpu().numpy().astype(np.float32)
            out.write(struct.pack("<cIIII", b"P", epoch, mel, p.shape[0], p.shape[1]) + p.tobytes())
            out.flush()
            mel += step
            keep = max(0, proc.audio_chunk_start(mel)) - buf0  # next chunk's first sample
            if keep > 0:
                buf, buf0 = buf[keep:], buf0 + keep


if __name__ == "__main__":
    main()
