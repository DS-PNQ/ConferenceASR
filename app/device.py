"""CUDA with CPU fall-back, shared by ASR + MT engines."""
from __future__ import annotations

import logging
import os

log = logging.getLogger("conf.device")


def resolve_device(pref: str = "auto") -> str:
    pref = (pref or "auto").lower()
    if pref in ("cuda", "gpu"):
        return "cuda"
    if pref == "cpu":
        return "cpu"
    # auto
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_dtype(device: str, pref: str = "auto"):
    """Return a torch dtype object (or None if torch missing)."""
    try:
        import torch
    except Exception:
        return None
    pref = (pref or "auto").lower()
    if pref == "bfloat16":
        return torch.bfloat16
    if pref in ("float16", "fp16", "half"):
        return torch.float16
    if pref in ("float32", "fp32", "float"):
        return torch.float32
    # auto: bf16 on CUDA (RTX 4060 supports it), fp32 on CPU
    if device == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def device_report(cfg: dict) -> dict:
    import shutil

    device = resolve_device(os.getenv("DEVICE", cfg.get("device", "auto")))
    dtype = resolve_dtype(device, os.getenv("DTYPE", cfg.get("asr_dtype", "auto")))
    info: dict = {"device": device, "dtype": str(dtype).replace("torch.", "") if dtype else "none"}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            free, total = torch.cuda.mem_get_info(0)
            info["gpu_mem_free_gb"] = round(free / 1e9, 2)
            info["gpu_mem_total_gb"] = round(total / 1e9, 2)
    except Exception as e:  # torch not installed
        info["torch"] = None
        info["cuda_available"] = False
        info["error"] = str(e)
    return info
