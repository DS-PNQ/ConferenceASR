"""Engine wiring shared by the server and scripts: config + ASR/MT/diarizer/
enhancer + pipeline. Models stay lazy until warmup/ensure_loaded."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from app.device import device_report, resolve_device, resolve_dtype
from app.diarizer import make_diarizer
from app.enhancer import DeepFilterNetEnhancer
from app.mt_engine import HyMT2Engine
from app.pipeline import ConferencePipeline
from app.zipformer_engine import ZipformerEngine

log = logging.getLogger("conf.engines")

ROOT = Path(__file__).resolve().parent.parent


def load_cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_app() -> dict:
    """Create engines + pipeline. Models stay lazy until warmup."""
    cfg = load_cfg()
    device = resolve_device(os.getenv("DEVICE", cfg.get("device", "auto")))
    mt_dtype = resolve_dtype(device, os.getenv("MT_DTYPE", cfg.get("mt_dtype", "auto")))

    asr = ZipformerEngine(cfg)
    mt = HyMT2Engine(cfg, device, mt_dtype)
    diarizer = make_diarizer(cfg)
    enhancer = DeepFilterNetEnhancer(cfg)
    pipe = ConferencePipeline(cfg, asr, mt, diarizer, enhancer)

    return {"cfg": cfg, "device": device_report(cfg), "asr": asr, "mt": mt,
            "diarizer": diarizer, "enhancer": enhancer, "pipe": pipe}
