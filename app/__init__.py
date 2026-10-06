import os
import sys
from pathlib import Path

# Optional triton-windows (enables compiled MT decode) installed off-C: in
# <parent of repo>/pydeps, with its compiler and caches beside it.
_PYDEPS = Path(__file__).resolve().parent.parent.parent / "pydeps"
if _PYDEPS.is_dir():
    sys.path.insert(0, str(_PYDEPS))
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_PYDEPS), os.environ.get("PYTHONPATH")]))  # compile workers
    os.environ.setdefault("CC", str(_PYDEPS / "triton" / "runtime" / "tcc" / "tcc.exe"))
    _cache = _PYDEPS.parent / ".triton-cache"
    os.environ.setdefault("TRITON_CACHE_DIR", str(_cache / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache / "inductor"))

