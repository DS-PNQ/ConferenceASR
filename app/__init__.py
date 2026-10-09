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


# Windows 11 runs a windowless backend as EcoQoS: threads parked on E-cores at
# low clocks. MEASURED i7-13620H live: MT chunk 0.6 s -> 3-4 s, backlog grows;
# opted out (or pinned to P-cores) every final lands < 1.1 s.
if sys.platform == "win32":
    import ctypes

    class _PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong),
                    ("StateMask", ctypes.c_ulong)]

    _s = _PowerThrottling(1, 0x1, 0)  # EXECUTION_SPEED controlled, state off
    _k32 = ctypes.windll.kernel32
    _k32.GetCurrentProcess.restype = ctypes.c_void_p
    _k32.SetProcessInformation(ctypes.c_void_p(_k32.GetCurrentProcess()), 4,  # ProcessPowerThrottling
                               ctypes.byref(_s), ctypes.sizeof(_s))
