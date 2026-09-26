import os
import sys
from pathlib import Path
from ctypes import CDLL

# Cache the module to avoid modifying sys.path and reloading DLLs multiple times within the same process
_LUMAPI_MODULE = None

def get_lumapi():
    """
    Lazily retrieve and initialize the lumapi module.
    Ensure the environment variable 'lumapi_path' is configured in your OS before calling this function.
    """
    global _LUMAPI_MODULE
    if _LUMAPI_MODULE is not None:
        return _LUMAPI_MODULE

    # 1. Retrieve the environment variable
    lumapi_path = os.getenv("lumapi_path")
    if not lumapi_path:
        raise EnvironmentError(
            "\n[hlp-eme] Environment variable 'lumapi_path' is not set!\n"
            "Please set 'lumapi_path' to the directory containing the Lumerical Python API.\n"
            r"Example: C:\Program Files\Lumerical\v241\api\python"
        )

    # 2. Validate directory existence
    lumapi_dir = Path(lumapi_path).resolve()
    if not lumapi_dir.is_dir():
        raise FileNotFoundError(
            f"[hlp-eme] Directory specified by 'lumapi_path' does not exist: {lumapi_dir}"
        )

    # 3. Add to sys.path if not already present
    str_dir = str(lumapi_dir)
    if str_dir not in sys.path:
        sys.path.append(str_dir)

    # 4. Explicitly load interopapi.dll on Windows if present
    dll_path = lumapi_dir / "interopapi.dll"
    if dll_path.exists():
        CDLL(str(dll_path))

    # 5. Import and cache lumapi
    try:
        import lumapi
    except ImportError as e:
        raise ImportError(
            f"[hlp-eme] Failed to import lumapi.py from '{lumapi_dir}'. Please verify the path."
        ) from e

    _LUMAPI_MODULE = lumapi
    return _LUMAPI_MODULE