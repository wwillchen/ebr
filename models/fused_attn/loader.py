"""Lazy loading of the Triton kernel modules, shared by the fused attention and the fused RoPE split.

``EBR_FUSED_ATTN_IMPL``: ``auto`` (default) uses Triton when it imports and falls back to the dense reference
otherwise, ``triton`` makes a missing kernel an error, ``reference`` forces the dense reference.
"""

import importlib
import os
from collections.abc import Sequence
from typing import Any

# module name -> (loaded module or None, import error)
_LOADED: dict[str, tuple[Any | None, str | None]] = {}


def triton_module(name: str, what: str, required: Sequence[str] = ()) -> Any | None:
    """``models.fused_attn.<name>`` (imported once), or None when unavailable / ``EBR_FUSED_ATTN_IMPL=reference``.

    ``required``: callables the module must define. ``what`` names the kernels in the error message.
    """
    mode = os.environ.get("EBR_FUSED_ATTN_IMPL", "auto").lower()
    if mode == "reference":
        return None
    if name not in _LOADED:
        try:
            mod = importlib.import_module(f"models.fused_attn.{name}")
            missing = [n for n in required if not callable(getattr(mod, n, None))]
            if missing:
                raise ImportError(f"{name} lacks {missing}")
            _LOADED[name] = (mod, None)
        except Exception as exc:  # any import failure means "use the reference"
            _LOADED[name] = (None, f"{type(exc).__name__}: {exc}")
    mod, error = _LOADED[name]
    if mod is None and mode == "triton":
        raise RuntimeError(f"EBR_FUSED_ATTN_IMPL=triton but the {what} are unavailable: {error}")
    return mod


__all__ = ["triton_module"]
