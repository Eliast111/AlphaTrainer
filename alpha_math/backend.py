"""Detecao de aceleradores opcionais do AlphaTrainer.

O motor continua funcional sem dependencias externas. Em Termux, NumPy/SciPy
sao usados automaticamente quando estao instalados e importaveis.
"""
from importlib import import_module

def _optional(name):
    try:
        return import_module(name)
    except Exception:
        return None

np = _optional("numpy")
scipy_special = _optional("scipy.special")
matplotlib = _optional("matplotlib")

NUMPY_AVAILABLE = np is not None
SCIPY_AVAILABLE = scipy_special is not None
MATPLOTLIB_AVAILABLE = matplotlib is not None

def backend_info():
    """Devolve versoes e disponibilidade sem falhar se um pacote estiver ausente."""
    result = {"python_backend": "disponivel", "numpy": None, "scipy": None, "matplotlib": None}
    if np is not None:
        result["numpy"] = getattr(np, "__version__", "instalado")
    scipy = _optional("scipy")
    if scipy is not None:
        result["scipy"] = getattr(scipy, "__version__", "instalado")
    if matplotlib is not None:
        result["matplotlib"] = getattr(matplotlib, "__version__", "instalado")
    return result
