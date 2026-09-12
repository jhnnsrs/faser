"""Vectorial PSF simulator (Richards–Wolf / Török stratified medium).

``generate_psf`` is the default simulator. It uses the compiled multi-core
backend (``faser._core``) when available and falls back to the pure numpy
reference implementation otherwise. Both give the same result.
"""

import warnings

from faser.generators.base import PSFGenerator
from .tilted_coverslip import generate_psf as generate_psf_numpy

try:
    from .rust_backend import generate_psf

    BACKEND = "rust"
except ImportError as e:  # extension not built (e.g. source checkout without cargo)
    warnings.warn(
        f"faser native backend unavailable ({e}); falling back to the slow numpy "
        "simulator. Reinstall faser (or run `maturin develop --release`) to build it.",
        RuntimeWarning,
        stacklevel=2,
    )
    generate_psf = generate_psf_numpy
    BACKEND = "numpy"

__all__ = ["generate_psf", "generate_psf_numpy", "BACKEND", "PSFGenerator"]
