"""Native (Rust, multi-core) backend for the vectorial PSF simulator.

The heavy Debye integral lives in the compiled extension ``faser._core``.
This module only maps a :class:`PSFConfig` (with all of its derived
properties) onto the extension's flat config struct.
"""

import numpy as np

from faser import _core
from faser.generators.base import PSFConfig, mode, normalize

_MODES = {
    mode.GAUSSIAN: _core.Mode.Gaussian,
    mode.DONUT: _core.Mode.Donut,
    mode.BOTTLE: _core.Mode.Bottle,
    mode.DONUT_BOTTLE: _core.Mode.DonutBottle,
}

_ZERNIKE = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a12", "a24"]


def to_native_config(s: PSFConfig) -> "_core.PsfConfig":
    """Flatten a PSFConfig, including derived quantities, into a native config."""
    if s.Mode not in _MODES:
        raise NotImplementedError(f"Mode {s.Mode} is not supported by the native backend")

    c = _core.PsfConfig()
    c.n1, c.n2, c.n3 = s.n1, s.n2, s.n3
    c.r0, c.r0_eff = s.r0, s.r0_eff
    c.n_xy, c.n_z, c.n_theta, c.n_phi = s.Nxy, s.Nz, s.Ntheta, s.Nphi
    c.deltatheta, c.deltaphi, c.k0 = s.deltatheta, s.deltaphi, s.k0
    c.waist, c.wd, c.l_obs_xy, c.l_obs_z = s.Waist, s.WD, s.L_obs_XY, s.L_obs_Z
    c.dfoc, c.thickness, c.depth, c.collar = s.Dfoc, s.Thickness, s.Depth, s.Collar
    c.mode = _MODES[s.Mode]
    c.normalize = _core.Normalize.Yes if s.Normalize == normalize.YES else _core.Normalize.No
    c.polarization = int(s.Polarization) - 1  # enum is 1-based, native is 0-based
    c.psi, c.eps, c.alpha_eff, c.cg, c.sg = s.psi, s.eps, s.alpha_eff, s.cg, s.sg
    for name in _ZERNIKE:
        setattr(c, name, float(getattr(s, name)))
    c.ampl_offset_x, c.ampl_offset_y = s.Ampl_offset_x, s.Ampl_offset_y
    c.mask_offset_x, c.mask_offset_y = s.Mask_offset_x, s.Mask_offset_y
    c.aberration_offset_x, c.aberration_offset_y = s.Aberration_offset_x, s.Aberration_offset_y
    c.vc, c.rc, c.ring_radius, c.p = s.VC, s.RC, s.Ring_Radius, s.p
    return c


def generate_psf(s: PSFConfig) -> np.ndarray:
    """Vectorial PSF intensity of shape (Nz, Nxy, Nxy), computed natively.

    Identical to :func:`faser.generators.vectorial.stephane.tilted_coverslip.generate_psf`
    up to floating-point reordering, but runs on all cores and releases the GIL.
    """
    return _core.generate_psf(to_native_config(s))
