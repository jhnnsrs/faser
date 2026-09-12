"""The native backend must reproduce the numpy reference implementation."""

import numpy as np
import pytest

from faser.generators.base import PSFConfig, mode, polarization
from faser.generators.vectorial import stephane
from faser.generators.vectorial.stephane import generate_psf, generate_psf_numpy

SMALL = {"Normalize": "YES", "Ntheta": 30, "Nphi": 24, "Nxy": 15, "Nz": 9}

CASES = {
    "gaussian_elliptical": {},
    "gaussian_radial": {"Polarization": polarization.RADIAL},
    "gaussian_azimuthal": {"Polarization": polarization.AZIMUTHAL},
    "donut": {"Mode": mode.DONUT},
    "bottle": {"Mode": mode.BOTTLE},
    "donut_bottle": {"Mode": mode.DONUT_BOTTLE},
    "depth_tilt_zernike": {"Depth": 50, "Tilt": 5, "Thickness": 180, "a12": 0.5, "a7": 0.2, "Ampl_offset_x": 2},
    "supercritical": {"NA": 1.2, "n3": 1.0},
    "even_grid": {"Nxy": 14, "Nz": 6},
    "cranial_window": {"Window": "CUSTOM", "Wind_Radius": 1.0, "Wind_Depth": 2.0},
}


def test_native_backend_is_default():
    assert stephane.BACKEND == "rust"
    assert generate_psf is not generate_psf_numpy


@pytest.mark.parametrize("name", CASES)
def test_native_matches_numpy(name):
    config = PSFConfig(**{**SMALL, **CASES[name]})
    ref = generate_psf_numpy(config)
    out = generate_psf(config)
    assert out.shape == ref.shape == (config.Nz, config.Nxy, config.Nxy)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-10)


def test_loaded_mode_not_supported():
    from faser.generators.vectorial.stephane.rust_backend import to_native_config

    config = PSFConfig(**SMALL, Mode=mode.LOADED, loaded_phase_mask=np.zeros((3, 3)))
    with pytest.raises(NotImplementedError):
        to_native_config(config)
