"""The native backend must reproduce the numpy reference implementation."""

import numpy as np
import pytest

from faser.generators.base import PSFConfig, SLMConfig, mode, polarization, theta_sampling
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


def _vortex(n: int) -> np.ndarray:
    """A charge-1 vortex sampled at the pixel centres, first row at +y."""
    u = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(u, -u)
    return np.arctan2(y, x)


def test_loaded_mode_uses_the_phase_mask_as_an_slm():
    """Mode LOADED with a fine vortex `loaded_phase_mask` reproduces the DONUT mode."""
    donut = generate_psf(PSFConfig(**SMALL, Mode=mode.DONUT, VC=1.0))
    loaded = generate_psf(PSFConfig(**SMALL, Mode=mode.LOADED, loaded_phase_mask=_vortex(512)))
    assert loaded.shape == donut.shape
    np.testing.assert_allclose(loaded, donut, rtol=0, atol=2e-3)
    assert loaded[4, 7, 7] < 1e-3  # dark centre


def test_slm_quantization_and_json_round_trip():
    """An 8-bit SLM is a small perturbation; a 93 % fill factor leaks light
    into the donut centre; and the JSON path (as used by the playground)
    agrees with the native config path."""
    donut = generate_psf(PSFConfig(**SMALL, Mode=mode.DONUT, VC=1.0))
    eight_bit = generate_psf(PSFConfig(**SMALL, Mode=mode.LOADED, SLM=SLMConfig.from_array(_vortex(256), levels=256)))
    assert np.abs(eight_bit - donut).max() < 5e-2
    assert eight_bit[4, 7, 7] < 1e-3
    config = PSFConfig(**SMALL, Mode=mode.LOADED, SLM=SLMConfig.from_array(_vortex(256), levels=256, fill_factor=0.93))
    leaky = generate_psf(config)
    assert 1e-3 < leaky[4, 7, 7] < 5e-2
    via_json = _core_generate_json(config)
    np.testing.assert_allclose(via_json, leaky, rtol=0, atol=1e-10)


def _core_generate_json(config: PSFConfig) -> np.ndarray:
    from faser import _core

    return _core.generate_psf_json(config.model_dump_json(exclude={"loaded_phase_mask"}))


def test_slm_shape_is_validated():
    with pytest.raises(ValueError):
        SLMConfig(n=4, phase=[0.0] * 15)
    with pytest.raises(ValueError):
        PSFConfig(**SMALL, Mode=mode.LOADED)


def test_adaptive_theta_sampling():
    """Identical to uniform without a critical angle in the aperture; closer to
    a very fine reference than uniform with one (oil objective into water)."""
    sub = PSFConfig(**SMALL)
    np.testing.assert_array_equal(
        generate_psf(sub.model_copy(update={"Theta_sampling": theta_sampling.ADAPTIVE})), generate_psf(sub)
    )
    mismatch = dict(NA=1.4, n1=1.518, n2=1.518, n3=1.33, Depth=20, Wavelength=0.488, L_obs_Z=3.0, Nphi=48, Nxy=11, Nz=5, Normalize="YES")
    reference = generate_psf(PSFConfig(**mismatch, Ntheta=4000))
    uniform = generate_psf(PSFConfig(**mismatch, Ntheta=300))
    adaptive = generate_psf(PSFConfig(**mismatch, Ntheta=300, Theta_sampling=theta_sampling.ADAPTIVE))
    err_uniform = np.abs(uniform - reference).max()
    err_adaptive = np.abs(adaptive - reference).max()
    assert err_adaptive < err_uniform / 3
    assert err_adaptive < 1e-2


@pytest.mark.parametrize("name", CASES)
def test_json_config_derivation_matches_python(name):
    """generate_psf_json derives k0, alpha, r0, Dfoc, ... in Rust (the code
    path of the WebAssembly playground); it must agree with the Python model."""
    from faser import _core

    config = PSFConfig(**{**SMALL, **CASES[name]})
    ref = generate_psf(config)
    out = _core.generate_psf_json(config.model_dump_json())
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-10)
