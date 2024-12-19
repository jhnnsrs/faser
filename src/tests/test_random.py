from faser.generators.base import PSFConfig
from faser.generators.vectorial.stephane.tilted_coverslip import generate_psf
import numpy as np
import pytest


@pytest.mark.parametrize("limit", [0, 0.4, 0.8, 1])
def test_generation(limit):

    abb = PSFConfig(a1=limit)
    ndarray = generate_psf(abb)
