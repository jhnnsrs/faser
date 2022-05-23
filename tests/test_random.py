from faser.generators.base import Aberration
from faser.utils import generate_random_abberation
import numpy as np
import pytest


@pytest.mark.parametrize("limits", [(0, 1), (0, 0.5)])
def test_random_abberation(limits):
    abb = generate_random_abberation(limits=limits)
    assert isinstance(abb, Aberration)
    assert isinstance(abb.a1, float)
    for key, value in Aberration.__fields__.items():
        assert isinstance(getattr(abb, key), float)
        assert getattr(abb, key) >= limits[0]
        assert getattr(abb, key) <= limits[1]
