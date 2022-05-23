from typing import Tuple
from .generators.base import Aberration
import numpy as np


def generate_random_abberation(
    limits: Tuple[float, float], limit_map=None
) -> Aberration:
    if not limit_map:
        limit_map = {
            key: (limits[0], limits[1]) for key, value in Aberration.__fields__.items()
        }

    return Aberration(
        **{key: float(np.random.uniform(*value)) for key, value in limit_map.items()}
    )
