#%%
from faser.generators.base import PSFGeneratorConfig
from faser.generators.scalar.phasenet import PhaseNetPSFGenerator
from faser.generators.vectorial.stephane import StephanePSFGenerator

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import napari
from faser.io import load
from prysm.psf import centroid
from scipy.ndimage import center_of_mass
from faser.retrievers.gs import ger_sax

config = PSFGeneratorConfig()


x = load("test_phase_net.zarr")
s = load("test_stephane.zarr")

xn = x / np.max(x)
sn = s / np.max(s)


#%%
print(center_of_mass(xn), center_of_mass(sn))

pm = ger_sax(xn[xn.shape[0] // 2], 30)
print(pm.shape)

print(x.shape, s.shape)


#%%
viewer = napari.Viewer()
viewer.add_image(xn)
viewer.add_image(sn)
viewer.add_image(pm)
viewer.add_image(np.abs(xn - sn))
napari.run()
# %%
