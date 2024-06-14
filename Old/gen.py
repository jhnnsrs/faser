#%%
from faser.generators.base import PSFGeneratorConfig
from faser.generators.scalar.phasenet import PhaseNetPSFGenerator
from faser.generators.vectorial.stephane import StephanePSFGenerator
from faser.generators.scalar.gibson_lanny import GibsonLannyPSFGenerator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import napari

config = PSFGeneratorConfig(Nx=65, Ny=65, Nz=65)


from faser.io import save


save(PhaseNetPSFGenerator, config, "test_phase_net.zarr")
save(StephanePSFGenerator, config, "test_stephane.zarr")
