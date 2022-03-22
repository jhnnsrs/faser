#%%
from faser.generators.base import PSFGeneratorConfig
from faser.generators.scalar.phasenet import PhaseNetPSFGenerator
from faser.generators.vectorial.stephane import StephanePSFGenerator
from faser.generators.scalar.gibson_lanny import GibsonLannyPSFGenerator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import napari

config = PSFGeneratorConfig(Nx=65, Ny=65, Nz=65, numerical_aperature=1.2)


from faser.io import save


x = StephanePSFGenerator(PSFGeneratorConfig(numerical_aperature=1.2)).generate()
napari.view_image(x)
napari.run()
