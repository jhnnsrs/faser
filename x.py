#%%
from faser.generators.base import Aberration, PSFGeneratorConfig
from faser.generators.scalar.phasenet import PhaseNetPSFGenerator
from faser.generators.vectorial.stephane import StephanePSFGenerator
from faser.generators.scalar.gibson_lanny import GibsonLannyPSFGenerator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import napari
from faser.retrievers.gs import ger_sax

config = PSFGeneratorConfig(Nx=33, Ny=33, Nz=33, aberration=Aberration(a7=0.5))


x = StephanePSFGenerator(config)

h1 = x.generate()

plt.figure(figsize=(20, 20))
for i in range(3):
    plt.subplot(1, 4, i + 1)
    mid_plane = h1.shape[0] // 2
    plt.imshow(h1[(mid_plane - 2) + i * 2], cmap="hot")
    plt.title(f"{-2+i*2} $\mu m$")
    plt.axis("OFF")


pm = ger_sax(h1[h1.shape[0] // 2], 30)
plt.subplot(1, 4, 4)
plt.imshow(pm, cmap="hot")
plt.colorbar()
plt.title(f"{-2+i*2} $\mu m$")
plt.axis("OFF")
plt.show()

#%%
viewer = napari.Viewer()
viewer.add_image(h1)
napari.run()
