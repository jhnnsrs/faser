#%%
from faser.generators.base import Aberration, PSFGeneratorConfig, Mode
from faser.generators.scalar.phasenet import PhaseNetPSFGenerator
from faser.generators.vectorial.stephane import StephanePSFGenerator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import napari

config = PSFGeneratorConfig()


x = PhaseNetPSFGenerator(config)
z = StephanePSFGenerator(config)

#%%
h1 = x.generate()

def gen(abs: Aberration):
    return StephanePSFGenerator(PSFGeneratorConfig(aberration=abs)).generate()

def gen_don(abs: Aberration):
    return StephanePSFGenerator(PSFGeneratorConfig(aberration=abs, mode=Mode.DONUT)).generate()


x = {
    "Gaussian" : {
        "Pure": gen(Aberration(a5=0)),
        "t": gen(Aberration(a5=0.7))
    },
    "Donut" : {
        "Pure": gen_don(Aberration(a5=0)),
        "t": gen_don(Aberration(a5=0.7))
    }
}

nrows = len(x.keys())

#%%
fix, axes = plt.subplots(nrows=nrows, ncols=len(x["Gaussian"].keys()), figsize=(20, 20))


for i, row in enumerate(x.items()):
    key, value = row
    for y, col in enumerate(value.items()):
        subkey, image = col
        mid_plane = image.shape[0] // 2
        axes[i,y].imshow(image[mid_plane], cmap="hsv", interpolation="bilinear")


plt.show()
# %%

# %%
