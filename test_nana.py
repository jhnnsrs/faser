#%%
from faser.generators.base import Aberration, PSFGeneratorConfig, Mode
from faser.generators.scalar.phasenet import PhaseNetPSFGenerator
from faser.generators.vectorial.stephane import StephanePSFGenerator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import napari

#%%
def gen_vectorial(x):
    return StephanePSFGenerator(PSFGeneratorConfig(numerical_aperature=x)).generate()

def gen_scalar(x):
    return PhaseNetPSFGenerator(PSFGeneratorConfig(numerical_aperature=x)).generate()

#%%
t = list(np.linspace(0.1, 1.7, 30))
images = []
for el in t:
    x = gen_scalar(el)
    y = gen_vectorial(el)
    print(np.abs(x-y).sum())
    images.append((x, y))

#%%
plt.plot(t)
plt.show()

# %%
errors = []
for x in images:
    errors.append(np.abs(x[0]-x[1]).sum())



# %%
print(errors)
# %%
