#%%
import numpy as np
from faser.generators.base import PSFConfig, Mode

from faser.generators.vectorial.stephane import generate_psf

#%%
config = PSFConfig(Nx=32, Ny=32, Nz=32, mode=Mode.DONUT)


h1 = generate_psf(config)
print(h1.max())
h1.shape

# %%
h1.dtype
# %%
h1
# %%
