from scipy import ndimage
import numpy as np
from faser.generators.base import PSFConfig, Mode
from skimage import data
from faser.generators.vectorial.stephane import generate_psf

#%%
config = PSFConfig(Nx=32, Ny=32, Nz=32, mode=Mode.GAUSSIAN)

x = np.expand_dims(data.coins(), axis=2)
print(x.shape)
h1 = generate_psf(config)
print("Here")

print(ndimage.convolve(x, h1, mode="constant", cval=0.0, origin=0))
print
