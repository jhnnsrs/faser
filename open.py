import tifffile


x = tifffile.imread("PSF GL.tif")
print(x.shape)
