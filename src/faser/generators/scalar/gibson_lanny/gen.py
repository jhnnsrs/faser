import os
import tifffile
from faser.generators.base import PSFGeneratorConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
generator_path = os.path.join(dir_path, "PSFGenerator.jar")
config_path = "config.txt"


def generate_psf(config: PSFGeneratorConfig):

    with open(config_path, "w") as f:
        f.write(
            f"""
#PSFGenerator
#Sun Jan 14 21:27:57 CET 2018
PSF-shortname=GL
ResLateral=100.0
ResAxial=250.0
NY={config.Ny}
NX={config.Nx}
NZ={config.Nz}
Type=32-bits
NA={config.numerical_aperature}
LUT=Grays
Lambda=610.0
Scale=Linear
psf-BW-NI=1.5
psf-BW-accuracy=Good
psf-RW-NI=1.5
psf-RW-accuracy=Good
psf-GL-NI={config.refractive_index}
psf-GL-NS=1.33
psf-GL-accuracy=Good
psf-GL-ZPos=2000.0
psf-GL-TI={config.working_distance * 10e5}
psf-TV-NI=1.5
psf-TV-ZPos=2000.0
psf-TV-TI=150.0
psf-TV-NS=1.0
psf-Circular-Pupil-defocus=100.0
psf-Circular-Pupil-axial=Linear
psf-Circular-Pupil-focus=0.0
psf-Oriented-Gaussian-axial=Linear
psf-Oriented-Gaussian-focus=0.0
psf-Oriented-Gaussian-defocus=100.0
psf-Astigmatism-focus=0.0
psf-Astigmatism-axial=Linear
psf-Astigmatism-defocus=100.0
psf-Defocus-DBot=30.0
psf-Defocus-ZI=2000.0
psf-Defocus-DTop=30.0
psf-Defocus-DMid=1.0
psf-Defocus-K=275.0
psf-Cardinale-Sine-axial=Linear
psf-Cardinale-Sine-defocus=100.0
psf-Cardinale-Sine-focus=0.0
psf-Lorentz-axial=Linear
psf-Lorentz-focus=0.0
psf-Lorentz-defocus=100.0
psf-Koehler-dMid=3.0
psf-Koehler-dTop=1.5
psf-Koehler-n1=1.0
psf-Koehler-n0=1.5
psf-Koehler-dBot=6.0
psf-Double-Helix-defocus=100.0
psf-Double-Helix-axial=Linear
psf-Double-Helix-focus=0.0
psf-Gaussian-axial=Linear
psf-Gaussian-focus=0.0
psf-Gaussian-defocus=100.0
psf-Cosine-axial=Linear
psf-Cosine-focus=0.0
psf-Cosine-defocus=100.0
psf-VRIGL-NI=1.5
psf-VRIGL-accuracy=Good
psf-VRIGL-NS2=1.4
psf-VRIGL-NS1=1.33
psf-VRIGL-TG=170.0
psf-VRIGL-NG=1.5
psf-VRIGL-TI=150.0
psf-VRIGL-RIvary=Linear
psf-VRIGL-ZPos=2000.0
"""
        )

    print(f"java -cp {generator_path} {config_path}")
    os.system(f"java -cp {generator_path} PSFGenerator {config_path}")

    x = tifffile.imread("PSF GL.tif")
    os.remove("PSF GL.tif")
    os.remove(config_path)
    return x
