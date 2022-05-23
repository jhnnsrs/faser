from magicgui import magicgui
from faser.generators.base import Aberration, PSFConfig, Mode, Polarization
from faser.generators.vectorial.stephane import StephanePSFGenerator
import numpy as np
import napari


viewer = napari.Viewer()


slider = {"widget_type": "FloatSlider", "min": 0, "max": 1, "step": 0.05}
focal_slider = {"widget_type": "Slider", "min": 1, "max": 10, "step": 1}


@magicgui(
    call_button="Generate",
    LfocalXY=focal_slider,
    LfocalZ=focal_slider,
    piston=slider,
    tip=slider,
    tilt=slider,
    defocus=slider,
    astigmatism_v=slider,
    astigmatism_h=slider,
    coma_v=slider,
    coma_h=slider,
    trefoil_v=slider,
    trefoil_h=slider,
    spherical=slider,
)
def generate_psf(
    Nx=32,
    Ny=32,
    Nz=32,
    LfocalXY=1,  # observation scale X
    LfocalZ=1,  # observation scale Z
    Ntheta=50,
    Nphi=20,
    piston=0.0,
    tip=0.0,
    tilt=0.0,
    defocus=0.0,
    astigmatism_v=0.0,
    astigmatism_h=0.0,
    coma_v=0.0,
    coma_h=0.0,
    trefoil_v=0.0,
    trefoil_h=0.0,
    spherical=0.0,
    mode: Mode = Mode.GAUSSIAN,
    polarization: Polarization = Polarization.LEFT_CIRCULAR,
):
    aberration = Aberration(
        a1=piston,
        a2=tip,
        a3=tilt,
        a4=defocus,
        a5=astigmatism_v,
        a6=astigmatism_h,
        a7=coma_v,
        a8=coma_h,
        a9=trefoil_v,
        a10=trefoil_h,
        a11=spherical,
    )
    config = PSFConfig(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        Ntheta=Ntheta,
        Nphi=Nphi,
        aberration=aberration,
        mode=mode,
        polarization=polarization,
        LfocalX=LfocalXY * 1e-6,
        LfocalY=LfocalXY * 1e-6,  # observation scale Y
        LfocalZ=LfocalZ * 1e-6,
    )

    psf = StephanePSFGenerator(config).generate()
    return viewer.add_image(psf, name=f"PSF {config.aberration} ")
