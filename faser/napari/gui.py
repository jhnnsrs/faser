from magicgui import magic_factory, magicgui
from faser.generators.base import Aberration, PSFConfig, Mode, Polarization
from faser.generators.vectorial.stephane import generate_psf
import numpy as np
import napari
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

slider = {"widget_type": "FloatSlider", "min": 0, "max": 1, "step": 0.05}
detector_slider = {"widget_type": "FloatSlider", "min": 0, "max": 1, "step": 0.05}
focal_slider = {"widget_type": "Slider", "min": 1, "max": 10, "step": 1}

viewer = None


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
    gaussian_beam_noise=detector_slider,
    detector_gaussian_noise=detector_slider,
)
def generate_psf_gui(
    viewer: napari.Viewer,
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
    gaussian_beam_noise=0.0,
    detector_gaussian_noise=0.0,
    add_detector_poisson_noise=False,
    rescale=True,
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
        gaussian_beam_noise=gaussian_beam_noise,
        detector_gaussian_noise=detector_gaussian_noise,
        add_detector_poisson_noise=add_detector_poisson_noise,
        LfocalX=LfocalXY * 1e-6,
        LfocalY=LfocalXY * 1e-6,  # observation scale Y
        LfocalZ=LfocalZ * 1e-6,
        rescale=rescale,
    )

    psf = generate_psf(config)
    print(psf.max())
    return viewer.add_image(psf, name=f"PSF {config.aberration} ")
