from magicgui import magic_factory, magicgui
from faser.generators.base import Aberration, PSFConfig, Mode, Polarization
from faser.generators.vectorial.stephane import generate_psf
import numpy as np
import napari
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from scipy import ndimage
from skimage.transform import resize

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
    return viewer.add_image(
        psf,
        name=f"PSF {config.mode.name} {config.aberration} ",
        metadata={"is_psf": True, "config": config},
    )


@magicgui(
    call_button="Convolve Image",
)
def convolve_image_gui(viewer: napari.Viewer, resize_psf=0):

    psf_layer = next(
        layer
        for layer in viewer.layers.selection
        if layer.metadata.get("is_psf", False)
    )
    image_layer = next(
        layer
        for layer in viewer.layers.selection
        if not layer.metadata.get("is_psf", False)
    )

    image_data = image_layer.data
    psf_data = psf_layer.data

    if image_data.ndim == 2:
        psf_data = psf_data[psf_data.shape[0] // 2, :, :]

        con = ndimage.convolve(
            image_data, psf_data, mode="constant", cval=0.0, origin=0
        )

    if resize_psf > 0:
        psf_data = resize(psf_data, (resize_psf,) * psf_data.ndim)

    con = ndimage.convolve(image_data, psf_data, mode="constant", cval=0.0, origin=0)

    return viewer.add_image(
        con.squeeze(),
        name=f"Convoled {image_layer.name} with {psf_layer.name}",
    )


@magicgui(
    call_button="Combine PSFs",
    saturation_factor=slider,
)
def make_effective_gui(viewer: napari.Viewer, saturation_factor=0.1):

    gaussian_layers = (
        layer
        for layer in viewer.layers.selection
        if layer.metadata.get("is_psf", False)
        and layer.metadata.get("config", None)
        and layer.metadata.get("config", None).mode == Mode.GAUSSIAN
    )

    non_gaussian_layers = (
        layer
        for layer in viewer.layers.selection
        if layer.metadata.get("is_psf", False)
        and layer.metadata.get("config", None)
        and layer.metadata.get("config", None).mode != Mode.GAUSSIAN
    )

    psf_layer_one = next(gaussian_layers)
    psf_layer_two = next(non_gaussian_layers)
    new_psf = np.multiply(
        psf_layer_one.data, np.exp(-psf_layer_two.data / saturation_factor)
    )

    return viewer.add_image(
        new_psf,
        name=f"Combined PSF {psf_layer_one.name} {psf_layer_two.name}",
        metadata={"is_psf": True},
    )
