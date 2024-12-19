import napari
import numpy as np
from magicgui import magicgui

# from faser.generators.base import PSFConfig, mode, window
from faser.generators.vectorial.stephane.tilted_coverslip import generate_psf
import numpy as np

# import napari
# from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from scipy import ndimage
from skimage.transform import resize

# from faser.generators.base import Aberration, Mode, WindowType, Polarization, PSFConfig
# from faser.generators.vectorial.stephane.tilted_coverslip import generate_psf

slider = {"widget_type": "FloatSlider", "min": -1, "max": 1, "step": 0.05}
tilt_slider = {"widget_type": "FloatSlider", "min": 0, "max": 90, "step": 0.05}
detector_slider = {"widget_type": "FloatSlider", "min": 0, "max": 1, "step": 0.05}
focal_slider = {"widget_type": "Slider", "min": 1, "max": 10, "step": 1}
beam_slider = {"widget_type": "FloatSlider", "min": 0.5, "max": 50, "step": 0.5}

viewer = None


""" @magicgui(
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
    spherical2nd=slider,
    tilt_angle=tilt_slider,
    gaussian_beam_noise=detector_slider,
    detector_gaussian_noise=detector_slider,
) """


def generate_psf_gui():
    """viewer: napari.Viewer,
    Nx=31,  # discretization of image plane
    Ny=31,
    Nz=31,
    LfocalXY=2,  # observation scale X and Y
    LfocalZ=4,  # observation scale Z
    Ntheta=31,  # Integration steps
    Nphi=31,
    # Optical aberrations
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
    spherical2nd=0.0,
    tilt_angle=0.0,
    gaussian_beam_noise=0.0,
    detector_gaussian_noise=0.0,
    add_detector_poisson_noise=False,
    Normalize=True,
    # Phase profile
    Mode: mode = mode.GAUSSIAN,
    Window: window = window.NO,
    psi=0,
    eps=45,
    """
    """    config = PSFConfig(
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
        a24=spherical2nd
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        Ntheta=Ntheta,
        Nphi=Nphi,
        Mode=Mode,
        Polarization=Polarization,
        Window=Window,
        gaussian_beam_noise=gaussian_beam_noise,
        detector_gaussian_noise=detector_gaussian_noise,
        add_detector_poisson_noise=add_detector_poisson_noise,
        LfocalX=LfocalXY * 1e-6,
        LfocalY=LfocalXY * 1e-6,  # observation scale Y
        LfocalZ=LfocalZ * 1e-6,
        Normalize=Normalize,
        psi_degree=psi,
        eps_degree=eps,
        tilt_angle_degree=tilt_angle,
    )
 """
    # psf = generate_psf(config)
    # print(psf.max())
    # return viewer.add_image(
    #    psf,
    #    name=f"PSF {config.mode.name} {config} ",
    #    metadata={"is_psf": True, "config": config},
    #    colormap="viridis",
    # )


@magicgui(
    call_button="Effective PSF",
    I_sat=slider,
)
def make_effective_gui(viewer: napari.Viewer, I_sat=0.1):

    gaussian_layers = (
        layer for layer in viewer.layers.selection if layer.metadata.get("is_psf", True)
    )

    psf_layer_one = next(gaussian_layers)
    psf_layer_two = next(gaussian_layers)
    new_psf = np.multiply(psf_layer_one.data, np.exp(-psf_layer_two.data / I_sat))

    return viewer.add_image(
        new_psf,
        name=f"Combined PSF {psf_layer_one.name} {psf_layer_two.name}",
        metadata={"is_psf": True},
    )


""" @magicgui(
    call_button="Convolve Image",
) """


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


""" @magicgui(
    call_button="Generate Space",
) """


def generate_space(
    viewer: napari.Viewer, x_size=100, y_size=100, z_size=20, dots: int = 50
):

    x = np.random.randint(0, x_size, size=(dots))
    y = np.random.randint(0, y_size, size=(dots))
    z = np.random.randint(0, z_size, size=(dots))

    M = np.zeros((z_size, x_size, y_size))
    for p in zip(z, x, y):
        M[p] = 1

    viewer.add_image(M, name="Space")


""" @magicgui(
    call_button="Generate Space",
) """


def calculate_fwhm(viewer: napari.Viewer):

    psf_layer = next(
        layer
        for layer in viewer.layers.selection
        if layer.metadata.get("is_psf", False)
    )

    psf_data = psf_layer.data
    # caluclate max coordinate
    max_coord = np.argmax(psf_data, axis=None)
    viewer.add_image(M, name="Space")
