from typing import Tuple
import numpy as np  #
from faser.generators.base import *


def zernike(rho, theta, a: Aberration):
    Z1 = 1
    Z2 = 2 * rho * np.cos(theta)  # Tip
    Z3 = 2 * rho * np.sin(theta)  # Tilt
    Z4 = np.sqrt(3) * (2 * rho**2 - 1)  # Defocus
    Z5 = np.sqrt(6) * (rho**2) * np.cos(2 * theta)  # Astigmatisme
    Z6 = np.sqrt(6) * (rho**2) * np.sin(2 * theta)  # Astigmatisme
    Z7 = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(theta)  # coma
    Z8 = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.sin(theta)  # coma
    Z9 = np.sqrt(8) * (rho**3) * np.cos(3 * theta)  # Trefoil
    Z10 = np.sqrt(8) * (rho**3) * np.sin(3 * theta)  # Trefoil
    Z11 = np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)  # Spherical
    zer = (
        a.a1 * Z1
        + a.a2 * Z2
        + a.a3 * Z3
        + a.a4 * Z4
        + a.a5 * Z5
        + a.a6 * Z6
        + a.a7 * Z7
        + a.a8 * Z8
        + a.a9 * Z9
        + a.a10 * Z10
        + a.a11 * Z11
    )
    return zer


def poisson_noise(image, seed=None):
    """
    Add Poisson noise to an image.
    """

    if image.min() < 0:
        low_clip = -1.0
    else:
        low_clip = 0.0

    rng = np.random.default_rng(seed)
    # Determine unique values in image & calculate the next power of two
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))

    # Ensure image is exclusively positive
    if low_clip == -1.0:
        old_max = image.max()
        image = (image + 1.0) / (old_max + 1.0)

    # Generating noise for each unique value in image.
    out = rng.poisson(image * vals) / float(vals)

    # Return image to original range if input was signed
    if low_clip == -1.0:
        out = out * (old_max + 1.0) - 1.0

    return out


# phase mask function
def phase_mask(
    rho: np.ndarray,
    theta: np.ndarray,
    cutoff_radius: float,
    vortex_charge: float,
    ring_charge: float,
    mode: Mode,
):
    if mode == Mode.GAUSSIAN:  # guassian
        mask = 1
    elif mode == Mode.DONUT:  # donut
        mask = np.exp(1j * vortex_charge * theta)
    elif mode == Mode.BOTTLE:  # bottleMo
        if rho < cutoff_radius:
            mask = np.exp(1j * ring_charge * np.pi)
        else:
            mask = np.exp(1j * 0)
    else:
        raise NotImplementedError("Please use a specified Mode")
    return mask


def cart_to_polar(x, y) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.sqrt(np.square(x) + np.square(y))
    theta = np.arctan2(y, x)
    return rho, theta


def pol_to_cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def generate_psf(s: PSFConfig) -> np.ndarray:

    # Calulcated Parameters
    wavenumber = 2 * np.pi / s.wavelength  # wavenumber

    focusing_angle = np.arcsin(
        s.numerical_aperature / s.refractive_index_immersion
    )  # maximum focusing angle of the objective

    pupil_radius = s.working_distance * np.tan(focusing_angle)  # radius of the pupil

    # Sample Space
    x1 = np.linspace(-pupil_radius, pupil_radius, s.Nx)
    y1 = np.linspace(-pupil_radius, pupil_radius, s.Ny)
    [X1, Y1] = np.meshgrid(x1, y1)

    x2 = np.linspace(-s.LfocalX, s.LfocalX, s.Nx)
    y2 = np.linspace(-s.LfocalY, s.LfocalY, s.Ny)
    z2 = np.linspace(-s.LfocalZ, s.LfocalZ, s.Nz)
    [X2, Y2, Z2] = np.meshgrid(x2, y2, z2)  # TODO: Needs to be propüerly constructed

    ## incident beam polarization cases
    p0x = [1, 0, 1 / np.sqrt(2), 1j / np.sqrt(2)]
    p0y = [0, 1, 1j / np.sqrt(2), 1 / np.sqrt(2)]
    p0z = 0

    P0 = [
        [
            p0x[s.polarization - 1]
        ],  # indexing minus one to get corresponding polarization
        [
            p0y[s.polarization - 1]
        ],  # indexing minus one to get corresponding polarization
        [p0z],
    ]  #

    # TODO: Bring to loop to allow other polaraization

    # Step of integral
    deltatheta = focusing_angle / (s.Ntheta - 1)
    deltaphi = 2 * np.pi / s.Nphi

    # Initialization
    Ex2 = 0  # Ex?component in focal
    Ey2 = 0  # Ey?component in focal
    Ez2 = 0

    Noise = np.abs(np.random.normal(0, s.gaussian_beam_noise, (s.Ntheta, s.Nphi)))

    theta = 0
    phi = 0
    for slice in range(0, s.Ntheta):
        theta = slice * deltatheta
        for q in range(0, s.Nphi):
            phi = q * deltaphi

            ci = np.cos(phi)
            ca = np.cos(theta)
            si = np.sin(phi)
            sa = np.sin(theta)

            T = [
                [
                    1 + (ci**2) * (ca - 1),
                    si * ci * (ca - 1),
                    sa * ci,
                ],
                [
                    si * ci * (ca - 1),
                    ca * (si**2) + (ci**2),
                    sa * si,
                ],
                [
                    -sa * ci,
                    -sa * si,
                    ca,
                ],
            ]  # Pola matrix

            # polarization in focal region
            P = np.matmul(T, P0)
            # Cylindrical coordinates on pupil

            x_pup = s.working_distance * sa * ci
            y_pup = s.working_distance * sa * si

            # Cylindrical coordinates on pupil
            rho_pupil_polar, theta_pupil_polar = cart_to_polar(x_pup, y_pup)
            rho_amp_polar, theta_amp_polar = cart_to_polar(
                x_pup - pupil_radius / s.Nx * s.ampl_offsetX,
                y_pup - pupil_radius / s.Ny * s.ampl_offsetY,
            )
            rho_mask_polar, theta_mask_polar = cart_to_polar(
                x_pup - pupil_radius / s.Nx * s.mask_offsetX,
                y_pup - pupil_radius / s.Ny * s.mask_offsetY,
            )
            rho_ab_polar, theta_ab_polar = cart_to_polar(
                x_pup - pupil_radius / s.Nx * s.aberration_offsetX,
                y_pup - pupil_radius / s.Ny * s.aberration_offsetY,
            )
            # Apodization factor
            a = np.sqrt(ca)
            # Incident intensity profile
            Bi = np.exp(-np.square(rho_amp_polar) / np.square(s.beam_waist))
            Bi = Bi + Noise[slice][q]
            # Phase mask
            PM = phase_mask(
                rho_mask_polar,
                theta_mask_polar,
                s.unit_phase_radius * pupil_radius,
                s.vortex_charge,
                s.ring_charge,
                s.mode,
            )
            # Wavefront
            W = np.exp(
                1j
                * np.pi
                * zernike(rho_ab_polar / pupil_radius, theta_ab_polar, s.aberration)
            )

            # numerical calculation of field distribution in focal region

            propagation = (
                np.exp(
                    1j
                    * wavenumber
                    * s.refractive_index_immersion
                    * (X2 * ci * sa + Y2 * si * sa + Z2 * ca)
                )
                * deltaphi
                * deltatheta
            )

            factored = sa * a * Bi * PM * W * propagation

            Ex2 = Ex2 + factored * P[0, 0]
            Ey2 = Ey2 + factored * P[1, 0]
            Ez2 = Ez2 + factored * P[2, 0]

    Ix2 = np.multiply(np.conjugate(Ex2), Ex2)
    Iy2 = np.multiply(np.conjugate(Ey2), Ey2)
    Iz2 = np.multiply(np.conjugate(Ez2), Ez2)
    I1 = Ix2 + Iy2 + Iz2
    I1 = np.real(I1)

    # Add Poisson noise to I1
    if s.add_detector_poisson_noise:
        I1 = poisson_noise(I1)

    if s.rescale:
        # We are only rescaling to the max, not the min
        I1 = I1 / np.max(I1)

    return np.moveaxis(I1, 2, 0)
