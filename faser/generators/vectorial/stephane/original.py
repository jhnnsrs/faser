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


# phase mask function
def phase_mask(rho: np.ndarray, theta: np.ndarray, cutoff_radius: float, mode: Mode):
    if mode == Mode.GAUSSIAN:  # guassian
        mask = 1
    elif mode == Mode.DONUT:  # donut
        mask = np.exp(1j * theta)
    elif mode == Mode.BOTTLE:  # bottleMo
        if rho < cutoff_radius:
            mask = np.exp(1j * np.pi)
        else:
            mask = np.exp(1j * 0)
    else:
        raise NotImplementedError("Please use a specified Mode")
    return mask


def generate_psf(s: PSFGeneratorConfig) -> np.ndarray:

    # Calulcated Parameters

    focusing_angle = np.arcsin(
        s.numerical_aperature / s.refractive_index
    )  # maximum focusing angle of the objective
    # effective_focusing_angle=min(atan(radius_window/thicknes_window),focusing_angle);                                           # effective focalization angle in presence of the cranial window
    effective_focusing_angle = focusing_angle

    # effective_numerical_aperture= min(refractive_index*np.sin(effective_focusing_angle),numerical_aperature)    # Effective NA in presence of the cranial window
    effective_numerical_aperture = s.numerical_aperature
    wavenumber = 2 * np.pi * s.refractive_index / s.wavelength  # wavenumber

    pupil_radius = s.working_distance * np.tan(focusing_angle)
    effective_pupil_radius = s.working_distance * np.tan(effective_focusing_angle)

    # Sample Space
    x1 = np.linspace(-pupil_radius, pupil_radius, s.Nx)
    y1 = np.linspace(-pupil_radius, pupil_radius, s.Ny)
    [X1, Y1] = np.meshgrid(x1, y1)

    x2 = np.linspace(-s.Lfocal, s.Lfocal, s.Nx)
    y2 = np.linspace(-s.Lfocal, s.Lfocal, s.Ny)
    z2 = np.linspace(-s.Lfocal, s.Lfocal, s.Nz)
    [X2, Y2, Z2] = np.meshgrid(x2, y2, z2)

    rho_pupil = np.sqrt(
        np.square(X1) + np.square(Y1)
    )  # cylindrical coordinates on pupil plane
    theta_pupil = np.arctan2(Y1, X1)

    A_pupil = np.zeros(rho_pupil.shape)
    mask_pupil = np.zeros(rho_pupil.shape)
    mask_pupil_eff = np.zeros(rho_pupil.shape)
    W_pupil = np.zeros(rho_pupil.shape)

    A_pupil[rho_pupil <= pupil_radius] = np.exp(
        -(
            (
                np.square(X1[rho_pupil <= pupil_radius])
                + np.square(Y1[rho_pupil <= pupil_radius])
            )
            / s.beam_waist**2
        )
    )  # Amplitude profile
    mask_pupil[rho_pupil <= pupil_radius] = np.angle(
        phase_mask(
            rho_pupil[rho_pupil <= pupil_radius],
            theta_pupil[rho_pupil <= pupil_radius],
            s.unit_phase_radius * pupil_radius,
            s.mode,
        )
    )  # phase mask
    W_pupil[rho_pupil <= pupil_radius] = np.angle(
        np.exp(
            1j
            * zernike(
                rho_pupil[rho_pupil <= pupil_radius] / pupil_radius,
                theta_pupil[rho_pupil <= pupil_radius],
                s.aberration,
            )
        )
    )  # Wavefront

    # Step of integral
    deltatheta = effective_focusing_angle / s.Ntheta
    deltaphi = 2 * np.pi / s.Nphi

    # Initialization
    Ex2 = 0  # Ex?component in focal
    Ey2 = 0  # Ey?component in focal
    Ez2 = 0

    theta = 0
    phi = 0
    for slice in range(0, s.Ntheta + 1):
        theta = slice * deltatheta
        for q in range(0, s.Nphi):
            phi = q * deltaphi
            T = [
                [
                    1 + (np.cos(phi) ** 2) * (np.cos(theta) - 1),
                    np.sin(phi) * np.cos(phi) * (np.cos(theta) - 1),
                    -np.sin(theta) * np.cos(phi),
                ],
                [
                    np.sin(phi) * np.cos(phi) * (np.cos(theta) - 1),
                    1 + (np.sin(phi) ** 2) * (np.cos(theta) - 1),
                    -np.sin(theta) * np.sin(phi),
                ],
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ],
            ]  # Pola matrix

            # incident beam polarization cases
            p0x = [
                1,
                0,
                1 / np.sqrt(2),
                1j / np.sqrt(2),
                2 / np.sqrt(5),
                np.cos(phi),
                -np.sin(phi),
            ]
            p0y = [
                0,
                1,
                1j / np.sqrt(2),
                1 / np.sqrt(2),
                1j / np.sqrt(5),
                np.sin(phi),
                np.cos(phi),
            ]
            p0z = 0

            # selected incident beam polarization
            P0 = [
                [p0x[s.polarization - 1]],
                [p0y[s.polarization - 1]],
                [p0z],
            ]  # needs to be a colone vector
            # polarization in focal region
            P = np.matmul(T, P0)
            # Cylindrical coordinates on pupil
            rho_pup = s.working_distance * np.sin(theta)
            theta_pup = phi

            # Incident intensity profile
            Ai = np.exp(-(rho_pup**2) / (s.beam_waist**2))
            # Apodization factor
            B = np.sqrt(np.cos(theta))
            # Phase mask
            PM = phase_mask(
                rho_pup, theta_pup, s.unit_phase_radius * pupil_radius, s.mode
            )
            # Wavefront
            W = zernike(rho_pup / pupil_radius, theta_pup, s.aberration)

            # numerical calculation of field distribution in focal region

            term1 = X2 * np.cos(phi) + Y2 * np.sin(phi)
            term2 = np.multiply(np.sin(theta), term1)
            temp = (
                np.exp(1j * wavenumber * (Z2 * np.cos(theta) + term2))
                * deltatheta
                * deltaphi
            )  # element by element
            factored = np.sin(theta) * Ai * PM * B * np.exp(1j * W) * temp

            Ex2 = Ex2 + factored * P[0, 0]
            Ey2 = Ey2 + factored * P[1, 0]
            Ez2 = Ez2 + factored * P[2, 0]

    (n1, n2) = rho_pupil.shape  # effective phase mask

    for i in range(0, n1):  # Amplitude profile
        for j in range(0, n2):
            if rho_pupil[i, j] <= pupil_radius:
                mask_pupil_eff[i, j] = -1.03 * np.pi
            if rho_pupil[i, j] <= effective_pupil_radius:
                mask_pupil_eff[i, j] = mask_pupil[i, j]

    Ix2 = np.abs(Ex2) ** 2
    Iy2 = np.abs(Ey2) ** 2
    Iz2 = np.abs(Ez2) ** 2
    I1 = Ix2 + Iy2 + Iz2

    return np.moveaxis(I1, 2, 0)
