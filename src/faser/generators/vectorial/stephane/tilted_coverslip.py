from typing import Tuple
import numpy as np

# from pkg_resources import working_set
from faser.generators.base import *

# from faser.generators.utils import polar_to_cartesian


def cart_to_polar(x, y) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.sqrt(np.square(x) + np.square(y))
    # rho=rho/s.pupil_radius
    theta = np.arctan2(y, x)
    return rho, theta


def Amplitude(x, y, s: PSFConfig):
    Amp = np.exp(-(x**2 + y**2) / s.Waist**2)
    return Amp


def zernike(x: np.ndarray, y: np.ndarray, s: PSFConfig):

    temp, phi = cart_to_polar(x, y)
    rho = temp / s.r0

    Z0 = 1
    Z1 = 2 * rho * np.sin(phi)  # Tilt -vertical tilt
    Z2 = 2 * rho * np.cos(phi)  # Tip - horizontal tilt
    Z3 = np.sqrt(6) * (rho**2) * np.sin(2 * phi)  # Oblique astigmatism
    Z4 = np.sqrt(3) * (2 * rho**2 - 1)  # Defocus
    Z5 = np.sqrt(6) * (rho**2) * np.cos(2 * phi)  # Vertical astigmatism
    Z6 = np.sqrt(8) * (rho**3) * np.sin(3 * phi)  # Vertical trefoil
    Z7 = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.sin(phi)  # Vertical coma
    Z8 = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(phi)  # Horizontal coma
    Z9 = np.sqrt(8) * (rho**3) * np.cos(3 * phi)  # Oblique trefoil
    Z12 = np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)  # Primary spherical
    Z24 = np.sqrt(7) * (
        20 * rho**6 - 30 * rho**4 + 12 * rho**2 - 1
    )  # Secondary spherical
    zer = (
        s.a0 * Z0
        + s.a1 * Z1
        + s.a2 * Z2
        + s.a3 * Z3
        + s.a4 * Z4
        + s.a5 * Z5
        + s.a6 * Z6
        + s.a7 * Z7
        + s.a8 * Z8
        + s.a9 * Z9
        + s.a12 * Z12
        + s.a24 * Z24
    )
    return zer


def Fresnel_coeff(s: PSFConfig, ca, c2a, c2at, c3a):

    t1p = 2 * s.n1 * ca / (s.n2 * ca + s.n1 * c2a)
    t2p = 2 * s.n2 * c2a / (s.n3 * c2a + s.n2 * c3a)
    r1p = (s.n2 * ca - s.n1 * c2a) / (s.n2 * ca + s.n1 * c2a)
    r2p = (s.n3 * c2a - s.n2 * c3a) / (s.n3 * c2a + s.n2 * c3a)

    t1s = 2 * s.n1 * ca / (s.n1 * ca + s.n2 * c2a)
    t2s = 2 * s.n2 * c2a / (s.n2 * c2a + s.n3 * c3a)
    r1s = (s.n1 * ca - s.n2 * c2a) / (s.n1 * ca + s.n2 * c2a)
    r2s = (s.n2 * c2a - s.n3 * c3a) / (s.n2 * c2a + s.n3 * c3a)

    beta = s.k0 * s.n2 * (s.Thickness * c2a - s.Collar * c2at)

    Tp = t2p * t1p * np.exp(1j * beta) / (1 + r1p * r2p * np.exp(2 * 1j * beta))
    Ts = t2s * t1s * np.exp(1j * beta) / (1 + r1s * r2s * np.exp(2 * 1j * beta))

    return Tp, Ts


def poisson_noise(image, seed=None):

    # Add Poisson noise to an image.

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


def generate_intensity_profile(s: PSFConfig):

    # Discretization of the pupil plan
    x1 = np.linspace(-s.r0, s.r0, s.Nxy)
    y1 = np.linspace(-s.r0, s.r0, s.Nxy)
    X1, Y1 = np.meshgrid(x1, y1)

    rho, phi = cart_to_polar(X1, Y1)

    intensity = np.empty((s.Nxy, s.Nxy)) * np.nan

    mask1 = rho <= s.r0
    temp = np.zeros((s.Nxy, s.Nxy))
    intensity[mask1] = temp[mask1]

    mask2 = rho <= s.r0_eff
    intensity[mask2] = Amplitude(
        X1[mask2] - s.r0 / s.Nxy * s.Ampl_offset_x,
        Y1[mask2] - s.r0 / s.Nxy * s.Ampl_offset_y,
        s,
    )

    return intensity


def generate_phase_mask(s: PSFConfig):

    if s.Mode == mode.DONUT_BOTTLE:  # Donut & Bottle
        raise NotImplementedError("No display Donut and Bottle")
    else:
        # Discretization of the pupil plan
        x1 = np.linspace(-s.r0, s.r0, s.Nxy)
        y1 = np.linspace(-s.r0, s.r0, s.Nxy)
        [X1, Y1] = np.meshgrid(x1, y1)

        rho, phi = cart_to_polar(X1, Y1)
        phased = np.empty((s.Nxy, s.Nxy)) * np.nan

        mask1 = rho <= s.r0
        temp = np.zeros((s.Nxy, s.Nxy))
        phased[mask1] = temp[mask1]

        mask2 = rho <= s.r0_eff
        phased[mask2] = phase_mask_array(
            X1[mask2] - s.r0 / s.Nxy * s.Mask_offset_x,
            Y1[mask2] - s.r0 / s.Nxy * s.Mask_offset_y,
            s,
        )

    return phased


def generate_aberration(s: PSFConfig):

    # Discretization of the pupil plan
    x1 = np.linspace(-s.r0, s.r0, s.Nxy)
    y1 = np.linspace(-s.r0, s.r0, s.Nxy)
    [X1, Y1] = np.meshgrid(x1, y1)

    rho, phi = cart_to_polar(X1, Y1)
    Ab = np.empty((s.Nxy, s.Nxy)) * np.nan

    mask1 = rho <= s.r0
    temp = np.zeros((s.Nxy, s.Nxy))
    Ab[mask1] = temp[mask1]

    mask2 = rho <= s.r0_eff
    Ab[mask2] = zernike(
        X1[mask2] - s.r0 / s.Nxy * s.Aberration_offset_x,
        Y1[mask2] - s.r0 / s.Nxy * s.Aberration_offset_y,
        s,
    )

    return Ab


# phase mask function for input array
def phase_mask_array(
    x: np.ndarray,
    y: np.ndarray,
    s: PSFConfig,
):
    rho, phi = cart_to_polar(x, y)

    if s.Mode == mode.GAUSSIAN:  # gaussian
        mask = np.ones(rho.shape)
    elif s.Mode == mode.DONUT:  # donut
        mask = s.VC * phi
    elif s.Mode == mode.BOTTLE:  # bottle
        cutoff_radius = s.Ring_Radius * s.r0
        mask = s.RC * np.pi * np.ones(rho.shape)
        mask[rho > cutoff_radius] = 0
    elif s.Mode == mode.LOADED:

        print(s.loaded_phase_mask)
        raise NotImplementedError("Needs to be implemented")
        # interpolate value from

    elif s.Mode == mode.DONUT_BOTTLE:  # Donut & Bottle
        raise NotImplementedError("No display Donut and Bottle")
    else:
        raise NotImplementedError("Please use a specified Mode")
    return mask


# phase mask function for scalar input
def phase_mask(
    x: np.ndarray,
    y: np.ndarray,
    s: PSFConfig,
):
    rho, phi = cart_to_polar(x, y)

    if s.Mode == mode.GAUSSIAN:  # gaussian
        mask = 1
    elif s.Mode == mode.DONUT:  # donut
        mask = np.exp(1j * s.VC * phi)
    elif s.Mode == mode.BOTTLE:  # bottle
        cutoff_radius = s.Ring_Radius * s.r0
        if rho <= cutoff_radius:
            mask = np.exp(1j * s.RC * np.pi)
        else:
            mask = np.exp(1j * 0)
    elif s.Mode == mode.DONUT_BOTTLE:  # Donut & Bottle
        raise NotImplementedError("No display Donut and Bottle")
    else:
        raise NotImplementedError("Please use a specified Mode")
    return mask


def calculate_electric_field(s: PSFConfig) -> np.array:
    # Discretization of the pupil plan
    x1 = np.linspace(-s.r0, s.r0, s.Nxy)
    y1 = np.linspace(-s.r0, s.r0, s.Nxy)
    [X1, Y1] = np.meshgrid(x1, y1)

    # Discretization of the sample volumle
    x2 = np.linspace(-s.L_obs_XY, s.L_obs_XY, s.Nxy)
    y2 = np.linspace(-s.L_obs_XY, s.L_obs_XY, s.Nxy)
    z2 = np.linspace(-s.L_obs_Z, s.L_obs_Z, s.Nz)
    [X2, Y2, Z2] = np.meshgrid(x2, y2, z2 + s.Dfoc / s.L_obs_Z)

    # Initialization electric field near focus
    Ex2 = 0
    Ey2 = 0
    Ez2 = 0

    # Noise = np.abs(np.random.normal(0, s.Gaussian_beam_noise, (s.Ntheta, s.Nphi)))

    theta = 0
    phi = 0
    for p in range(0, s.Ntheta):
        theta = p * s.deltatheta
        for q in range(0, s.Nphi):  # TODO check the -1
            phi = q * s.deltaphi

            ci = np.cos(phi)
            ca = np.cos(theta)
            si = np.sin(phi)
            sa = np.sin(theta)

            # refracted angles
            theta2 = np.arcsin((s.n1 / s.n2) * np.sin(theta))
            c2a = np.cos(theta2)
            theta3 = np.arcsin((s.n2 / s.n3) * np.sin(theta2))
            c3a = np.cos(theta3)
            s3a = np.sin(theta3)

            # Cartesian coordinate on pupil
            x_pup = s.WD * sa * ci
            y_pup = s.WD * sa * si

            # Rotation and projection of the pupil function
            x_pup_t = s.cg * x_pup - s.sg * s.WD
            y_pup_t = y_pup
            # Spherical coordinate
            theta_t = np.arcsin(np.sqrt(x_pup_t**2 + y_pup_t**2) / s.WD)
            cat = np.cos(theta_t)

            # refracted tilted angles
            theta2_t = np.arcsin((s.n1 / s.n2) * np.sin(theta_t))
            c2at = np.cos(theta2_t)

            if theta_t <= s.alpha_eff:

                # Amplitude of the incident beam on the objective pupil
                Amp = Amplitude(
                    x_pup_t - s.r0 / s.Nxy * s.Ampl_offset_x,
                    y_pup_t - s.r0 / s.Nxy * s.Ampl_offset_y,
                    s,
                )
                # Amp = Amp + Noise[slice][q]

                # Phase mask on the objective pupil
                PM = phase_mask(
                    x_pup_t - s.r0 / s.Nxy * s.Mask_offset_x,
                    y_pup_t - s.r0 / s.Nxy * s.Mask_offset_y,
                    s,
                )

                # Wavefront on the objective pupil
                W = np.exp(
                    1j  # *2*np.pi
                    * zernike(
                        x_pup_t - s.r0 / s.Nxy * s.Aberration_offset_x,
                        y_pup_t - s.r0 / s.Nxy * s.Aberration_offset_y,
                        s,
                    )
                )
            else:
                Amp = 0
                PM = 1
                W = 0

            # incident beam polarization cases
            p0x = [
                np.cos(s.psi) * np.cos(s.eps) - 1j * np.sin(s.psi) * np.sin(s.eps),
                ci,
                -si,
            ]
            p0y = [
                np.sin(s.psi) * np.cos(s.eps) + 1j * np.cos(s.psi) * np.sin(s.eps),
                si,
                ci,
            ]
            p0z = 0

            # Selected incident beam polarization
            P0 = [
                [
                    p0x[s.Polarization - 1]
                ],  # indexing minus one to get corresponding polarization
                [
                    p0y[s.Polarization - 1]
                ],  # indexing minus one to get corresponding polarization
                [p0z],
            ]

            [Tp, Ts] = Fresnel_coeff(s, ca, c2a, c2at, c3a)

            T = [  # Polarization matrix
                [
                    Tp * c3a * ci**2 + Ts * si**2,
                    Tp * si * ci * c3a - Ts * ci * si,
                    Tp * s3a * ci,
                ],
                [
                    Tp * c3a * ci * si - Ts * si * ci,
                    Tp * c3a * si**2 + Ts * ci**2,
                    Tp * s3a * si,
                ],
                [
                    -Tp * ci * s3a,
                    -Tp * s3a * si,
                    Tp * c3a,
                ],
            ]

            # polarization in focal region
            P = np.matmul(T, P0)

            # Apodization factor
            a = np.sqrt(cat)

            # numerical calculation of electric field distribution in focal region
            propagation = (
                np.exp(
                    1j * s.k0 * s.n1 * (X2 * ci * sa + Y2 * si * sa)
                    + 1j * s.k0 * s.n3 * c3a * Z2
                )
                * s.deltaphi
                * s.deltatheta
            )

            # Aberration term from the coverslip
            Psi_coverslip = s.n3 * s.Depth * c3a - s.n1 * (s.Thickness + s.Depth) * ca
            Psi_collar = -s.n1 * s.Collar * cat  ### TO DO: ca?
            Psi_w = Psi_coverslip - Psi_collar
            Ab_wind = np.exp(1j * s.k0 * Psi_w)

            # TODO: Implement the offset of the coverslip

            factored = sa * a * Amp * PM * W * Ab_wind * propagation

            Ex2 = Ex2 + factored * P[0, 0]
            Ey2 = Ey2 + factored * P[1, 0]
            Ez2 = Ez2 + factored * P[2, 0]

    return Ex2, Ey2, Ez2


def generate_psf(s: PSFConfig) -> np.ndarray:

    if s.Mode == mode.DONUT_BOTTLE:
        s.Mode = mode.DONUT  # Calculate Donut PSF
        [Ex2d, Ey2d, Ez2d] = calculate_electric_field(s)
        Ix2 = np.multiply(np.conjugate(Ex2d), Ex2d)
        Iy2 = np.multiply(np.conjugate(Ey2d), Ey2d)
        Iz2 = np.multiply(np.conjugate(Ez2d), Ez2d)
        I1d = Ix2 + Iy2 + Iz2

        s.Mode = mode.BOTTLE  # Calculated Bottle PSF
        [Ex2b, Ey2b, Ez2b] = calculate_electric_field(s)
        Ix2 = np.multiply(np.conjugate(Ex2b), Ex2b)
        Iy2 = np.multiply(np.conjugate(Ey2b), Ey2b)
        Iz2 = np.multiply(np.conjugate(Ez2b), Ez2b)
        I1b = Ix2 + Iy2 + Iz2

        I1 = s.p * I1d + (1 - s.p) * I1b

        s.Mode = mode.DONUT_BOTTLE

    else:
        [Ex2, Ey2, Ez2] = calculate_electric_field(s)
        Ix2 = np.multiply(np.conjugate(Ex2), Ex2)
        Iy2 = np.multiply(np.conjugate(Ey2), Ey2)
        Iz2 = np.multiply(np.conjugate(Ez2), Ez2)
        I1 = Ix2 + Iy2 + Iz2

    # I1 = I1 + np.abs(np.random.normal(0, s.Detector_gaussian_noise, I1.shape))

    if s.Normalize == normalize.YES:
        # We are only rescaling to the max, not the min
        I1 = I1 / np.max(I1)

    return np.real(np.moveaxis(I1, 2, 0))
