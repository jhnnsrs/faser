from typing import Tuple
import numpy as np
from pkg_resources import working_set  #
from faser.generators.base import *

def cart_to_polar(x, y) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.sqrt(np.square(x) + np.square(y))
    # rho=rho/s.pupil_radius
    theta = np.arctan2(y, x)
    return rho, theta

'''
def pol_to_cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)
'''

def Amplitude(x, y, s: PSFConfig):
    Amp=np.exp(
             -(x**2 + y**2)/s.waist**2
    )
    return Amp

def zernike(rho, phi, a: PSFConfig):
    
    Z1 = 1
    Z2 = 2 * rho * np.cos(phi)  # Tip
    Z3 = 2 * rho * np.sin(phi)  # Tilt
    Z4 = np.sqrt(3) * (2 * rho**2 - 1)  # Defocus
    Z5 = np.sqrt(6) * (rho**2) * np.cos(2 * phi)  # Astigmatism vertical
    Z6 = np.sqrt(6) * (rho**2) * np.sin(2 * phi)  # Astigmatism oblque
    Z7 = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(phi)  # coma horizontal
    Z8 = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.sin(phi)  # coma vertical
    Z9 = np.sqrt(8) * (rho**3) * np.cos(3 * phi)  # Trefoil vertical
    Z10 = np.sqrt(8) * (rho**3) * np.sin(3 * phi)  # Trefoil oblique
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

def Fresnel_coeff(s: PSFConfig, ca,c2a,c2at,c3a):
    
    t1p=2*s.n1*ca/(s.n2*ca+s.n1*c2a)
    t2p=2*s.n2*c2a/(s.n3*c2a+s.n2*c3a)
    r1p=(s.n2*ca-s.n1*c2a)/(s.n2*ca+s.n1*c2a)
    r2p=(s.n3*c2a-s.n2*c3a)/(s.n3*c2a+s.n2*c3a)

    t1s=2*s.n1*ca/(s.n1*ca+s.n2*c2a)
    t2s=2*s.n2*c2a/(s.n2*c2a+s.n3*c3a)
    r1s=(s.n1*ca-s.n2*c2a)/(s.n1*ca+s.n2*c2a)
    r2s=(s.n2*c2a-s.n3*c3a)/(s.n2*c2a+s.n3*c3a)

    beta=s.k0*s.n2*(s.thick*c2a-s.collar*c2at)

    Tp=t2p*t1p*np.exp(1j*beta)/(1+r1p*r2p*np.exp(2*1j*beta))
    Ts=t2s*t1s*np.exp(1j*beta)/(1+r1s*r2s*np.exp(2*1j*beta))
   
    return Tp, Ts


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
    phi: np.ndarray,
    s: PSFConfig,
):
    if s.mode == Mode.GAUSSIAN:  # gaussian
        mask = 1
    elif s.mode == Mode.DONUT:  # donut
        mask = np.exp(1j * s.vc * phi)
    elif s.mode == Mode.BOTTLE:  # bottle
        cutoff_radius=s.rc*s.r0
        if rho <= cutoff_radius:
            mask = np.exp(1j * s.rc * np.pi)
        else:
            mask = np.exp(1j * 0)
    elif s.mode== Mode.DONUT_BOTTLE: # Donut & Bottle
        raise NotImplementedError("No display Donut and Bottle")
    else:
        raise NotImplementedError("Please use a specified Mode")
    return mask


def generate_psf(s: PSFConfig) -> np.ndarray:

    # Sample Space
    x1 = np.linspace(-s.r0, s.r0, s.Nx)
    y1 = np.linspace(-s.r0, s.r0, s.Ny)
    [X1, Y1] = np.meshgrid(x1, y1)

    x2 = np.linspace(-s.LfocalX, s.LfocalX, s.Nx)
    y2 = np.linspace(-s.LfocalY, s.LfocalY, s.Ny)
    z2 = np.linspace(-s.LfocalZ, s.LfocalZ, s.Nz)
    [X2, Y2, Z2] = np.meshgrid(x2, y2, z2)  # TODO: Needs to be propüerly constructed

    # Step of integral
    deltatheta = s.alpha_int / s.Ntheta
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
        for q in range(0, s.Nphi-1):
            phi = q * deltaphi

            ci = np.cos(phi)
            ca = np.cos(theta)
            si = np.sin(phi)
            sa = np.sin(theta)

            # refracted angles
            theta2=np.arcsin((s.n1/s.n2)*np.sin(theta))
            c2a=np.cos(theta2)
            theta3=np.arcsin((s.n2/s.n3)*np.sin(theta2))
            c3a=np.cos(theta3)
            s3a=np.sin(theta3)

            # Cartesian coordinate on pupil
            x_pup=s.WD*sa*ci
            y_pup=s.WD*sa*si

            # Tilt of the pupil function
            x_pup_t=s.cg*x_pup-s.sg*s.WD
            y_pup_t=y_pup
            theta_t=np.arcsin(np.sqrt(x_pup_t**2+y_pup_t**2)/s.WD)    # Spherical (also = theta-gamma)
            cat=np.cos(theta_t)

            # refracted angles
            theta2_t=np.arcsin((s.n1/s.n2)*np.sin(theta_t))
            c2at=np.cos(theta2_t)
            # theta3_t=np.asin((s.n2/s.n3)*np.sin(theta2_t));
            # c3at=np.cos(theta3_t);

            # Cylindrical coordinates on pupil
            # rho_pupil, phi_pupil = cart_to_polar(x_pup, y_pup)
            # rho_amp, phi_amp = cart_to_polar(
            #    x_pup - s.r0 / s.Nx * s.ampl_offset(1),
            #    y_pup - s.r0 / s.Ny * s.ampl_offset(2),
            #)
            rho_mask, phi_mask = cart_to_polar(
                x_pup_t - s.r0 / s.Nx * s.mask_offset_x,
                y_pup_t - s.r0 / s.Ny * s.mask_offset_y,
            )
            rho_ab, phi_ab = cart_to_polar(
                x_pup_t - s.r0 / s.Nx * s.aberration_offset_x,
                y_pup_t - s.r0 / s.Ny * s.aberration_offset_y,
            )

            if theta_t <= s.alpha:
                # Amplitude profile of the incident beam
                Amp = Amplitude(
                    x_pup_t - s.r0 / s.Nx * s.ampl_offset_x,
                    y_pup_t - s.r0 / s.Ny * s.ampl_offset_y,
                    s,
                )
                Amp = Amp + Noise[slice][q]

                # Phase mask
                PM = phase_mask(
                    rho_mask,
                    phi_mask,
                    s,
                )
                # Wavefront
                W = np.exp(
                    1j #*2*np.pi
                    * zernike(
                        rho_ab / s.r0,
                        phi_ab,
                        s,
                    )
                )
            else:
                Amp=0
                PM=1
                W=0
        
            # incident beam polarization cases
            p0x=[np.cos(s.psi)*np.cos(s.eps)-1j*np.sin(s.psi)*np.sin(s.eps),ci,-si]
            p0y=[np.sin(s.psi)*np.cos(s.eps)+1j*np.cos(s.psi)*np.sin(s.eps),si,ci]
            p0z=0

            # Selected incident beam polarization
            P0 = [
            [p0x[s.polarization - 1]],  # indexing minus one to get corresponding polarization
            [p0y[s.polarization - 1]],  # indexing minus one to get corresponding polarization
            [p0z],]  #


            [Tp,Ts]=Fresnel_coeff(s,ca,c2a,c2at,c3a)

            T=[
                [
                    Tp*c3a*ci**2+Ts*si**2,
                    Tp*si*ci*c3a-Ts*ci*si,
                    Tp*s3a*ci,
                ],
                [
                    Tp*c3a*ci*si-Ts*si*ci,
                    Tp*c3a*si**2+Ts*ci**2,
                    Tp*s3a*si,
                ],
                [
                    -Tp*ci*s3a,
                    -Tp*s3a*si,
                    Tp*c3a,
                ],
            ] # Pola matrix 

            # polarization in focal region
            P = np.matmul(T, P0)
            
            # Apodization factor
            a = np.sqrt(cat)

            # numerical calculation of field distribution in focal region
            propagation = np.exp(
                1j * s.k0 * s.n1 * (X2 * ci * sa + Y2 * si * sa)
                + 1j * s.k0 * s.n3 * c3a * Z2
                ) * deltaphi * deltatheta

            # Aberration term from the coverslip
            Psi_coverslip=s.n3*s.depth*c3a-s.n1*(s.thick+s.depth)*ca
            Psi_collar=-s.n1*s.collar*cat   ### TO DO: ca?
            Psi=Psi_coverslip-Psi_collar
            Ab_wind=np.exp(1j*s.k0*Psi)

            factored = sa * a * Amp * PM * W * Ab_wind * propagation

            Ex2 = Ex2 + factored * P[0, 0]
            Ey2 = Ey2 + factored * P[1, 0]
            Ez2 = Ez2 + factored * P[2, 0]

    Ix2 = np.multiply(np.conjugate(Ex2), Ex2)
    Iy2 = np.multiply(np.conjugate(Ey2), Ey2)
    Iz2 = np.multiply(np.conjugate(Ez2), Ez2)
    I1 = Ix2 + Iy2 + Iz2
    #I1 = np.real(I1)

    I1 = I1 + np.abs(np.random.normal(0, s.detector_gaussian_noise, I1.shape))

    if s.rescale:
        # We are only rescaling to the max, not the min
        I1 = I1 / np.max(I1)

    return np.real(np.moveaxis(I1, 2, 0))
