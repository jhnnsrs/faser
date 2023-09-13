from arkitekt import register
from faser.generators.base import PSFConfig, Mode, Polarization, Aberration
from mikro.api.schema import RepresentationFragment, from_xarray, from_df
from pandas import json_normalize
from faser.generators.vectorial.stephane.tilted_coverslip import generate_psf


@register
def generate_psf_arkitekt(
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
    tilt_angle=0.0,
    gaussian_beam_noise=0.0,
    detector_gaussian_noise=0.0,
    add_detector_poisson_noise=False,
    rescale=True,
    mode: Mode = Mode.GAUSSIAN,
    polarization: Polarization = Polarization.ELLIPTICAL,
    psi=0,
    eps=45,
) -> RepresentationFragment:
    """Generate a PSF

    Generates a PSF utilizing common parameters for the PSF generation.
    """

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
        # wavelength=wavelength*1e-9,
        # waist=waist*1e-3,
        # ampl_offset=ampl_offset,
        psi_degree=psi,
        eps_degree=eps,
        # aberration_offset=aberration_offset,
        # vc=vc,
        # rc=rc,
        # ring_radius=ring_radius,
        # mask_offset=mask_offset,
        # p=p,
        # NA=NA,
        # WD=WD*1.e-3,
        # n1=n1,
        # n2=n2,
        # n3=n3,
        # thick=thick*1e-6,
        # depth=depth*1e-6,
        tilt_angle_degree=tilt_angle,
        # wind=wind,
        # t_wind=t_wind*1e-3,
        # r_wind=r_wind*1e-3,
    )

    aberrationdf = from_df(json_normalize(config.dict()), name="Abberation (dataframe)")

    psf = generate_psf(config)

    return from_xarray(psf, name="PSF (xarray)", table_origins=[aberrationdf])
