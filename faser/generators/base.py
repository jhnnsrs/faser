from enum import Enum
from typing import Callable
import numpy as np
from pydantic import BaseModel, validator, root_validator, validate_model


class Mode(int, Enum):
    GAUSSIAN = 1
    DONUT = 2
    BOTTLE = 3
    DONUT_BOTTLE = 4


class WindowType(str, Enum):
    OLD = "OLD"
    NEW = "NEW"
    NO = "NO"


class Polarization(int, Enum):
    ELLIPTICAL = 1
    RADIAL = 2
    AZIMUTHAL = 3


class AberrationFloat(float):
    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, float):
            raise TypeError("Float required")
        # you could also return a string here which would mean model.post_code
        # would be a string, pydantic won't care but you could end up with some
        # confusion since the value's type won't match the type annotation
        # exactly
        return v


class Aberration(BaseModel):
    a1: AberrationFloat = 0
    a2: AberrationFloat = 0
    a3: AberrationFloat = 0
    a4: AberrationFloat = 0
    a5: AberrationFloat = 0
    a6: AberrationFloat = 0
    a7: AberrationFloat = 0
    a8: AberrationFloat = 0
    a9: AberrationFloat = 0
    a10: AberrationFloat = 0
    a11: AberrationFloat = 0

    def to_name(self) -> str:
        return "_".join(
            map(lambda value: f"{value[0]}-{value[1]}", self.dict().items())
        )


#class EffectivePSFGeneratorConfig(BaseModel):
#    I_sat: float = 0.1  # Saturation factor of depletion


#class WindowConfig(BaseModel):
#    wind: WindowType = WindowType.NEW
#    r_window = 2.3e-3  # radius of the cranial window (in m)
#    t_window = 2.23e-3  # thickness of the cranial window (in m)


#class CoverslipConfig(BaseModel):
    # Coverslip Parameters
 #   refractive_index_immersion = 1.33 # refractive index of the immersion medium
 #   refractive_index_coverslip = 1.52  # refractive index of the coverslip
 #   refractive_index_sample = 1.38  # refractive index of immersion medium (BRIAN)

#    imaging_depth = 10e-6  # from coverslip down
#    thickness_coverslip = 100e-6  # thickness of coverslip in meter


class PSFConfig(BaseModel):
    mode: Mode = Mode.GAUSSIAN
    polarization: Polarization = Polarization.ELLIPTICAL

    # Window Type?
    wind: WindowType = WindowType.NEW
  
    # Geometry parameters
    NA: float = 1.0  # numerical aperture of objective lens
    WD = 2.8e-3  # working distance of the objective in meter
    n1 = 1.33  # refractive index of immersion medium
    n2=1.52 # refractive index of the glass coverslip
    n3=1.38 # refractive index of the Sample
    collar = 170e-6  # thickness of the coverslip corrected by the collar
    thick=170e-6    # Thickness of the coverslip
    depth=10e-6     # Depth in the sample
    gamma=1 # Tilt angle (in °)

    # Beam parameters
    wavelength = 592e-9  # wavelength of light in meter
    waist = 8e-3
    ampl_offset_x: float = 0  # offset of the amplitude profile in regard to pupil center
    ampl_offset_y: float = 0

    # Polarization parameters
    psi=0   # Direction of elliptical polar (0: horizontal, 90 vertical)
    eps=45  # Ellipticity (-45: right-handed circular polar, 0: linear, 45: left-handed circular)

    # STED parameters
    I_sat = 0.1  # Saturation factor of depletion
    ring_radius = 0.46  # radius of the ring phase mask (on unit pupil)
    vc: float = 1.0  # vortex charge (should be integer to produce donut) # TODO: topological charge
    rc: float = 1  # ring charge (should be integer to produce donut)
    mask_offset_x = 0  # offset of the phase mask in regard of the pupil center
    mask_offset_y = 0
    p=0.5   # intensity repartition in donut and bottle beam (p donut, (1-p) bottle)

    # Aberration
    aberration: Aberration = Aberration()
    aberration_offset_x = 0  # offset of the aberration in regard of the pupil center
    aberration_offset_y= 0

    # sampling parameters
    LfocalX = 1.5e-6  # observation scale X (in m)
    LfocalY = 1.5e-6  # observation scale Y (in m)
    LfocalZ = 2e-6  # observation scale Z (in m)
    Nx = 31  # discretization of image plane - better be odd number
    Ny = 31
    Nz = 31
    Ntheta = 31 # integration step
    Nphi = 31
    threshold=0.001
    it=1

    # Calculated Parameters
    k0 = 2 * np.pi / wavelength  # wavenumber (m^-1)
    alpha = np.arcsin(NA / n1)  # maximum focusing angle of the objective (in rad)
    r0 = WD * np.sin(alpha)  # radius of the pupil (in m)

    # convert angle in red
    gamma=gamma*np.pi/180 # tilt angle (in rad)
    psi=psi*np.pi/180 # polar direction
    eps=eps*np.pi/180 # ellipticity

    sg=np.sin(gamma)
    cg=np.cos(gamma)

    alpha_int=alpha+abs(gamma) # Integration range (in rad)
    r0_int=WD*np.sin(alpha_int)  # integration radius on pupil (in m)
    #waist=waist*r0

    #e_wind = 2.23e-3    # thichkness of the window (in m)
    #if wind == 1
    #   r_wind=1.5e-3 # radius of the old cranial window (in m)
    #elseif wind==2
    #    r_wind=2.3e-3   # radius of the new cranial window (in m)
    #elseif wind==3
    #    r_wind=1000*e_wind
    #else
    #    disp('unknown request');
    
    # Impact of the cranial window
    #alpha_eff=min(np.atan(r_wind/e_wind),alpha)  # effective focalization angle in presence of the cranial window
    #NA_eff=min(n1*sin(alpha_eff),NA)    # Effective NA in presence of the cranial window
    #r0_eff=WD*sin(alpha_eff)    # Effective pupil radius 'in m)
    
    # Corrected focus position
    alpha2=np.arcsin((n1/n2)*np.sin(alpha))
    alpha3=np.arcsin((n2/n3)*np.sin(alpha2))
    # Cp.Dfoc=Sp.depth*(tan(Cp.alpha_eff)/tan(Cp.alpha3_eff)-1)+Sp.thick*(tan(Cp.alpha_eff)-tan(Cp.alpha2_eff))/tan(Cp.alpha3_eff);
    Dfoc=0.053*depth+0.173*(thick-collar) # No aberration correction
    # Cp.Dfoc=0.053*Sp.depth+0.4*(Sp.thick); % No aberration correction

    # Step and range of integral
    deltatheta=alpha_int/Ntheta
    deltaphi=2*np.pi/Nphi

    # Noise Parameters
    gaussian_beam_noise = 0.0
    detector_gaussian_noise = 0.0

    add_detector_poisson_noise = False  # standard deviation of the noise

    # Normalization
    rescale = True  # rescale the PSF to have a maximum of 1

    @root_validator
    def validate_NA(cls, values):
        NA = values["NA"]
        if NA <= 0:
            raise ValueError("numerical_aperature must be positive")
        if values["n1"] < NA:
            raise ValueError(
                "numerical_aperature must be smaller than the refractive index"
            )

        return values
    

PSFGenerator = Callable[[PSFConfig], np.ndarray]
