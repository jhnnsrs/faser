from enum import Enum
from typing import Callable
import numpy as np
from pydantic import BaseModel, validator, root_validator, validate_model, Field
from functools import cached_property

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
    WD: float = 2.8e-3  # working distance of the objective in meter
    n1: float = 1.33  # refractive index of immersion medium
    n2: float=1.52 # refractive index of the glass coverslip
    n3: float =1.38 # refractive index of the Sample
    collar: float= 170e-6  # thickness of the coverslip corrected by the collar
    thick: float =170e-6    # Thickness of the coverslip
    depth: float =10e-6     # Depth in the sample

    # Beam parameters
    wavelength: float = 592e-9  # wavelength of light in meter
    waist: float = 8e-3
    ampl_offset_x: float = 0  # offset of the amplitude profile in regard to pupil center
    ampl_offset_y: float = 0

    # Polarization parameters
    psi_degree: float=0   # Direction of elliptical polar (0: horizontal, 90 vertical)
    eps_degree: float=45  # Ellipticity (-45: right-handed circular polar, 0: linear, 45: left-handed circular)
    tilt_angle_degree: float=0  # Tilt angle (in °)
   
    # STED parameters
    I_sat: float = 0.1  # Saturation factor of depletion
    ring_radius: float = 0.46  # radius of the ring phase mask (on unit pupil)
    vc: float = 1.0  # vortex charge (should be integer to produce donut) # TODO: topological charge
    rc: float = 1  # ring charge (should be integer to produce donut)
    mask_offset_x: float = 0  # offset of the phase mask in regard of the pupil center
    mask_offset_y: float = 0
    p: float =0.5   # intensity repartition in donut and bottle beam (p donut, (1-p) bottle)

    # Aberration
    aberration: Aberration = Field(default_factory=Aberration)
    aberration_offset_x: float = 0  # offset of the aberration in regard of the pupil center
    aberration_offset_y: float= 0

    # sampling parameters
    LfocalX: float = 1.5e-6  # observation scale X (in m)
    LfocalY: float = 1.5e-6  # observation scale Y (in m)
    LfocalZ: float = 2e-6  # observation scale Z (in m)
    Nx: int = 31  # discretization of image plane - better be odd number
    Ny: int = 31
    Nz: int = 31
    Ntheta: int = 31 # integration step
    Nphi: int = 31
    threshold: float=0.001
    it: int =1

   

    # Noise Parameters
    gaussian_beam_noise: float = 0.0
    detector_gaussian_noise: float = 0.0

    add_detector_poisson_noise: bool = False  # standard deviation of the noise

    # Normalization
    rescale: bool = True  # rescale the PSF to have a maximum of 1


    @property
    def k0(self):
        return 2 * np.pi / self.wavelength

    @property
    def alpha(self):
        return np.arcsin(self.NA / self.n1)  # maximum focusing angle of the objective (in rad)
   
    @property
    def r0(self):
        return self.WD * np.sin(self.alpha)  # radius of the pupil (in m)

  # convert angle in red
    @property
    def gamma(self):
        return self.tilt_angle_degree*np.pi/180 # tilt angle (in rad)
    
    @property
    def psi(self):
        return self.psi_degree * np.pi/180 # polar direction
    
    @property
    def eps(self):
        return self.eps_degree * np.pi/180 # ellipticity
    
    
    @property
    def sg(self):
        return np.sin(self.gamma)
    
    @property
    def cg(self):
        return np.cos(self.gamma)
    
    @property
    def alpha_int(self):
        return self.alpha+abs(self.gamma) # Integration range (in rad)
    
    @property
    def r0_int(self):
        return self.WD*np.sin(self.alpha_int)  # integration radius on pupil (in m)
    
    @property
    def alpha2(self):
        return np.arcsin((self.n1/self.n2)*np.sin(self.alpha))
    
    @property
    def alpha3(self):
        return np.arcsin((self.n2/self.n3)*np.sin(self.alpha2))
    
    @property
    def Dfoc(self):
        return 0.053*self.depth+0.173*(self.thick-self.collar) # No aberration correction
    
    @property
    def deltatheta(self):
        return self.alpha_int/self.Ntheta
    
    @property
    def deltaphi(self):
        return 2*np.pi/self.Nphi
 



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
