from enum import Enum
from typing import Callable
import numpy as np
from pydantic import BaseModel, validator, root_validator, validate_model, Field
from functools import cached_property
from annotated_types import Gt, Len, Predicate, BaseMetadata, Lt
from dataclasses import dataclass
from typing import Protocol, TypeVar, Type, Any, Optional, Union, List, Tuple
from typing import Annotated


@dataclass(frozen=True, slots=True)
class Step(BaseMetadata):
    """Gt(gt=x) implies that the value must be greater than x.

    It can be used with any type that supports the ``>`` operator,
    including numbers, dates and times, strings, sets, and so on.
    """

    x: int


@dataclass(frozen=True, slots=True)
class Slider(BaseMetadata):
    """Gt(gt=x) implies that the value must be greater than x.

    It can be used with any type that supports the ``>`` operator,
    including numbers, dates and times, strings, sets, and so on.
    """

    step: int


class Mode(str, Enum):
    GAUSSIAN = "GAUSSIAN"
    DONUT = "DONUT"
    BOTTLE = "BOTTLE"
    DONUT_BOTTLE = "DONUT_BOTTLE"


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


# class EffectivePSFGeneratorConfig(BaseModel):
#    I_sat: float = 0.1  # Saturation factor of depletion


# class WindowConfig(BaseModel):
#    wind: WindowType = WindowType.NEW
#    r_window = 2.3e-3  # radius of the cranial window (in m)
#    t_window = 2.23e-3  # thickness of the cranial window (in m)


# class CoverslipConfig(BaseModel):
# Coverslip Parameters
#   refractive_index_immersion = 1.33 # refractive index of the immersion medium
#   refractive_index_coverslip = 1.52  # refractive index of the coverslip
#   refractive_index_sample = 1.38  # refractive index of immersion medium (BRIAN)

#    imaging_depth = 10e-6  # from coverslip down
#    thickness_coverslip = 100e-6  # thickness of coverslip in meter


class PSFConfig(BaseModel):
    mode: Mode = Mode.GAUSSIAN
    polarization: Polarization = Polarization.ELLIPTICAL

    # Geometry parameters
    NA: Annotated[float, Step(0.1)] = Field(
        default=1, description="LABEL: numerical aperture of objective lens", gt=0.4, lt=1.4
    )
    WDMM: float = 2800  # working distance of the objective in meter
    n1: float = 1.33  # refractive index of immersion medium
    n2: float = 1.52  # refractive index of the glass coverslip
    n3: float = 1.38  # refractive index of the Sample
    thickMM: float = 170  # Thickness of the coverslip
    collarMM: float = Field(
        default=170, description="Correction Collar: The value will", gt=0, lt=400
    )
    depthMM: float = Field(
        default=10, description="Tilt of the coverslip in angle", gt=0, lt=150
    )

    # Beam parameters
    wavelengthMM: float = 0.592  # wavelength of light in meter
    waistMM: float = Field(
        default=8000, description="Tilt of the coverslip in angle", gt=0, lt=20000
    )
    ampl_offset_x: float = (
        0  # offset of the amplitude profile in regard to pupil center
    )
    ampl_offset_y: float = 0

    # Polarization parameters
    psi_degree: float = Field(
        default=0, description="Tilt of the coverslip in angle", gt=0, lt=180
    )
    eps_degree: float = Field(
        default=45, description="Tilt of the coverslip in angle", gt=-45, lt=45
    )
    tilt_angle_degree: float = Field(
        default=0, description="Tilt of the coverslip in angle", gt=-10, lt=10
    )

    # STED parameters
    I_sat: float = 0.1  # Saturation factor of depletion
    ring_radius: float = 0.46  # radius of the ring phase mask (on unit pupil)
    vc: float = 1.0  # vortex charge (should be integer to produce donut) # TODO: topological charge
    rc: float = 1.0  # ring charge (should be integer to produce donut)
    mask_offset_x: float = 0  # offset of the phase mask in regard of the pupil center
    mask_offset_y: float = 0
    p: float = Field(
        default=0.5, description="Tilt of the coverslip in angle", gt=0, lt=1
    )

    # Aberration
    a1: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a2: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a3: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a4: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a5: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a6: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a7: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a8: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a9: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a10: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    a11: AberrationFloat = Field(default=0, description="piston", gt=-1, lt=1)
    aberration_offset_x: float = (
        0  # offset of the aberration in regard of the pupil center
    )
    aberration_offset_y: float = 0

    # sampling parameters
    LfocalXMM: float = 1.5  # observation scale X (in m)
    LfocalYMM: float = 1.5  # observation scale Y (in m)
    LfocalZMM: float = 2  # observation scale Z (in m)
    Nx: int = 31  # discretization of image plane - better be odd number
    Ny: int = 31
    Nz: int = 31
    Ntheta: int = 31  # integration step
    Nphi: int = 31

    # Noise Parameters
    gaussian_beam_noise: float = 0.0
    detector_gaussian_noise: float = 0.0

    add_detector_poisson_noise: bool = False  # standard deviation of the noise

    # Normalization
    rescale: bool = True  # rescale the PSF to have a maximum of 1

    @property
    def WD(self):
        return self.WDMM / 1000000

    @property
    def wavelength(self):
        return self.wavelengthMM / 1000000

    @property
    def waist(self):
        return self.waistMM / 1000000

    @property
    def collar(self):
        return self.collarMM / 1000000

    @property
    def thick(self):
        return self.thickMM / 1000000

    @property
    def depth(self):
        return self.depthMM / 1000000

    @property
    def LfocalX(self):
        return self.LfocalXMM / 1000000

    @property
    def LfocalZ(self):
        return self.LfocalZMM / 1000000

    @property
    def LfocalY(self):
        return self.LfocalYMM / 1000000


    @property
    def k0(self):
        return 2 * np.pi / self.wavelength

    @property
    def alpha(self):
        return np.arcsin(
            self.NA / self.n1
        )  # maximum focusing angle of the objective (in rad)

    @property
    def r0(self):
        return self.WD * np.sin(self.alpha)  # radius of the pupil (in m)

    # convert angle in red
    @property
    def gamma(self):
        return self.tilt_angle_degree * np.pi / 180  # tilt angle (in rad)

    @property
    def psi(self):
        return self.psi_degree * np.pi / 180  # polar direction

    @property
    def eps(self):
        return self.eps_degree * np.pi / 180  # ellipticity

    @property
    def sg(self):
        return np.sin(self.gamma)

    @property
    def cg(self):
        return np.cos(self.gamma)

    @property
    def alpha_int(self):
        return self.alpha + abs(self.gamma)  # Integration range (in rad)

    @property
    def r0_int(self):
        return self.WD * np.sin(self.alpha_int)  # integration radius on pupil (in m)

    @property
    def alpha2(self):
        return np.arcsin((self.n1 / self.n2) * np.sin(self.alpha))

    @property
    def alpha3(self):
        return np.arcsin((self.n2 / self.n3) * np.sin(self.alpha2))

    @property
    def Dfoc(self):
        return 0.053 * self.depth + 0.173 * (
            self.thick - self.collar
        )  # No aberration correction

    @property
    def deltatheta(self):
        return self.alpha_int / self.Ntheta

    @property
    def deltaphi(self):
        return 2 * np.pi / self.Nphi

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

    class Config:
        validate_assignment = True
        extra = "forbid"


PSFGenerator = Callable[[PSFConfig], np.ndarray]
