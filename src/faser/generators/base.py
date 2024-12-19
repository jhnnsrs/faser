from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import (
    Annotated,
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from annotated_types import BaseMetadata, Gt, Len, Lt, Predicate
from psygnal import evented
from pydantic import BaseModel, Field, field_validator, model_validator


@dataclass(frozen=True, slots=True)
class Step(BaseMetadata):
    """Gt(gt=x) implies that the value must be greater than x.

    It can be used with any type that supports the ``>`` operator,
    including numbers, dates and times, strings, sets, and so on.
    """

    x: int


class mode(str, Enum):
    GAUSSIAN = "GAUSSIAN"
    DONUT = "DONUT"
    BOTTLE = "BOTTLE"
    DONUT_BOTTLE = "DONUT BOTTLE"
    LOADED = "LOADED"


class window(str, Enum):
    NO = "NO"
    CUSTOM = "CUSTOM"


class normalize(str, Enum):
    YES = "YES"
    NO = "NO"


class noise(str, Enum):
    YES = "YES"
    NO = "NO"


class polarization(int, Enum):
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
    def validate(cls, v, info):
        if isinstance(v, int):
            v = float(v)

        if not isinstance(v, float):
            raise TypeError("Float required, got {}".format(type(v)))
        return v


@evented
class PSFConfig(BaseModel):
    # sampling parameters
    L_obs_XY: float = Field(default=2.0, description="Observation scale in XY (in µm)")
    L_obs_Z: float = Field(default=2.0, description="Observation scale in Z (in µm)")
    Nxy: int = Field(
        default=31,
        description="Discretization of image plane - better be odd number for perfect 0",
    )
    Nz: int = Field(
        default=31,
        description="Discretization of Z axis - better be odd number for perfect 0",
    )
    Ntheta: int = Field(
        default=31, description="Integration sted of the focalization angle"
    )
    Nphi: int = Field(
        default=31, description="Integration sted of the aximutal angle on the pupil"
    )

    # Normalization
    Normalize: normalize = (
        normalize.NO
    )  # Normalize the intensity of the PSF to have a maximum of 1

    # Geometry parameters
    NA: float = Field(default=1, description="Numerical Aperture of Objective Lens")
    WD: float = Field(
        default=2800.0, description="Working Distance of the objective lens (in µm)"
    )
    n1: float = Field(
        default=1.33, description="Refractive index of the immersion medium"
    )
    n2: float = Field(default=1.52, description="Refractive index of the coverslip")
    n3: float = Field(default=1.38, description="Refractive index of the sample")
    Thickness: float = Field(
        default=170.0, description="Thickness of the coverslip (in µm)", ge=0, lt=300
    )
    Collar: float = Field(
        default=170.0,
        description="Correction collar setting to compensate coverslip thickness",
        gt=0,
        lt=300,
    )
    Depth: float = Field(
        default=0, description="Imaging depth in the sample (in µm)", ge=0, lt=1000
    )
    Tilt: float = Field(
        default=0.0, description="Tilt angle of the coverslip (in °)", ge=-20, lt=20
    )

    Window: window = window.NO

    Wind_Radius: float = Field(
        default=2.3, description="Diameter of the cranial window (in mm)"
    )
    Wind_Depth: float = Field(
        default=2.23, description="Depth of the cranial window (in mm)"
    )

    Wind_Offset_x: float = Field(
        default=0.0,
        description="X offset of the cranial window in regard to pupil center",
    )
    Wind_Offset_y: float = Field(
        default=0.0,
        description="Y offset of the cranial window in regard to pupil center",
    )

    # Aberrations
    a0: AberrationFloat = Field(default=0.0, description="Piston", ge=-1, le=1)
    a1: AberrationFloat = Field(default=0.0, description="Vertical Tilt", ge=-1, le=1)
    a2: AberrationFloat = Field(default=0.0, description="Horizontal Tilt", ge=-1, le=1)
    a3: AberrationFloat = Field(
        default=0, description="Oblique Astigmatism", ge=-1, le=1
    )
    a4: AberrationFloat = Field(default=0.0, description="Defocus", ge=-1, le=1)
    a5: AberrationFloat = Field(
        default=0, description="Vertical Astigmatism", ge=-1, le=1
    )
    a6: AberrationFloat = Field(
        default=0.0, description="Vertical Trefoil", ge=-1.0, le=1
    )
    a7: AberrationFloat = Field(default=0.0, description="Vertical Coma", ge=-1, le=1)
    a8: AberrationFloat = Field(default=0.0, description="Horizontal Coma", ge=-1, le=1)
    a9: AberrationFloat = Field(default=0.0, description="Oblique Trefoil", ge=-1, le=1)
    a12: AberrationFloat = Field(
        default=0.0, description="Primary spherical", ge=-1, le=1
    )
    a24: AberrationFloat = Field(
        default=0.0, description="Secondary spherical", ge=-1, le=1
    )
    Aberration_offset_x: float = Field(
        default=0.0,
        description="X offset of the aberration function in regard to pupil center",
    )
    Aberration_offset_y: float = Field(
        default=0.0,
        description="Y offset of the aberration function in regard to pupil center",
    )

    # Beam parameters
    Mode: mode = mode.GAUSSIAN
    Polarization: polarization = polarization.ELLIPTICAL
    Wavelength: float = Field(default=0.592, description="Wavelength of light (in µm)")
    Waist: float = Field(
        default=8000.0,
        description="Diameter of the input beam on the objective pupil (in µm)",
        gt=0,
        lt=25000,
    )
    Ampl_offset_x: float = Field(
        default=0.0,
        description="X offset of the amplitude profile in regard to pupil center",
    )
    Ampl_offset_y: float = Field(
        default=0.0,
        description="Y offset of the amplitude profile in regard to pupil center",
    )

    # Polarization parameters
    Psi: float = Field(
        default=0.0, description="Direction of the polarization (in °)", ge=0, le=180
    )
    Epsilon: float = Field(
        default=45.0,
        description="Ellipticity of the polarization (in °)",
        ge=-45,
        le=45,
    )

    # STED parameters
    VC: float = Field(
        default=1.0,
        description="Vortex charge (should be integer to produce donut)",
        gt=-6,
        lt=6,
    )
    RC: float = Field(
        default=1.0,
        description="Ring charge (should be odd to produce bottle)",
        gt=-6,
        lt=6,
    )
    Ring_Radius: float = Field(
        default=0.707,
        description="Radius of the ring phase mask (on unit pupil)",
        gt=0,
        lt=1,
    )
    Mask_offset_x: float = Field(
        default=0.0, description="X offset of the phase mask in regard to pupil center"
    )
    Mask_offset_y: float = Field(
        default=0.0, description="Y offset of the phase mask in regard to pupil center"
    )
    p: float = Field(
        default=0.5,
        description="Ratio between Donut (p) and Bottle (1-p) intensity",
        gt=0,
        lt=1,
    )

    loaded_phase_mask: Optional[np.ndarray] = Field(
        default=None, description="Loaded Phasemak"
    )

    # Noise Parameters
    Add_noise: noise = noise.YES  # Add noise to the PSF

    Gaussian_beam_noise: Annotated[float, Step(0.1)] = Field(
        default=0.0, description="Gaussian_beam noise", ge=0, lt=1
    )
    Detector_gaussian_noise: Annotated[float, Step(0.1)] = Field(
        default=0.0, description="Detector Gaussian noise", ge=0, lt=1
    )
    # Add_detector_poisson_noise: bool = False  # standard deviation of the noise

    @property
    def t_wind(self):
        if self.Window == window.CUSTOM:
            return self.Wind_Depth * 1e3
        else:
            return 2.23e3

    @property
    def r_wind(self):  # No cranial window
        if self.Window == window.NO:
            return 100 * self.t_wind
        elif self.Window == window.CUSTOM:
            return self.Wind_Radius * 1e3
        else:
            raise NotImplementedError("Please use a specified window type")

    @property
    def k0(self):  # Wavevector (in µm^-1)
        return 2 * np.pi / self.Wavelength

    @property
    def alpha(self):  # maximum focusing angle of the objective (in rad)
        return np.arcsin(self.NA / self.n1)

    @property
    def r0(self):  # radius of the pupil (in µm)
        return self.WD * np.sin(self.alpha)

    # convert angle in red
    @property
    def gamma(self):  # tilt angle (in rad)
        return self.Tilt * np.pi / 180

    @property
    def psi(self):  # polar direction (in rad)
        return self.Psi * np.pi / 180

    @property
    def eps(self):  # ellipticity (in rad)
        return self.Epsilon * np.pi / 180

    @property
    def sg(self):
        return np.sin(self.gamma)

    @property
    def cg(self):
        return np.cos(self.gamma)

    @property
    def alpha_eff(
        self,
    ):  # effective focalization angle due to the cranial window (in rad)
        return min(np.arctan(self.r_wind / self.t_wind), self.alpha)

    @property
    def NA_eff(self):  # Effective NA in presence of the cranial window
        return min(self.n1 * np.sin(self.alpha_eff), self.NA)

    @property
    def r0_eff(self):  # Effective pupil radius (in µm)
        return self.WD * np.sin(self.alpha_eff)

    @property
    def alpha_int(self):  # Integration range (in rad)
        return self.alpha_eff + abs(self.gamma)

    @property
    def r0_int(self):  # integration radius on pupil (in µm)
        return self.WD * np.sin(self.alpha_int)

    @property
    def alpha2_eff(self):
        return np.arcsin((self.n1 / self.n2) * np.sin(self.alpha_eff))

    @property
    def alpha3_eff(self):
        return np.arcsin((self.n2 / self.n3) * np.sin(self.alpha2_eff))

    @property
    def Dfoc(self):  # Corrected focus position
        # return 0.0564 * self.Depth + 0.1692 * (self.Thickness - self.Collar)  # No aberration correction
        return (
            1
            / np.tan(self.alpha3_eff)
            * (
                self.Depth * (np.tan(self.alpha_eff) - np.tan(self.alpha3_eff))
                + (self.Thickness - self.Collar)
                * (np.tan(self.alpha_eff) - np.tan(self.alpha2_eff))
            )
        )

    @property
    def deltatheta(self):
        return self.alpha_int / self.Ntheta

    @property
    def deltaphi(self):
        return 2 * np.pi / self.Nphi

    @model_validator(mode="after")
    def validate_NA(cls, self: "PSFConfig", info):
        NA = self.NA
        if NA <= 0:
            raise ValueError("numerical_aperature must be positive")
        if self.n1 < NA:
            raise ValueError(
                "numerical_aperature must be smaller than the refractive index"
            )
        if self.Mode == mode.LOADED:
            if self.loaded_phase_mask is None:
                raise ValueError("You need to load a phase mask to use the loaded mode")

        return self

    class Config:
        validate_assignment = True
        extra = "forbid"
        arbitrary_types_allowed = True


PSFGenerator = Callable[[PSFConfig], np.ndarray]
