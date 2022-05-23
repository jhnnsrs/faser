from enum import Enum
from pydantic import BaseModel, validator, root_validator, validate_model


class Mode(int, Enum):
    GAUSSIAN = 1
    DONUT = 2
    BOTTLE = 3


class WindowType(str, Enum):
    OLD = "OLD"
    NEW = "NEW"
    NO = "NO"


class Polarization(int, Enum):
    X_LINEAR = 1
    Y_LINEAR = 2
    LEFT_CIRCULAR = 3
    RIGHT_CIRCULAR = 4
    ELLIPTICAL = 5
    RADIAL = 6
    AZIMUTHAL = 7


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

        if v < 0:
            raise ValueError("Must be positive")

        if v > 1:
            raise ValueError("Must be less than 1")
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


class EffectivePSFGeneratorConfig(BaseModel):
    Isat: float = 0.15  # Saturation factor of depletion


class WindowConfig(BaseModel):
    wind: WindowType = WindowType.NEW
    radius_window = 2.3e-3  # radius of the cranial window (in m)
    thicknes_window = 2.23e-3  # thickness of the cranial window (in m)


class CoverslipConfig(BaseModel):
    # Coverslip Parameters
    refractive_index_coverslip = 1.5  # refractive index of immersion medium
    refractive_index_sample = 1.38  # refractive index of immersion medium (BRIAN)

    imaging_depth = 10e-6  # from coverslip down
    thickness_coverslip = 100e-6  # thickness of coverslip in meter


class PSFConfig(BaseModel):
    mode: Mode = Mode.GAUSSIAN
    polarization: Polarization = Polarization.LEFT_CIRCULAR

    # Window Type?
    wind: WindowType = WindowType.NEW

    # Geometry parameters
    numerical_aperature: float = 1.0  # numerical aperture of objective lens
    working_distance = 2.8e-3  # working distance of the objective in meter
    refractive_index_immersion = 1.33  # refractive index of immersion medium

    # Beam parameters
    wavelength = 592e-9  # wavelength of light in meter
    beam_waist = 8e-3

    ampl_offsetX = (
        0.0  # offset of the amplitude profile in regard to pupil center in x direction
    )

    ampl_offsetY = 0.0  # offset of the amplitude profile in regard to pupil center i in y direction

    # STED parameters
    saturation_factor = 0.1  # Saturation factor of depletion

    # Phase Pattern
    unit_phase_radius = 0.46  # radius of the ring phase mask (on unit pupil)
    vortex_charge: float = 1.0  # vortex charge (should be integer to produce donut) # TODO: topological charge
    ring_charge: float = 1  # ring charge (should be integer to produce donut)
    mask_offsetX: float = 0.0  # offset of the phase mask in x direction
    mask_offsetY: float = 0.0  # offset of the phase mask in y direction

    # Aberration
    aberration: Aberration = Aberration()
    aberration_offsetX: float = 0.0  # offset of the aberration in x direction
    aberration_offsetY: float = 0.0  # offset of the aberration in y direction

    # sampling parameters
    LfocalX = 1e-6  # observation scale X
    LfocalY = 1e-6  # observation scale Y
    LfocalZ = 1e-6  # observation scale Z
    Nx = 32  # discretization of image plane
    Ny = 32
    Nz = 32
    Ntheta = 40
    Nphi = 40

    @root_validator
    def validate_numerical_aperature(cls, values):
        numerical_aperature = values["numerical_aperature"]
        if numerical_aperature <= 0:
            raise ValueError("numerical_aperature must be positive")
        if values["refractive_index_immersion"] < values["numerical_aperature"]:
            raise ValueError(
                "numerical_aperature must be smaller than the refractive index"
            )

        return values


class PSFGenerator:
    def __init__(self, config: PSFConfig) -> None:
        self.config = config

    def generate(self):
        raise NotImplementedError()
