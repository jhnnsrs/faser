from enum import Enum
from pydantic import BaseModel, validator, root_validator, validate_model


class Mode(int, Enum):
    GAUSSIAN = 1
    DONUT = 2
    BOTTLE = 3


class WindowType(int, Enum):
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


class Aberration(BaseModel):
    a1: float = 0
    a2: float = 0
    a3: float = 0
    a4: float = 0
    a5: float = 0
    a6: float = 0
    a7: float = 0
    a8: float = 0
    a9: float = 0
    a10: float = 0
    a11: float = 0

    def to_name(self) -> str:
        return "_".join(
            map(lambda value: f"{value[0]}-{value[1]}", self.dict().items())
        )


class EffectivePSFGeneratorConfig(BaseModel):
    Isat: float = 0.15  # Saturation factor of depletion


class PSFGeneratorConfig(BaseModel):
    mode: Mode = Mode.GAUSSIAN
    polarization: Polarization = Polarization.LEFT_CIRCULAR

    # Window Type?
    wind: WindowType = WindowType.NEW

    # Geometry parameters
    numerical_aperature: float = 1.0  # numerical aperture of objective lens
    working_distance = 2.8e-3  # working distance of the objective in meter
    refractive_index_immersion = 1.33  # refractive index of immersion medium
    refractive_index_coverslip = 1.5  # refractive index of immersion medium
    refractive_index_sample = 1.38  # refractive index of immersion medium (BRIAN)

    imaging_depth = 10e-6  # from coverslip down
    thickness_coverslip = 100e-6  # thickness of coverslip in meter

    radius_window = 2.3e-3  # radius of the cranial window (in m)
    thicknes_window = 2.23e-3  # thickness of the cranial window (in m)

    # Beam parameters
    wavelength = 592e-9  # wavelength of light in meter
    beam_waist = 0.008
    ampl_offsetX = (
        0.0  # offset of the amplitude profile in regard to pupil center in x direction
    )
    ampl_offsetY = 0.0  # offset of the amplitude profile in regard to pupil center i in y direction

    # Phase Pattern
    unit_phase_radius = 0.45  # radius of the ring phase mask (on unit pupil)
    vortex_charge: float = 1.0  # vortex charge (should be integer to produce donut) # TODO: topological charge
    ring_charge: float = 1  # ring charge (should be integer to produce donut)
    mask_offsetX: float = 0.0  # offset of the phase mask in x direction
    mask_offsetY: float = 0.0  # offset of the phase mask in y direction

    aberration: Aberration = Aberration()

    # sampling parameters
    LfocalX = 3e-6  # observation scale X
    LfocalY = 3e-6  # observation scale Y
    LfocalZ = 10e-6  # observation scale Z
    Nx = 64  # discretization of image plane
    Ny = 64
    Nz = 64
    Ntheta = 40
    Nphi = 40

    @root_validator
    def validate_numerical_aperature(cls, values):
        numerical_aperature = values["numerical_aperature"]
        if numerical_aperature <= 0:
            raise ValueError("numerical_aperature must be positive")
        if values["refractive_index"] < values["numerical_aperature"]:
            raise ValueError(
                "numerical_aperature must be smaller than the refractive index"
            )

        return values


class PSFGenerator:
    def __init__(self, config: PSFGeneratorConfig) -> None:
        self.config = config

    def generate(self):
        raise NotImplementedError()
