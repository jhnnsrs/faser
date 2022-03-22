from faser.generators.base import PSFGenerator
from faser.generators.scalar.phasenet.generator import PsfGenerator3D
from .wavefront import ZernikeWavefront


class PhaseNetPSFGenerator(PSFGenerator):
    def generate(self):

        psf = PsfGenerator3D(
            psf_shape=(self.config.Nx, self.config.Ny, self.config.Nz),
            units=(0.1, 0.1, 0.1),  # TODO: Figure out how to solve?
            na_detection=self.config.numerical_aperature,
            lam_detection=self.config.wavelength * 10e5,  # meter to microns
            n=self.config.refractive_index,
        )

        print(self.config.wavelength * 10e5)

        # Aberration to ansi
        params = {
            int(key[1:]): value for key, value in self.config.aberration.dict().items()
        }

        wf = ZernikeWavefront(params, order="ansi")

        return psf.incoherent_psf(wf)
