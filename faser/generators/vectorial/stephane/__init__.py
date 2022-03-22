from faser.generators.base import PSFGenerator
from faser.generators.scalar.phasenet.generator import PsfGenerator3D
from .original import generate_psf


class StephanePSFGenerator(PSFGenerator):
    def generate(self):

        return generate_psf(self.config)
