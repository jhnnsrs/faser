from faser.generators.base import PSFGenerator
from faser.generators.scalar.gibson_lanny.gen import generate_psf


class GibsonLannyPSFGenerator(PSFGenerator):
    def generate(self):
        return generate_psf(self.config)
