import json

import rich_click as click
import tifffile

from faser.generators.base import PSFConfig
from faser.generators.vectorial.stephane.tilted_coverslip import generate_psf
from faser.napari.main import main as qtmain

arg_map = {}

logo = r"""
  __                     
 / _| __ _ ___  ___ _ __ 
| |_ / _` / __|/ _ \ '__|
|  _| (_| \__ \  __/ |   
|_|  \__,_|___/\___|_|   
"""


def make_nice_name(name):
    return name.replace("_", "_").lower()


def psf_attributes(func):

    for key, field in PSFConfig.model_fields.items():

        nice_key = make_nice_name(key)

        arg_map[nice_key] = key

        func = click.option(
            f"--{nice_key}",
            type=field.annotation,
            help=field.description,
            default=field.default,
        )(func)
    return func


@click.command()
@click.option(
    "--config", type=click.File(mode="r"), help="Path to a JSON file", default=None
)
@psf_attributes
def main(config=None, **kwargs):

    qtmain()
