# import contextlib
# from faser.napari.gui import generate_psf_gui
# import napari
# import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse

from faser.napari.main import main

if __name__ == "__main__":

    os.environ["NAPARI_ASYNC"] = "1"
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()
