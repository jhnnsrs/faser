import contextlib
from faser.napari.gui import generate_psf_gui
from faser.napari.main import main

import napari
import numpy as np
import argparse
import os
if __name__ == "__main__":

    #stephqnepnie
    os.environ["NAPARI_ASYNC"] = "1"
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()
