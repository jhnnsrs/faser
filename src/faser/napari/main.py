from faser.napari.gui import (
    generate_psf_gui,
    convolve_image_gui,
    make_effective_gui,
    generate_space,
)

# from skimage import data
import napari

# import numpy as np
import argparse

# from scipy.sparse import random
from faser.napari.widgets.main_widget import MainWidget
from faser.napari.widgets.helper_widget import HelperWidget

# from faser.napari.widgets.tob_bar import TopBar


def main(**kwargs):
    viewer = napari.Viewer()

    main = MainWidget(viewer)
    helper = HelperWidget(viewer)
    viewer.window.add_dock_widget(main, area="right", name="PSF")
    viewer.window.add_dock_widget(helper, area="right", name="Helper")

    napari.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()
