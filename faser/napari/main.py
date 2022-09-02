from faser.napari.gui import generate_psf_gui, convolve_image_gui, make_effective_gui
from skimage import data
import napari
import numpy as np
import argparse
from scipy.sparse import random


def main(**kwargs):
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(generate_psf_gui, area="right", name="Faser")
    viewer.window.add_dock_widget(convolve_image_gui, area="right", name="Convov")
    viewer.window.add_dock_widget(make_effective_gui, area="right", name="Convov")

    XY = 100
    Z = 20

    x = np.random.randint(0, XY, size=(50))
    y = np.random.randint(0, XY, size=(50))
    z = (10,) * 100

    M = np.zeros((Z, XY, XY))
    for p in zip(z, x, y):
        M[p] = 1

    viewer.add_image(M, name="Space")
    napari.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()
