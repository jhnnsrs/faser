from faser.napari.gui import generate_psf_gui, convolve_image_gui, make_effective_gui, generate_space
from skimage import data
import napari
import numpy as np
import argparse
from scipy.sparse import random


def main(**kwargs):
    viewer = napari.Viewer()
    # viewer.window.add_dock_widget(parameters, area="right", name="Sampling parameters")
    #░viewer.window.add_dock_widget(input_beam_gui, area="right", name="Input Beam")
    # viewer.window.add_dock_widget(focusing_geometry_gui, area="right", name="Focusing Geometry")
    viewer.window.add_dock_widget(generate_psf_gui, area="right", name="Faser")
    viewer.window.add_dock_widget(convolve_image_gui, area="right", name="Convov")
    viewer.window.add_dock_widget(make_effective_gui, area="right", name="Effective STED PSF")
    viewer.window.add_dock_widget(generate_space, area="right", name="Generate Space")

    
    napari.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()
