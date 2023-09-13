import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget
from qtpy import QtWidgets, QtGui
import napari
from superqt import QDoubleRangeSlider, QLabeledDoubleRangeSlider, QLabeledDoubleSlider
import pydantic
from faser.env import get_asset_file
from faser.generators.base import AberrationFloat, PSFConfig
from typing import Callable, Type, Any
import typing
from pydantic.fields import ModelField
from superqt import QEnumComboBox
from enum import Enum
from faser.generators.utils import polar_phase_mask, polar_to_cartesian
from faser.generators.vectorial.stephane.tilted_coverslip import generate_psf, generate_intensity_profile, generate_phase_mask, generate_aberration
from pydantic.types import ConstrainedFloat
import numpy as np
import itertools
import dask.array as da
import dask
import os
from .fields import generate_single_widgets_from_model, build_key_filter
from .mpl_canvas import MatplotlibDialog

class ScrollableWidget(QtWidgets.QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        *args,
        filter_fields=None,
        callback=None,
        range_callback=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.managed_widgets = generate_single_widgets_from_model(
            PSFConfig,
            callback=callback,
            range_callback=range_callback,
            parent=self,
            filter_fields=filter_fields,
        )

        self.mylayout = QtWidgets.QVBoxLayout()

        for widget in self.managed_widgets:
            widget.init_ui()
            self.mylayout.addWidget(widget)

        self.active_base_model = PSFConfig()
        self.setLayout(self.mylayout)

    def model_value_changed(self, name, value):
        self.active_base_model.__setattr__(name, value)
        print(self.active_base_model.dict())


class SampleTab(QtWidgets.QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        main = None,
        *args,
        filter_fields=None,
        callback=None,
        range_callback=None,
        image="placeholder.png",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.main = main


        self.scroll = (
            QtWidgets.QScrollArea()
        )  # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = ScrollableWidget(
            self.viewer,
            callback=callback,
            range_callback=range_callback,
            filter_fields=filter_fields,
        )  # Widget that contains the collection of Vertical Box

        # Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.mylayout = QtWidgets.QVBoxLayout()

        if image is not None:
            self.im = QtGui.QPixmap(get_asset_file(image))
            self.im = self.im.scaledToHeight(400)
            self.label = QtWidgets.QLabel()
            self.label.setPixmap(self.im)
            self.mylayout.addWidget(self.label)

        self.mylayout.addWidget(self.scroll)

        self.setLayout(self.mylayout)




@dask.delayed
def lazy_generate_psf(config: PSFConfig):
    return generate_psf(config)



def generate_wavefront(config: PSFConfig):
    num_radii = 150
    num_angles = 300
    polar_mask = polar_phase_mask(num_radii, num_angles)

    # Convert to Cartesian coordinates
    cartesian_mask = polar_to_cartesian(polar_mask)
    return cartesian_mask





class AbberationTab(SampleTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_button = QtWidgets.QPushButton("Show Wavefront")
        self.show_button.clicked.connect(self.show_wavefront)
        self.wavefront_dialog = MatplotlibDialog("WaveFront", parent=self)

        self.mylayout.addWidget(self.show_button)

    def show_wavefront(self):
        self.wavefront_dialog.update(generate_aberration(self.main.active_base_model), "Wavefront")
        self.wavefront_dialog.show()



class BeamTab(SampleTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_button = QtWidgets.QPushButton("Show Intensity")
        self.show_button.clicked.connect(self.show_intensity)

        self.showp_button = QtWidgets.QPushButton("Show Phase Mask")
        self.showp_button.clicked.connect(self.show_phase_mask)

        self.intensity_dialog = MatplotlibDialog("Intensity", parent=self)
        self.phase_mask_dialog = MatplotlibDialog("Phase Mask", parent=self)

        self.mylayout.addWidget(self.show_button)
        self.mylayout.addWidget(self.showp_button)

    def show_intensity(self):
        self.intensity_dialog.update(generate_intensity_profile(self.main.active_base_model), "Intensity")
        self.intensity_dialog.show()

    def show_phase_mask(self):
        self.phase_mask_dialog.update(generate_phase_mask(self.main.active_base_model), "Phase mask")
        self.phase_mask_dialog.show()

simulation_set = [
    "LfocalXMM",
    "LfocalYMM",
    "LfocalZMM",
    "Ntheta",
    "Nphi",
    "threshold",
    "it",
    "Nx",
    "Ny",
    "Nz",
]


noise_set = [
    "gaussian_beam_noise",
    "detector_gaussian_noise",
    "add_detector_poisson_noise",
]

geometry_set = [
    "n1",
    "n2",
    "n3",
    "collarMM",
    "thickMM",
    "depthMM",
    "tilt_angle_degree",
    "WDMM",
]

beam_set = [
    "mode",
    "polarization",
    "wavelengthMM",
    "waistMM",
    "ampl_offset_x",
    "ampl_offset_y",
    "psi_degree",
    "eps_degree",
    "ring_radius",
    "vc",
    "rc",
    "mask_offset_x",
    "mask_offset_y",
    "p",
]


aberration_set = [
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "a8",
    "a9",
    "a10",
    "a11",
    "aberration",
    "aberration_offset_x",
    "aberration_offset_y",
]


class MainWidget(QtWidgets.QWidget):
    def __init__(self, viewer: napari.Viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.simulation_tab = SampleTab(

            self.viewer,

            main = self,
            image="placeholder.png",
            callback=self.callback,
            range_callback=self.range_callback,
            filter_fields=build_key_filter(simulation_set),
        )
        self.geometry_tab = SampleTab(
            self.viewer,
            main = self,
            callback=self.callback,
            range_callback=self.range_callback,
            filter_fields=build_key_filter(geometry_set),
        )
        self.aberration_tab = AbberationTab(
            self.viewer,

            main = self,
            callback=self.callback,
            range_callback=self.range_callback,
            filter_fields=build_key_filter(aberration_set),
        )
        self.beam_tab = BeamTab(
            self.viewer,

            main = self,
            callback=self.callback,
            range_callback=self.range_callback,
            filter_fields=build_key_filter(beam_set),
        )

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        tabwidget = QtWidgets.QTabWidget()
        tabwidget.addTab(self.simulation_tab, "Simulation")
        tabwidget.addTab(self.geometry_tab, "Geometry")
        tabwidget.addTab(self.aberration_tab, "Abberation")
        tabwidget.addTab(self.beam_tab, "Beam")
        layout.addWidget(tabwidget, 0, 0)

        self.generate = QtWidgets.QPushButton("Generate")

        self.generate.clicked.connect(self.generate_psf)

        self.loadb = QtWidgets.QPushButton("Load")
        self.loadb.clicked.connect(self.load_model)

        self.saveb = QtWidgets.QPushButton("Save")
        self.saveb.clicked.connect(self.save_model)
        layout.addWidget(self.generate, 2, 0)
        layout.addWidget(self.saveb, 3, 0)
        layout.addWidget(self.loadb, 4, 0)
        self.active_base_model = PSFConfig()

        self.active_batchers = {}

    def callback(self, name, value):
        split = name.split(".")
        if len(split) > 1:
            self.active_base_model.__getattribute__(split[0]).__setattr__(
                split[1], value
            )
        else:
            self.active_base_model.__setattr__(name, value)

    def range_callback(self, name, value):
        if value == None:
            if name in self.active_batchers:
                del self.active_batchers[name]

        else:
            self.active_batchers[name] = value

        print(self.active_batchers)

        self.update_ui()

    def calculate_batcher_length(self):
        length = 1
        for key, value in self.active_batchers.items():
            length *= len(value.to_list())

        return length

    def update_ui(self):
        if len(self.active_batchers) > 0:
            self.generate.setText(f"Generate {self.calculate_batcher_length()} PSFs")

        else:
            self.generate.setText("Generate")
            print(self.active_batchers)

    def reinit_ui(self):
        for key, value in self.active_batchers.items():
            self.update_default(key, value)

    def save_model(self):
        default = os.path.join(os.getcwd(), "psf_config.json")
        name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", default, "Config Files (*.json)"
        )
        if name:
            file = open(name, "w")
            text = self.active_base_model.json()
            file.write(text)
            file.close()

    def load_model(self):
        name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", "", "Config Files (*.json)"
        )
        if name:
            file = open(name[0], "r")
            text = file.read()
            self.active_base_model = PSFConfig.parse_raw(text)
            self.reinit_ui()

    def start_map(self):
        mapper = self.active_batchers

        permutations = list(
            itertools.product(*map(lambda x: x.to_list(), mapper.values()))
        )

        models = []

        for i in permutations:
            values = {k: v for k, v in zip(mapper.keys(), i)}
            new_model = self.active_base_model.copy(update=values)
            models.append(new_model)
            print(new_model)

        t = [
            da.from_delayed(
                lazy_generate_psf(new_model),
                (new_model.Nz, new_model.Nx, new_model.Ny),
                dtype=np.float64,
            )
            for new_model in models
        ]
        t = da.stack(t, axis=0)

        reshape_start = [
            len(value.to_list()) for key, value in self.active_batchers.items()
        ]

        if len(reshape_start) < 3:
            t = t.reshape(
                reshape_start
                + [
                    self.active_base_model.Nz,
                    self.active_base_model.Nx,
                    self.active_base_model.Ny,
                ],
                merge_chunks=False,
            )

        return self.viewer.add_image(
            t,
            name="Batch PSF",
            metadata={"is_psf": True, "configs": models, "is_batch": True},
            multiscale=False,
        )

    def generate_psf(self):

        if len(self.active_batchers) > 0:
            self.start_map()

        else:

            config = self.active_base_model
            psf = generate_psf(config)
            print(psf.max())
            self.viewer.add_image(
                psf,
                name=f"PSF ",
                metadata={"is_psf": True, "config": config, "is_batch": False},
            )
