# import typing
import itertools
import os
import time

# from typing import Callable, Type, Any
# import typing
# from pydantic.fields import ModelField
# from superqt import QEnumComboBox
from enum import Enum

import dask
import dask.array as da
import napari

# from pydantic.types import ConstrainedFloat
import numpy as np

# from PyQt5.QtWidgets import QWidget
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QStyle

# from superqt import QDoubleRangeSlider, QLabeledDoubleRangeSlider, QLabeledDoubleSlider
# import pydantic
from faser.env import get_asset_file
from faser.generators.base import AberrationFloat, PSFConfig
from faser.generators.utils import polar_phase_mask, polar_to_cartesian
from faser.generators.vectorial.stephane.tilted_coverslip import (
    generate_aberration,
    generate_intensity_profile,
    generate_phase_mask,
    generate_psf,
)
from faser.napari.widgets.helper_widget import comparative_value

from .fields import build_key_filter, generate_single_widgets_from_model
from .mpl_canvas import (
    BeamDialog,
    MatplotlibDialog,
    MaximumDialog,
    PhaseMaskDialog,
    WavefrontDialog,
)


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
        self.mylayout.setContentsMargins(0, 0, 0, 0)
        self.mylayout.setSpacing(1)

        for widget in self.managed_widgets:
            widget.init_ui()
            self.mylayout.addWidget(widget)

        self.active_base_model = PSFConfig()
        self.setLayout(self.mylayout)

    def model_value_changed(self, name, value):
        self.active_base_model.__setattr__(name, value)
        print(self.active_base_model.model_dump_json())

    def reset(self):
        for widget in self.managed_widgets:
            widget.reset_default()


class SampleTab(QtWidgets.QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        main=None,
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
        self.mylayout.setContentsMargins(10, 10, 10, 10)
        self.mylayout.setSpacing(1)

        if image is not None:
            self.im = QtGui.QPixmap(get_asset_file(image))
            self.im = self.im.scaledToHeight(300)
            self.label = QtWidgets.QLabel()
            self.label.setPixmap(self.im)
            self.mylayout.addWidget(self.label)

        self.mylayout.addWidget(self.scroll)

        self.setLayout(self.mylayout)

    def reset(self):
        self.widget.reset()


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
        self.wavefront_dialog = WavefrontDialog("Wavefront", parent=self)

        self.mylayout.addWidget(self.show_button)

    def show_wavefront(self):
        self.wavefront_dialog.update(
            generate_aberration(self.main.active_base_model), "Wavefront"
        )
        self.wavefront_dialog.show()


class BeamTab(SampleTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_button = QtWidgets.QPushButton("Show Intensity")
        self.show_button.clicked.connect(self.show_intensity)

        self.showp_button = QtWidgets.QPushButton("Show Phase Mask")
        self.showp_button.clicked.connect(self.show_phase_mask)

        self.intensity_dialog = BeamDialog("Intensity", parent=self)
        self.phase_mask_dialog = PhaseMaskDialog("Phase Mask", parent=self)

        self.mylayout.addWidget(self.show_button)
        self.mylayout.addWidget(self.showp_button)

    def show_intensity(self):
        self.intensity_dialog.update(
            generate_intensity_profile(self.main.active_base_model), "Intensity"
        )
        self.intensity_dialog.show()

    def show_phase_mask(self):
        self.phase_mask_dialog.update(
            generate_phase_mask(self.main.active_base_model), "Phase mask"
        )
        self.phase_mask_dialog.show()


simulation_set = [
    "L_obs_XY",
    "L_obs_Z",
    "Ntheta",
    "Nphi",
    "Nxy",
    "Nz",
    "Normalize",  # ,
    # "add_detector_poisson_noise"
]

geometry_set = [
    "NA",
    "WD",
    "n1",
    "n2",
    "n3",
    "Thickness",
    "Collar",
    "Depth",
    "Tilt",
    "Window",
    "Wind_Radius",
    "Wind_Depth",
]

beam_set = [
    "Mode",
    "Polarization",
    "Wavelength",
    "Waist",
    "Ampl_offset_x",
    "Ampl_offset_y",
    "Psi",
    "Epsilon",
    "Ring_Radius",
    "VC",
    "RC",
    "Mask_offset_x",
    "Mask_offset_y",
    "p",
]

aberration_set = [
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "a8",
    "a9",
    "a12",
    "a24",
    "Aberration_offset_x",
    "Aberration_offset_y",
]

default_config = PSFConfig()


def create_label(config: PSFConfig):
    label = ""
    for field in config.__fields__:
        a = default_config.__getattribute__(field)
        b = config.__getattribute__(field)

        if a != b:
            label += f" {field}: {comparative_value(a, b)}"

    if label == "":
        return "Default"

    return label


class MainWidget(QtWidgets.QWidget):
    def __init__(self, viewer: napari.Viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.simulation_tab = SampleTab(
            self.viewer,
            main=self,
            image="Figure_simu.png",
            callback=self.callback,
            range_callback=self.range_callback,
            filter_fields=build_key_filter(simulation_set),
        )
        self.geometry_tab = SampleTab(
            self.viewer,
            main=self,
            image="Figure_geom.png",
            callback=self.callback,
            range_callback=self.range_callback,
            filter_fields=build_key_filter(geometry_set),
        )
        self.beam_tab = BeamTab(
            self.viewer,
            main=self,
            image="Figure_beam.png",
            callback=self.callback,
            range_callback=self.range_callback,
            filter_fields=build_key_filter(beam_set),
        )
        self.aberration_tab = AbberationTab(
            self.viewer,
            main=self,
            image="Figure_ab.png",
            callback=self.callback,
            range_callback=self.range_callback,
            filter_fields=build_key_filter(aberration_set),
        )

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(1)
        self.setLayout(layout)
        tabwidget = QtWidgets.QTabWidget()
        tabwidget.addTab(self.simulation_tab, "Simulation")
        tabwidget.addTab(self.geometry_tab, "Geometry")
        tabwidget.addTab(self.beam_tab, "Beam")
        tabwidget.addTab(self.aberration_tab, "Abberation")

        self.generate = QtWidgets.QPushButton("Generate")

        self.generate.clicked.connect(self.generate_psf)
        self.generate.setMinimumHeight(20)
        self.generate.setMaximumHeight(20)

        load_pixmal = QtGui.QPixmap(get_asset_file("save.png")).scaled(
            15, 15, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        load_icon = QtGui.QIcon(load_pixmal)
        self.loadb = QtWidgets.QPushButton()
        self.loadb.setIcon(load_icon)
        self.loadb.clicked.connect(self.load_model)
        self.loadb.setMinimumWidth(20)
        self.loadb.setMaximumWidth(20)
        self.loadb.setMinimumHeight(20)
        self.loadb.setMaximumHeight(20)

        save_icon = QtGui.QIcon(
            QtGui.QPixmap(get_asset_file("load.png")).scaled(
                15, 15, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        )
        self.saveb = QtWidgets.QPushButton()
        self.saveb.setIcon(save_icon)
        self.saveb.clicked.connect(self.save_model)
        self.saveb.setMinimumWidth(20)
        self.saveb.setMaximumWidth(20)
        self.saveb.setMinimumHeight(20)
        self.saveb.setMaximumHeight(20)

        refresh_icon = QtGui.QIcon(
            QtGui.QPixmap(get_asset_file("refresh.png")).scaled(
                15, 15, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        )
        self.resetb = QtWidgets.QPushButton()
        self.resetb.setIcon(refresh_icon)
        self.resetb.clicked.connect(self.reset_values)
        self.resetb.setMinimumWidth(20)
        self.resetb.setMaximumWidth(20)
        self.resetb.setMinimumHeight(20)
        self.resetb.setMaximumHeight(20)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.setContentsMargins(10, 10, 10, 10)
        hlayout.addWidget(self.generate)
        hlayout.addWidget(self.loadb)
        hlayout.addWidget(self.saveb)
        hlayout.addWidget(self.resetb)

        layout.addWidget(tabwidget)

        layout.addLayout(hlayout)
        self.active_base_model = PSFConfig()

        self.active_batchers = {}

    def reset_values(self):
        self.simulation_tab.reset()
        self.geometry_tab.reset()
        self.beam_tab.reset()
        self.aberration_tab.reset()
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
        print("Range Callback called with", name)
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
            file = open(name, "r")
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
                (new_model.Nz, new_model.Nxy, new_model.Nxy),
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
                    self.active_base_model.Nxy,
                    self.active_base_model.Nxy,
                ],
                merge_chunks=False,
            )

        return self.viewer.add_image(
            t,
            name="Batch PSF",
            metadata={"is_psf": True, "configs": models, "is_batch": True},
            multiscale=False,
            colormap="viridis",
        )

    def generate_psf(self):

        start = time.time()

        self.generate.setText("Generating...")

        if len(self.active_batchers) > 0:
            self.start_map()

        else:

            config = self.active_base_model
            psf = generate_psf(config)
            print(psf.max())
            self.viewer.add_image(
                psf,
                name=create_label(config),
                metadata={"is_psf": True, "config": config, "is_batch": False},
                colormap="viridis",
            )
            self.generate.setText("Generate")

        end = time.time()
        print(end - start)
