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
from faser.generators.vectorial.stephane.tilted_coverslip import generate_psf
from pydantic.types import ConstrainedFloat
import numpy as np
import itertools
import dask.array as da
import dask
import os
from superqt.utils import thread_worker
import tifffile


class HelperTab(QtWidgets.QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.mylayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mylayout)



# Step 1: Create a worker class
class ExportWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)


    def __init__(self, layers, export_dir):
        super().__init__()
        self.layers = layers
        self.export_dir = export_dir

    def export_layer_with_config_data_to_file(self, data, export_dir, layer_name, config):
        export_file_dir = os.path.join(export_dir, layer_name)
        os.makedirs(export_file_dir, exist_ok=True)
        with open(os.path.join(export_file_dir, "config.txt"), "w") as f:
            f.write(config.json())

        tifffile.imsave(os.path.join(export_file_dir, "psf.tif"), data)
        print("Exported")


    def run(self):
        """Long-running task."""
        print("Running")
        for layer in self.layers:
            if layer.metadata.get("is_psf", False) is True:
                if layer.metadata.get("is_batch", False) is True:
                    first_dim = layer.data.shape[0]
                    for i in range(first_dim):

                        self.progress.emit(i + 1)
                        self.export_layer_with_config_data_to_file(layer.data[i, :, :, :], self.export_dir, layer.name, layer.metadata["configs"][i])
                else:
                    print("Exporting this one")
                    self.export_layer_with_config_data_to_file(layer.data, self.export_dir, layer.name, layer.metadata["config"])

        self.finished.emit()

@thread_worker
def export_layer_with_config_data_to_file(data, export_dir, layer_name, config):
    export_file_dir = os.path.join(export_dir, layer_name)
    os.makedirs(export_file_dir, exist_ok=True)
    with open(os.path.join(export_file_dir, "config.txt"), "w") as f:
        f.write(config.json())

    tifffile.imsave(os.path.join(export_file_dir, "psf.tif"), data)
    print("Exported")



class ExportTab(HelperTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show = QtWidgets.QPushButton("Select PSF Layers")
        self.show.setEnabled(False)
        self.show.clicked.connect(self.export_pressed)

        self.viewer.layers.selection.events.connect(self.update_selection)

        self.mylayout.addWidget(self.show)

    def on_worker_done(self):
        print("done")
        self.show.setEnabled(True)

    def on_worker_progress(self, value):
        print(value)


    def update_selection(self, event):
        selection = self.viewer.layers.selection

        if not selection:
            self.show.setEnabled(False)
            self.show.setText("Select PSF")

        else:
            layers = [layer for layer in selection if layer.metadata.get("is_psf", False)]
            if len(layers) == 0:
                self.show.setEnabled(False)
                self.show.setText("Select PSF Layers")

            else :
                self.show.setEnabled(True)
                self.show.setText("Export PSF" if len(layers) == 1 else "Export PSFs")



        print(self.viewer.layers.selection.active)

    def export_layer_with_config_data_to_file(data, export_dir, layer_name, config):
        export_file_dir = os.path.join(export_dir, layer_name)
        os.makedirs(export_file_dir, exist_ok=True)
        with open(os.path.join(export_file_dir, "config.txt"), "w") as f:
            f.write(config.json())

        tifffile.imsave(os.path.join(export_file_dir, "psf.tif"), data)
        print("Exported")

    def export_active_selection(self, export_dir):
        layers = []
        for layer in self.viewer.layers.selection:
            if layer.metadata.get("is_psf", False) == True:
                layers.append(layer)


        self.thread = QtCore.QThread()
        self.worker = ExportWorker(layers, export_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.on_worker_progress)
        self.thread.start()
                
                

    def export_pressed(self):
        export_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory"
        )

        if export_dir:
            self.export_active_selection(export_dir=export_dir)


class SampleTab(HelperTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show = QtWidgets.QPushButton("Show Intensity")
        self.show.clicked.connect(self.show_wavefront)

        self.showp = QtWidgets.QPushButton("Show Phasemask")
        self.showp.clicked.connect(self.show_wavefront)

        self.mylayout.addWidget(self.show)
        self.mylayout.addWidget(self.showp)

    def show_wavefront(self):
        raise NotImplementedError()


class InspectTab(HelperTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show = QtWidgets.QPushButton("Show Intensity")
        self.show.clicked.connect(self.show_wavefront)

        self.showp = QtWidgets.QPushButton("Show Phasemask")
        self.showp.clicked.connect(self.show_wavefront)

        self.viewer.layers.selection.events.connect(self.update_selection)

        self.mylayout.addWidget(self.show)
        self.mylayout.addWidget(self.showp)

    def update_selection(self, event):
        print(self.viewer.layers.selection.active)

    def show_wavefront(self):
        raise NotImplementedError()


class HelperWidget(QtWidgets.QWidget):
    def __init__(self, viewer: napari.Viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.export_tab = ExportTab(
            self.viewer,
        )
        self.sample_tab = SampleTab(
            self.viewer,
        )
        self.inspect_tab = InspectTab(
            self.viewer,
        )
        
        layout = QtWidgets.QGridLayout()
        tabwidget = QtWidgets.QTabWidget()
        tabwidget.addTab(self.export_tab, "Export")
        tabwidget.addTab(self.sample_tab, "Sample")
        tabwidget.addTab(self.inspect_tab, "Inspect")
        layout.addWidget(tabwidget, 0, 0)


        self.setLayout(layout)
