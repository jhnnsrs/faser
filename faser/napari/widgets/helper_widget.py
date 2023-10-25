import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget
from qtpy import QtWidgets, QtGui
import napari
from scipy import ndimage
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
from slugify import slugify
from faser.napari.widgets.fields import generate_single_widgets_from_model


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
        export_file_dir = os.path.join(export_dir, slugify(layer_name))
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



class SpaceModel(pydantic.BaseModel):
    x_size: int = 1000
    y_size: int = 1000
    z_size: int = 10
    dots: int = 50
    

class SampleTab(HelperTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show = QtWidgets.QPushButton("Create space")
        self.show.clicked.connect(self.generate_space)

        self.managed_widgets = generate_single_widgets_from_model(
            SpaceModel,
            callback=self.callback,
            range_callback=None,
            parent=self,
        )

        print(self.managed_widgets)


        for widget in self.managed_widgets:
            widget.init_ui()
            self.mylayout.addWidget(widget)


        self.mylayout.addWidget(self.show)
        self.space_model = SpaceModel()

    def callback(self, name, value):
        split = name.split(".")
        if len(split) > 1:
            self.space_model.__getattribute__(split[0]).__setattr__(
                split[1], value
            )
        else:
            self.space_model.__setattr__(name, value)


    def show_wavefront(self):
        raise NotImplementedError()
    

    def generate_space(self):
        x = np.random.randint(0, self.space_model.x_size, size=(self.space_model.dots))
        y = np.random.randint(0, self.space_model.y_size, size=(self.space_model.dots))
        z = np.random.randint(0, self.space_model.z_size, size=(self.space_model.dots))

        M = np.zeros((self.space_model.z_size, self.space_model.y_size,  self.space_model.y_size,))
        for p in zip(z, x, y):
            M[p] = 1

        self.viewer.add_image(M, name="Space")
        

class EffectiveModel(pydantic.BaseModel):
    Isat: float = pydantic.Field(default=0.1, lt=1, gt=0)
    

class EffectiveTab(HelperTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.managed_widgets = generate_single_widgets_from_model(
            EffectiveModel,
            callback=self.callback,
            range_callback=None,
            parent=self,
        )

        print(self.managed_widgets)


        for widget in self.managed_widgets:
            widget.init_ui()
            self.mylayout.addWidget(widget)

        self.show = QtWidgets.QPushButton("Select exactly 2 PSFs")
        self.show.setEnabled(False)
        self.show.clicked.connect(self.make_effective_psf)

        self.mylayout.addWidget(self.show)


        self.effective_model = EffectiveModel()

        self.viewer.layers.selection.events.connect(self.update_selection)


    def callback(self, name, value):
        split = name.split(".")
        if len(split) > 1:
            self.effective_model.__getattribute__(split[0]).__setattr__(
                split[1], value
            )
        else:
            self.effective_model.__setattr__(name, value)

    def make_effective_psf(self):
        I_sat = self.effective_model.Isat
        gaussian_layers = (
            layer for layer in self.viewer.layers.selection if layer.metadata.get("is_psf", True)
        )

        psf_layer_one = next(gaussian_layers)   # Excitation PSF
        psf_layer_two = next(gaussian_layers)   # Depletion PSF
        new_psf = np.multiply(psf_layer_one.data, np.exp(-psf_layer_two.data / I_sat))

        return self.viewer.add_image(
            new_psf,
            name=f"Combined PSF {psf_layer_one.name} {psf_layer_two.name}",
            metadata={"is_psf": True},
        )


    def update_selection(self, event):
        selection = self.viewer.layers.selection

        if not selection:
            self.show.setEnabled(False)
            self.show.setText("Select exactly 2 PSFs")

        else:
            layers = [layer for layer in selection if layer.metadata.get("is_psf", False)]
            if len(layers) != 2:
                self.show.setEnabled(False)
                self.show.setText("Select exactly 2 PSFs")
            
            else:
                self.show.setEnabled(True)
                self.show.setText("Create Effective PSF")


class ConvolveModel(pydantic.BaseModel):
    pass
    

class ConvolveTab(HelperTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.managed_widgets = generate_single_widgets_from_model(
            ConvolveModel,
            callback=self.callback,
            range_callback=None,
            parent=self,
        )

        print(self.managed_widgets)


        for widget in self.managed_widgets:
            widget.init_ui()
            self.mylayout.addWidget(widget)

        self.show = QtWidgets.QPushButton("Select image and PSF")
        self.show.setEnabled(False)
        self.show.clicked.connect(self.make_effective_psf)

        self.mylayout.addWidget(self.show)


        self.effective_model = EffectiveModel()

        self.viewer.layers.selection.events.connect(self.update_selection)


    def callback(self, name, value):
        split = name.split(".")
        if len(split) > 1:
            self.effective_model.__getattribute__(split[0]).__setattr__(
                split[1], value
            )
        else:
            self.effective_model.__setattr__(name, value)

    def make_effective_psf(self):
        print("Making effective PSF")
        psf_layer = next(
            layer
            for layer in self.viewer.layers.selection
            if layer.metadata.get("is_psf", False)
        )
        image_layer = next(
            layer
            for layer in self.viewer.layers.selection
            if not layer.metadata.get("is_psf", False)
        )

        image_data = image_layer.data
        psf_data = psf_layer.data

        if image_data.ndim == 2:
            psf_data = psf_data[psf_data.shape[0] // 2, :, :]

            con = ndimage.convolve(
                image_data, psf_data, mode="constant", cval=0.0, origin=0
            )


        con = ndimage.convolve(image_data, psf_data, mode="constant", cval=0.0, origin=0)
    
        return self.viewer.add_image(
            con.squeeze(),
            name=f"Convoled {image_layer.name} with {psf_layer.name}",
        )

    def update_selection(self, event):
        selection = self.viewer.layers.selection

        if not selection:
            self.show.setEnabled(False)
            self.show.setText("Select a PSF and the Image")

        else:
            layers = [layer for layer in selection if layer.metadata.get("is_psf", False)]
            if len(layers) != 1:
                self.show.setEnabled(False)
                self.show.setText("Select only one PSF ")
            
            else:
                self.show.setEnabled(True)
                self.show.setText("Convolve Image")



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
        self.effective_tab = EffectiveTab(
            self.viewer,
        )
        self.convolve_tab = ConvolveTab(
            self.viewer,
        )
        
        layout = QtWidgets.QGridLayout()
        tabwidget = QtWidgets.QTabWidget()
        tabwidget.addTab(self.export_tab, "Export")
        tabwidget.addTab(self.sample_tab, "Sample")
        tabwidget.addTab(self.inspect_tab, "Inspect")
        tabwidget.addTab(self.effective_tab, "Effective")
        tabwidget.addTab(self.convolve_tab, "Convolve")
        layout.addWidget(tabwidget, 0, 0)


        self.setLayout(layout)
