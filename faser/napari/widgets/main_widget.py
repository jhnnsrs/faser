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


class FormField(QtWidgets.QWidget):
    on_child_value_changed = QtCore.pyqtSignal(str, object)
    on_child_range_value_changed = QtCore.pyqtSignal(str,object)

    def __init__(self, key: str, field: ModelField, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.field = field
        self.key = key
        self.label = key
        self.child = None
        self.mode = "single"
        self.description = field.field_info.description
        self.steps = 3


    def replace_widget(self, oldwidget, newwidget):
        self.xlayout.removeWidget(oldwidget)
        oldwidget.setParent(None)
        self.xlayout.addWidget(newwidget)



    def emit_child_value_changed(self, value):
        self.on_child_value_changed.emit(self.key, value)

    def on_child_range_changed(self, value):
        self.on_child_range_value_changed.emit(self.key, value)

    def on_change_mode(self):

        if self.mode == "single":
            
            self.replace_widget(self.child, self.range_child)
            

        elif self.mode == "range":
            self.on_child_range_changed(None)
            self.replace_widget(self.range_child, self.child)

        
        self.mode = "range" if self.mode == "single" else "single"



    def init_ui(self):
        assert self.child is not None, "Child widget must be set before init_ui()"
        assert self.range_child is not None, "Child widget must be set before init_ui()"
        self.xlayout = QtWidgets.QVBoxLayout()


        self.label = QtWidgets.QLabel(self.label)
        self.label.setToolTip(self.description or "No description yet")
        self.button = QtWidgets.QPushButton("Change Mode")
        self.button.clicked.connect(self.on_change_mode)

        layout = QtWidgets.QHBoxLayout()
        self.labelWidget = QtWidgets.QWidget()

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.button)

        self.labelWidget.setLayout(layout)
        self.xlayout.addWidget(self.labelWidget)
        self.xlayout.addWidget(self.child)

        self.setLayout(self.xlayout)




class FloatRange(pydantic.BaseModel):
    min: float
    max: float
    steps: int

    def to_list(self):
        return np.linspace(self.min, self.max, self.steps, dtype=np.float64).tolist()


class FloatSliderField(FormField):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QLabeledDoubleSlider(QtCore.Qt.Horizontal)
        self.child.setMinimum(self.field.field_info.gt or 0.)
        self.child.setValue(self.field.default)
        self.child.setMaximum(self.field.field_info.lt or 1.)
        self.child.valueChanged.connect(self.emit_child_value_changed)


        self.range_child = QLabeledDoubleRangeSlider(QtCore.Qt.Horizontal)
        self.range_child.setRange(self.field.field_info.gt or 0., self.field.field_info.lt or 1.)
        self.range_child.valueChanged.connect(self.on_range_valued_callback)

    def on_range_valued_callback(self, value):
        self.on_child_range_changed(FloatRange(min=float(value[0]), max=float(value[1]), steps=self.steps))




class IntRange(pydantic.BaseModel):
    min: int
    max: int
    steps: int

    def to_list(self):
        return np.linspace(self.min, self.max, self.steps, dtype=np.int64).tolist()


class IntInputField(FormField):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QtWidgets.QLineEdit()
        self.child.setText(str(self.field.default))
        self.child.setValidator(QtGui.QIntValidator())
        self.child.textChanged.connect(self.emit_text_changed)

        self.range_child = QLabeledDoubleRangeSlider(QtCore.Qt.Horizontal)
        self.range_child.setRange(self.field.field_info.gt or 0., self.field.field_info.lt or 1.)
        self.range_child.valueChanged.connect(self.on_range_valued_callback)

    def on_range_valued_callback(self, value):
        self.on_child_range_changed(IntRange(min=int(value[0]), max=int(value[1]), steps=self.steps))

    def emit_text_changed(self, value):
        self.emit_child_value_changed(int(self.child.text()))


class FloatInputField(FormField):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QtWidgets.QLineEdit()
        self.child.setText(str(self.field.default))
        self.child.textChanged.connect(self.emit_text_changed)

        self.range_child = QLabeledDoubleRangeSlider(QtCore.Qt.Horizontal)
        self.range_child.setRange(self.field.field_info.gt or 0., self.field.field_info.lt or 1.)
        self.range_child.valueChanged.connect(self.on_range_valued_callback)

    def on_range_valued_callback(self, value):
        self.on_child_range_changed(FloatRange(min=float(value[0]), max=float(value[1]), steps=self.steps))


    def emit_text_changed(self, value):
        self.emit_child_value_changed(float(self.child.text()))

class EnumField(FormField):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QEnumComboBox()
        self.child.setEnumClass(self.field.type_)
        self.child.currentEnumChanged.connect(self.emit_child_value_changed)
        self.child.show()


        self.range_child = QLabeledDoubleRangeSlider(QtCore.Qt.Horizontal)
        self.range_child.valueChanged.connect(self.on_child_range_changed)



class BaseModelField(FormField):
    
    def __init__(self, *args, **kwargs,) -> None:
        super().__init__(*args, **kwargs)
        self.managed_widgets = generate_single_widgets_from_model(self.field.type_, self.slider_value_changed, self.range_slider_value_changed, parent=self)



    def slider_value_changed(self, name, value):
        self.on_child_value_changed.emit(f"{self.key}.{name}", value)

    def range_slider_value_changed(self, name, value):
        self.on_child_range_value_changed.emit(f"{self.key}.{name}", value)

    def init_ui(self):
        self.label = QtWidgets.QLabel(self.label)
        self.description = QtWidgets.QLabel(self.description)

        self.xlayout = QtWidgets.QVBoxLayout()
        for widget in self.managed_widgets:
            widget.init_ui()
            self.xlayout.addWidget(widget)

        self.setLayout(self.xlayout)





def generate_single_widgets_from_model(model: Type[PSFConfig], callback: Callable[[str, Any], None], range_callback: Callable[[str, Any], None] = None, parent=None, filter_fields=None) -> typing.List[QWidget]:

    widgets = []

    for key, value in model.__fields__.items():
        
        if filter_fields:
            if not filter_fields(key, value):
                continue

        widget = None

        field_type = value.type_
        print(field_type)

        if issubclass(field_type, ConstrainedFloat):
            widget = FloatSliderField(key, value, parent=parent)

        elif field_type == float:
            widget = FloatInputField(key, value, parent=parent)

        elif field_type == int:
            widget = IntInputField(key, value, parent=parent)

        elif issubclass(field_type, pydantic.BaseModel):
            print("BaseModel")
            widget = BaseModelField(key,value, parent=parent)

        elif issubclass(field_type, Enum):
            widget = EnumField(key,value, parent=parent)

        else:
            print(field_type)



        if widget is not None:

            widget.on_child_value_changed.connect(callback)
            widget.on_child_range_value_changed.connect(range_callback)
            widgets.append(widget)


    return widgets




class ScrollableWidget(QtWidgets.QWidget):

    def __init__(self, viewer: napari.Viewer, *args,  filter_fields=None, callback=None, range_callback=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.managed_widgets = generate_single_widgets_from_model(PSFConfig, callback=callback, range_callback=range_callback, parent=self, filter_fields=filter_fields)

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

    def __init__(self, viewer: napari.Viewer, *args, filter_fields=None, callback=None, range_callback=None, image="placeholder.png", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.viewer = viewer

        self.print = QtWidgets.QPushButton("Print")
        self.print.clicked.connect(self.print_model)


        self.scroll = QtWidgets.QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = ScrollableWidget(self.viewer, callback=callback, range_callback=range_callback, filter_fields=filter_fields)                 # Widget that contains the collection of Vertical Box
       


        #Scroll Area Properties
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



        self.active_base_model = PSFConfig()

    def model_value_changed(self, name, value):
        self.active_base_model.__setattr__(name, value)
        print(self.active_base_model.dict())


    def print_model(self):
        print(self.active_base_model.dict())



def build_key_filter(field_set: typing.Set[str]) -> typing.Callable[[str, ModelField], bool]:
    def key_filter(key, value):
        return key in field_set
    return key_filter


@dask.delayed
def lazy_generate_psf(config: PSFConfig):
    return generate_psf(config)


class AbberationTab(SampleTab):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.show = QtWidgets.QPushButton("Show Wavefront")
        self.show.clicked.connect(self.show_wavefront)


        self.mylayout.addWidget(self.show)

    def show_wavefront(self):
        raise NotImplementedError()
    

class BeamTab(SampleTab):

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
        self.simulation_tab = SampleTab(self.viewer, image="placeholder.png", callback=self.callback, range_callback=self.range_callback, filter_fields=build_key_filter(simulation_set))
        self.geometry_tab = SampleTab(self.viewer, callback=self.callback, range_callback=self.range_callback,filter_fields=build_key_filter(geometry_set))
        self.aberration_tab = AbberationTab(self.viewer,callback=self.callback,  range_callback=self.range_callback,filter_fields=build_key_filter(aberration_set))
        self.beam_tab = BeamTab(self.viewer, callback=self.callback, range_callback=self.range_callback, filter_fields=build_key_filter(beam_set))

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
        layout.addWidget(self.generate, 2, 0)
        self.active_base_model = PSFConfig()


        self.active_batchers = {}

    def callback(self, name, value):
        split = name.split(".")
        if len(split) > 1:
            self.active_base_model.__getattribute__(split[0]).__setattr__(split[1], value)
        else:
            self.active_base_model.__setattr__(name, value)


    def range_callback(self, name, value):
        if value == None:
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


    def start_map(self):
        mapper = self.active_batchers 


        permutations = list(itertools.product(*map(lambda x: x.to_list(), mapper.values())))

        models = []

        for i in permutations:
            values = {k: v for k, v in zip(mapper.keys(), i)}
            new_model = self.active_base_model.copy(update=values)
            models.append(new_model)
            print(new_model)


        t = [da.from_delayed(lazy_generate_psf(new_model), (new_model.Nz, new_model.Nx, new_model.Ny), dtype=np.float64) for new_model in models]
        t = da.stack(t, axis=0)

        reshape_start = [len(value.to_list()) for key, value in self.active_batchers.items()]

        t = t.reshape(reshape_start + [self.active_base_model.Nz, self.active_base_model.Nx, self.active_base_model.Ny])

        print(t.shape)


        return self.viewer.add_image(t, name="Batch PSF", metadata={"is_psf": True, "config": self.active_base_model})




    def generate_psf(self):

        if len(self.active_batchers) > 0:
            self.start_map()

        else:

            config = self.active_base_model
            psf = generate_psf(config)
            print(psf.max())
            self.viewer.add_image(
                psf,
                name=f"PSF {config.mode.name} {config} ",
                metadata={"is_psf": True, "config": config},
            )


        



