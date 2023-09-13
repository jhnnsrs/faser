import itertools
import os
import typing
from enum import Enum
from typing import Any, Callable, List, Type

import dask
import dask.array as da
import napari
import numpy as np
import pydantic
from pydantic.fields import ModelField
from pydantic.types import ConstrainedFloat
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget
from qtpy import QtGui, QtWidgets
from superqt import (
    QDoubleRangeSlider,
    QEnumComboBox,
    QLabeledDoubleRangeSlider,
    QLabeledDoubleSlider,
)

from faser.env import get_asset_file
from faser.generators.base import AberrationFloat, PSFConfig
from faser.generators.vectorial.stephane.tilted_coverslip import generate_psf
from pydantic import BaseModel

class FormField(QtWidgets.QWidget):
    on_child_value_changed = QtCore.pyqtSignal(str, object)
    on_child_range_value_changed = QtCore.pyqtSignal(str, object)

    def __init__(
        self, key: str, field: ModelField, toggable: bool = True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.field = field
        self.key = key
        self.label = key
        self.child = None
        self.mode = "single"
        self.description = field.field_info.description
        self.steps = 3
        self.toggable = toggable
        self.range_child = None

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
        layout = QtWidgets.QHBoxLayout()
        self.labelWidget = QtWidgets.QWidget()

        layout.addWidget(self.label)
        layout.addStretch()

        if self.toggable:
            self.toggle_button = QtWidgets.QPushButton("Change Mode")
            self.toggle_button.clicked.connect(self.on_change_mode)
            layout.addWidget(self.toggle_button)

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


class FloatRangeStepSliderField(QtWidgets.QWidget):
    on_range_changed = QtCore.pyqtSignal(object)

    def __init__(self, *args, gt=None, lt=None, steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = QtWidgets.QHBoxLayout()

        self.range_slider = QLabeledDoubleRangeSlider(QtCore.Qt.Horizontal)
        self.range_slider.setRange(gt or 0.0, lt or 1.0)
        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setFixedWidth(30)
        self.text_input.setText(str(steps) or 3)
        self.text_input.setValidator(QtGui.QIntValidator())

        self.range_slider.valueChanged.connect(self.on_range_valued_callback)
        self.layout.addWidget(self.range_slider)
        self.layout.addWidget(self.text_input)
        self.text_input.textChanged.connect(self.on_text_changed)
        self.setLayout(self.layout)

    def on_range_valued_callback(self, value):
        self.on_range_changed.emit(
            FloatRange(
                min=float(value[0]),
                max=float(value[1]),
                steps=int(self.text_input.text()),
            )
        )

    def on_text_changed(self, value):
        self.on_range_changed.emit(
            FloatRange(
                min=float(self.range_slider.value()[0]),
                max=float(self.range_slider.value()[1]),
                steps=int(value),
            )
        )


class IntRange(pydantic.BaseModel):
    min: int
    max: int
    steps: int

    def to_list(self):
        return np.linspace(self.min, self.max, self.steps, dtype=np.int64).tolist()


class IntRangeStepSliderField(QtWidgets.QWidget):
    on_range_changed = QtCore.pyqtSignal(object)

    def __init__(self, *args, gt=None, lt=None, steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = QtWidgets.QHBoxLayout()

        self.range_slider = QLabeledDoubleRangeSlider(QtCore.Qt.Horizontal)
        self.range_slider.setRange(gt or 0.0, lt or 1.0)
        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setFixedWidth(30)
        self.text_input.setText(str(steps) or 3)
        self.text_input.setValidator(QtGui.QIntValidator())

        self.range_slider.valueChanged.connect(self.on_range_valued_callback)

        self.layout.addWidget(self.range_slider)
        self.layout.addWidget(self.text_input)
        self.text_input.textChanged.connect(self.on_text_changed)
        self.setLayout(self.layout)

    def on_range_valued_callback(self, value):
        self.on_range_changed.emit(
            IntRange(
                min=int(value[0]), max=int(value[1]), steps=int(self.text_input.text())
            )
        )

    def on_text_changed(self, value):
        self.on_range_changed.emit(
            IntRange(
                min=int(self.range_slider.value()[0]),
                max=int(self.range_slider.value()[1]),
                steps=int(value),
            )
        )


class FloatSliderField(FormField):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QLabeledDoubleSlider(QtCore.Qt.Horizontal)
        self.child.setMinimum(self.field.field_info.gt or 0.0)
        self.child.setValue(self.field.default)
        self.child.setMaximum(self.field.field_info.lt or 1.0)
        self.child.valueChanged.connect(self.emit_child_value_changed)

        self.range_child = FloatRangeStepSliderField(
            gt=self.field.field_info.gt or 0.0, lt=self.field.field_info.lt or 1, steps=3
        )
        self.range_child.on_range_changed.connect(self.on_child_range_changed)


class IntInputField(FormField):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QtWidgets.QLineEdit()
        self.child.setText(str(self.field.default))
        self.child.setValidator(QtGui.QIntValidator())
        self.child.textChanged.connect(self.emit_text_changed)

        self.range_child = IntRangeStepSliderField(
            gt=self.field.field_info.gt or 0.0, lt=self.field.field_info.lt or 1, steps=3
        )
        self.range_child.on_range_changed.connect(self.on_child_range_changed)

    def emit_text_changed(self, value):
        self.emit_child_value_changed(int(self.child.text()))


class FloatInputField(FormField):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QtWidgets.QLineEdit()
        self.child.setText(str(self.field.default))
        self.child.textChanged.connect(self.emit_text_changed)

        self.range_child = FloatRangeStepSliderField(
            gt=self.field.field_info.gt or 0.0, lt=self.field.field_info.lt or 1, steps=3
        )
        self.range_child.on_range_changed.connect(self.on_child_range_changed)

    def emit_text_changed(self, value):
        self.emit_child_value_changed(float(self.child.text()))

class OptionRange(pydantic.BaseModel):
    options: List[Any]

    def to_list(self):
        return self.options


class MultiEnumField(QtWidgets.QWidget):
    on_range_changed = QtCore.pyqtSignal(object)

    def __init__(self, *args, enum: Enum,  **kwargs):
        super().__init__(*args, **kwargs)
        self.enum = enum
        self.layout = QtWidgets.QHBoxLayout()
        self.checkboxes = []
        self.checkable_values = []

        for i in enum:
            check = QtWidgets.QPushButton(i.name)
            check.setCheckable(True)
            check.clicked.connect(self.on_change_callback)

            self.checkboxes.append(check)
            self.checkable_values.append(i.value)
            self.layout.addWidget(check)

        self.setLayout(self.layout)

    def on_change_callback(self):
        print("Changed")
        check_enums = []

        for i, check in enumerate(self.checkboxes):
            if check.isChecked():
                check_enums.append(self.checkable_values[i])
            else:
                False

        self.on_range_changed.emit(OptionRange(options=check_enums))
        

    
class EnumField(FormField):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QEnumComboBox()
        self.child.setEnumClass(self.field.type_)
        self.child.currentEnumChanged.connect(self.emit_child_value_changed)
        

        self.range_child = MultiEnumField(enum=self.field.type_)
        self.range_child.on_range_changed.connect(self.on_child_range_changed)
        # TODO: Implement


def generate_single_widgets_from_model(
    model: Type[BaseModel],
    callback: Callable[[str, Any], None],
    range_callback: Callable[[str, Any], None] = None,
    parent=None,
    filter_fields=None,
) -> typing.List[QWidget]:

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

        elif issubclass(field_type, Enum):
            widget = EnumField(key, value, parent=parent)

        else:
            print(field_type)

        if widget is not None:
            if callback:
                widget.on_child_value_changed.connect(callback)
            if range_callback:
                widget.on_child_range_value_changed.connect(range_callback)
            widgets.append(widget)

    return widgets


def build_key_filter(
    field_set: typing.Set[str],
) -> typing.Callable[[str, ModelField], bool]:
    def key_filter(key, value):
        return key in field_set

    return key_filter
