# import itertools
# import os
import typing
from enum import Enum
from typing import Any, Callable, List, Type

# import dask
# import dask.array as da
import napari
import numpy as np
import pydantic
from annotated_types import Ge, Gt, Le, Lt

# from faser.generators.base import AberrationFloat, PSFConfig
# from faser.generators.vectorial.stephane.tilted_coverslip import generate_psf
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget
from superqt import QEnumComboBox  # QDoubleRangeSlider,
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider

from faser.env import get_asset_file

batch_png = QtGui.QPixmap(get_asset_file("batch.png"))
single_png = QtGui.QPixmap(get_asset_file("single.png"))


def get_field_gt(field: FieldInfo) -> float:
    for metadata in field.metadata:
        if isinstance(metadata, Gt):
            return metadata.gt
        if isinstance(metadata, Ge):
            return metadata.ge


def get_field_lt(field: FieldInfo) -> float:
    for metadata in field.metadata:
        if isinstance(metadata, Lt):
            return metadata.lt
        if isinstance(metadata, Le):
            return metadata.le


class FormField(QtWidgets.QWidget):
    on_child_value_changed = Signal(str, object)
    on_child_range_value_changed = Signal(str, object)

    def __init__(
        self, key: str, field: FieldInfo, toggable: bool = True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.field = field
        self.key = key
        self.label = key
        self.child = None
        self.mode = "single"
        self.description = field.description
        self.steps = 3
        self.toggable = toggable
        self.range_child = None

    def replace_widget(self, oldwidget, newwidget):
        self.xlayout.removeWidget(oldwidget)
        oldwidget.setParent(None)
        self.xlayout.addWidget(newwidget)

    def reset_value(self, value):
        raise NotImplementedError("reset_value() must be implemented")

    def reset_default(self):
        self.reset_value(self.field.default)

    def emit_child_value_changed(self, value):
        self.on_child_value_changed.emit(self.key, value)

    def on_child_range_changed(self, value):
        print("Emitting range change")
        self.on_child_range_value_changed.emit(self.key, value)

    def on_change_mode(self):

        if self.mode == "single":
            self.toggle_button.setIcon(QtGui.QIcon(batch_png))
            self.toggle_button.setIconSize(QtCore.QSize(15, 15))
            self.on_child_range_changed(
                FloatRange(
                    min=get_field_gt(self.field) or self.field.default,
                    max=get_field_lt(self.field) or self.field.default,
                    steps=3,
                )
            )
            self.replace_widget(self.child, self.range_child)

        elif self.mode == "range":
            self.toggle_button.setIcon(QtGui.QIcon(single_png))
            self.toggle_button.setIconSize(QtCore.QSize(15, 15))
            self.emit_child_value_changed(self.field.default)
            self.on_child_range_changed(None)
            self.replace_widget(self.range_child, self.child)

        self.mode = "range" if self.mode == "single" else "single"

    def init_ui(self):
        assert self.child is not None, "Child widget must be set before init_ui()"
        assert (
            self.range_child is not None
        ), "Range_child widget must be set before init_ui()"
        self.xlayout = QtWidgets.QVBoxLayout()
        self.xlayout.setContentsMargins(0, 0, 0, 0)
        self.xlayout.setSpacing(1)

        self.label = QtWidgets.QLabel(self.label)
        self.label.setToolTip(self.description or "No description yet")
        layout = QtWidgets.QHBoxLayout()
        self.labelWidget = QtWidgets.QWidget()

        if self.toggable:
            self.toggle_button = QtWidgets.QPushButton()
            self.toggle_button.setIcon(QtGui.QIcon(single_png))
            self.toggle_button.setIconSize(QtCore.QSize(15, 15))
            self.toggle_button.clicked.connect(self.on_change_mode)
            layout.addWidget(self.toggle_button)

        layout.addWidget(self.label)
        layout.addStretch()

        # if self.mode == "single":
        #     layout.addWidget(self.child)
        # elif self.mode == "range":
        #     layout.addWidget(self.range_child)

        self.labelWidget.setLayout(layout)
        self.xlayout.addWidget(self.labelWidget)
        self.xlayout.addWidget(self.child)

        self.setLayout(self.xlayout)
        # self.setLayout(layout)

    def init_ui_helper_eff(self):
        assert self.child is not None, "Child widget must be set before init_ui()"
        assert (
            self.range_child is not None
        ), "Range_child widget must be set before init_ui()"
        self.xlayout = QtWidgets.QVBoxLayout()
        self.xlayout.setContentsMargins(0, 0, 0, 0)
        self.xlayout.setSpacing(1)

        self.label = QtWidgets.QLabel(self.label)
        self.label.setToolTip(self.description or "No description yet")
        layout = QtWidgets.QHBoxLayout()
        self.labelWidget = QtWidgets.QWidget()

        if self.toggable:
            self.toggle_button = QtWidgets.QPushButton()
            self.toggle_button.setIcon(QtGui.QIcon(single_png))
            self.toggle_button.setIconSize(QtCore.QSize(15, 15))
            self.toggle_button.clicked.connect(self.on_change_mode)
            layout.addWidget(self.toggle_button)

        layout.addWidget(self.label)
        # layout.addStretch()

        # if self.mode == "single":
        #     layout.addWidget(self.child)
        # elif self.mode == "range":
        #     layout.addWidget(self.range_child)

        self.labelWidget.setLayout(layout)
        self.xlayout.addWidget(self.labelWidget)
        self.xlayout.addWidget(self.child)

        self.setLayout(self.xlayout)
        # self.setLayout(layout)

    def init_ui_helper(self):
        # self.xlayout = QtWidgets.QVBoxLayout()
        self.xlayout = QtWidgets.QHBoxLayout()
        self.xlayout.setContentsMargins(0, 0, 0, 0)
        self.xlayout.setSpacing(1)

        self.label = QtWidgets.QLabel(self.label)
        self.label.setToolTip(self.description or "No description yet")
        layout = QtWidgets.QHBoxLayout()
        self.labelWidget = QtWidgets.QWidget()

        layout.addWidget(self.label)
        # layout.addStretch()

        self.labelWidget.setLayout(layout)
        self.xlayout.addWidget(self.labelWidget)
        self.xlayout.addWidget(self.child)

        self.setLayout(self.xlayout)
        # self.setLayout(layout)


class FloatRange(pydantic.BaseModel):
    min: float
    max: float
    steps: int

    def to_list(self):
        return np.linspace(self.min, self.max, self.steps, dtype=np.float64).tolist()


class FloatList(pydantic.BaseModel):
    items: List[float]

    def to_list(self):
        return self.items


class IntList(pydantic.BaseModel):
    items: List[int]

    def to_list(self):
        return self.items


class FloatRangeStepSliderField(QtWidgets.QWidget):
    on_range_changed = Signal(object)

    def __init__(self, *args, gt=None, lt=None, steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.layout = QtWidgets.QHBoxLayout()
        self.xlayout = QtWidgets.QHBoxLayout()
        self.xlayout.setContentsMargins(0, 0, 0, 0)
        self.xlayout.setSpacing(1)

        self.range_slider = QLabeledDoubleRangeSlider(QtCore.Qt.Horizontal)
        self.range_slider.setRange(gt or 0.0, lt or 1.0)
        self.range_slider.setValue([gt or 0.0, lt or 1.0])
        self.range_slider.setStyleSheet("QWidget {font-size: 8pt;}")

        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setFixedWidth(15)
        self.text_input.setText(str(steps) or 3)
        self.text_input.setValidator(QtGui.QIntValidator())

        self.range_slider.valueChanged.connect(self.on_range_valued_callback)
        # self.layout.addWidget(self.range_slider)
        # self.layout.addWidget(self.text_input)
        self.xlayout.addWidget(self.range_slider)
        self.xlayout.addSpacing(25)
        self.xlayout.addWidget(self.text_input)
        self.text_input.textChanged.connect(self.on_text_changed)

        # self.setLayout(self.layout)
        self.setLayout(self.xlayout)

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


class FloatRangeArrayField(QtWidgets.QWidget):
    on_range_changed = Signal(object)

    def __init__(self, default, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.layout = QtWidgets.QHBoxLayout()
        self.xlayout = QtWidgets.QVBoxLayout()
        self.xlayout.setContentsMargins(0, 0, 0, 0)
        self.xlayout.setSpacing(1)

        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setText(str(default))

        self.add_button = QtWidgets.QPushButton("+")
        self.add_button.setFixedWidth(30)
        self.add_button.clicked.connect(self.on_add_button)

        self.add_bar = QtWidgets.QHBoxLayout()
        self.add_bar.setContentsMargins(2, 2, 2, 2)
        self.add_bar.addWidget(self.text_input)
        self.add_bar.addWidget(self.add_button)

        self.value_items = QtWidgets.QListWidget()
        self.value_items.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.value_items.clicked.connect(self.mousePressEvent)

        self.chosen_items = set()

        # self.layout.addWidget(self.range_slider)
        # self.layout.addWidget(self.text_input)
        self.xlayout.addWidget(self.value_items)
        self.xlayout.addLayout(self.add_bar)

        # self.setLayout(self.layout)
        self.setLayout(self.xlayout)

    def mousePressEvent(self, event):
        if self.value_items.currentItem() is None:
            return
        self.chosen_items.remove(float(self.value_items.currentItem().text()))
        self.on_range_changed.emit(FloatList(items=list(self.chosen_items)))
        self.update_ui()

    def update_ui(self):
        self.value_items.clear()
        for item in self.chosen_items:
            self.value_items.addItem(str(item))

    def on_add_button(self):
        self.chosen_items.add(float(self.text_input.text()))
        self.on_range_changed.emit(FloatList(items=list(self.chosen_items)))
        self.update_ui()


class IntRangeStepSliderField(QtWidgets.QWidget):
    on_range_changed = Signal(object)

    def __init__(self, *args, gt=None, lt=None, steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = QtWidgets.QHBoxLayout()

        self.range_slider = QLabeledDoubleRangeSlider(QtCore.Qt.Horizontal)
        self.range_slider.setRange(gt or 0.0, lt or 1.0)
        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setFixedWidth(20)
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
        self.child.setMinimum(get_field_gt(self.field) or 0.0)
        self.child.setValue(self.field.default)
        self.child.setMaximum(get_field_lt(self.field) or 1.0)
        self.child.valueChanged.connect(self.emit_child_value_changed)

        self.range_child = FloatRangeStepSliderField(
            gt=get_field_gt(self.field) or 0.0,
            lt=get_field_lt(self.field) or 1,
            steps=3,
        )

        self.range_child.on_range_changed.connect(self.on_child_range_changed)

    def reset_value(self, value):
        self.child.setValue(value)


class IntRangeArrayField(QtWidgets.QWidget):
    on_range_changed = Signal(object)

    def __init__(self, default, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.layout = QtWidgets.QHBoxLayout()
        self.xlayout = QtWidgets.QVBoxLayout()
        self.xlayout.setContentsMargins(0, 0, 0, 0)
        self.xlayout.setSpacing(1)

        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setText(str(default))

        self.add_button = QtWidgets.QPushButton("+")
        self.add_button.setFixedWidth(30)
        self.add_button.clicked.connect(self.on_add_button)

        self.add_bar = QtWidgets.QHBoxLayout()
        self.add_bar.setContentsMargins(2, 2, 2, 2)
        self.add_bar.addWidget(self.text_input)
        self.add_bar.addWidget(self.add_button)

        self.value_items = QtWidgets.QListWidget()
        self.value_items.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.value_items.clicked.connect(self.mousePressEvent)

        self.chosen_items = set()

        # self.layout.addWidget(self.range_slider)
        # self.layout.addWidget(self.text_input)
        self.xlayout.addWidget(self.value_items)
        self.xlayout.addLayout(self.add_bar)

        # self.setLayout(self.layout)
        self.setLayout(self.xlayout)

    def update_ui(self):
        self.value_items.clear()
        for item in self.chosen_items:
            self.value_items.addItem(str(item))

    def mousePressEvent(self, event):
        if self.value_items.currentItem() is None:
            return
        self.chosen_items.remove(int(self.value_items.currentItem().text()))
        self.on_range_changed.emit(IntList(items=list(self.chosen_items)))
        self.update_ui()

    def on_add_button(self):
        self.chosen_items.add(float(self.text_input.text()))
        self.on_range_changed.emit(IntList(items=list(self.chosen_items)))
        self.update_ui()


class IntInputField(FormField):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QtWidgets.QLineEdit()
        self.child.setText(str(self.field.default))
        self.child.setValidator(QtGui.QIntValidator())
        self.child.textChanged.connect(self.emit_text_changed)

        self.range_child = IntRangeArrayField(
            default=self.field.default,
        )
        self.range_child.on_range_changed.connect(self.on_child_range_changed)

    def emit_text_changed(self, value):
        self.emit_child_value_changed(int(self.child.text()))

    def reset_value(self, value):
        self.child.setText(str(value))


class FloatInputField(FormField):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QtWidgets.QLineEdit()
        self.child.setText(str(self.field.default))
        self.child.textChanged.connect(self.emit_text_changed)

        self.range_child = FloatRangeArrayField(
            default=self.field.default,
        )
        self.range_child.on_range_changed.connect(self.on_child_range_changed)

    def emit_text_changed(self, value):
        self.emit_child_value_changed(float(self.child.text()))

    def reset_value(self, value):
        self.child.setText(str(value))


class OptionRange(pydantic.BaseModel):
    options: List[Any]

    def to_list(self):
        return self.options


class MultiEnumField(QtWidgets.QWidget):
    on_range_changed = Signal(object)

    def __init__(self, *args, enum: Enum, **kwargs):
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

    def reset_value(self, value):
        for i, check in enumerate(self.checkboxes):
            if self.checkable_values[i] in value:
                check.setChecked(True)
            else:
                check.setChecked(False)


class EnumField(FormField):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child = QEnumComboBox()
        self.child.setEnumClass(self.field.annotation)
        self.child.setCurrentEnum(self.field.default)
        self.child.currentEnumChanged.connect(self.emit_child_value_changed)

        self.range_child = MultiEnumField(enum=self.field.annotation)
        self.range_child.on_range_changed.connect(self.on_child_range_changed)
        # TODO: Implement

    def reset_value(self, value):
        self.child.setCurrentEnum(value)


def generate_single_widgets_from_model(
    model: Type[BaseModel],
    callback: Callable[[str, Any], None],
    range_callback: Callable[[str, Any], None] = None,
    parent=None,
    filter_fields=None,
) -> typing.List[FormField]:

    widgets = []

    for key, value in model.model_fields.items():

        if filter_fields:
            if not filter_fields(key, value):
                continue

        widget = None

        field_type = value.annotation
        print(field_type)

        if issubclass(field_type, float):
            if get_field_gt(value) is None:
                widget = FloatInputField(key, value, parent=parent)
            else:
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
) -> typing.Callable[[str, FieldInfo], bool]:
    def key_filter(key, value):
        return key in field_set

    return key_filter
