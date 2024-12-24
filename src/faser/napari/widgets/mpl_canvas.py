import sys
import typing

import matplotlib
import numpy as np
from qtpy.QtWidgets import QWidget

matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.cm import coolwarm, turbo, viridis
from matplotlib.figure import Figure
from qtpy import QtCore, QtWidgets


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=300):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MatplotlibDialog(QtWidgets.QDialog):

    def __init__(self, title, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle(title)
        self.title = title

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setSpacing(1)
        self.setLayout(self.layout)

        self.again = QtWidgets.QPushButton("Popup")

        self.canvas = MplCanvas(self, width=5, height=5, dpi=100)
        self.canvas.axes.set_aspect("auto")
        self.layout.addWidget(self.canvas)

        self.layout.addWidget(self.again)
        self.again.clicked.connect(self.show_another)
        self.another = None

    def update(self, wavefront, new_title):
        self.data = wavefront
        self.canvas.axes.imshow(wavefront)
        self.canvas.axes.set_title(new_title)
        self.canvas.draw()

    def show_another(self) -> None:
        self.another = self.__class__(parent=self, title="Copy of " + self.title)
        self.another.update(self.data, "Copy of " + self.title)
        self.another.show()


class WavefrontDialog(MatplotlibDialog):

    def __init__(self, title, *args, **kwargs) -> None:
        super().__init__(title, *args, **kwargs)
        self.colorbar = None

    def update(self, wavefront, new_title):
        if self.colorbar is not None:
            self.colorbar.remove()
        self.canvas.axes.set_title(new_title)
        self.data = wavefront
        image = self.canvas.axes.imshow(wavefront, cmap=coolwarm)
        image.set_clim(-np.pi, np.pi)
        self.colorbar = self.canvas.fig.colorbar(
            image, ax=self.canvas.axes, orientation="vertical"
        )
        self.canvas.draw()


class MaximumDialog(MatplotlibDialog):

    def __init__(self, title, *args, **kwargs) -> None:
        super().__init__(title, *args, **kwargs)
        self.colorbar = None
        self.vals = None
        self.labels = None

    def update(self, vals: np.array, labels: np.array, new_title: str) -> None:

        if not isinstance(vals, np.ndarray):
            raise TypeError("vals must be a numpy array")

        self.vals = vals
        self.labels = labels

        shape = vals.shape

        self.canvas.axes.clear()

        if len(shape) == 1:
            length = shape[0]
            if length == 1:
                self.canvas.axes.bar([0], vals)
                self.canvas.axes.set_ylabel("Values")
                self.canvas.axes.set_xticks([0])
                self.canvas.axes.set_xticklabels(labels)
                self.canvas.axes.text(
                    0,
                    vals[0],
                    f"{vals[0]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

            else:

                self.canvas.axes.plot(vals)
                self.canvas.axes.set_ylabel("Values")
                if labels is not None:
                    self.canvas.axes.set_xticks(range(len(vals)))
                    self.canvas.axes.set_xticklabels(labels)

        if len(shape) == 2:

            self.canvas.axes.imshow(vals, cmap="coolwarm")
            self.canvas.axes.set_ylabel("Values")
            self.canvas.axes.set_label(labels)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    self.canvas.axes.text(
                        j,
                        i,
                        f"{vals[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

        if len(shape) == 3:
            raise NotImplementedError("3D data not supported")

        self.canvas.axes.set_title(new_title)
        self.canvas.draw()

    def show_another(self) -> None:
        self.another = self.__class__(parent=self, title="Copy of " + self.title)
        self.another.update(self.vals, self.labels, "Copy of " + self.title)
        self.another.show()


class BeamDialog(MatplotlibDialog):

    def __init__(self, title, *args, **kwargs) -> None:
        super().__init__(title, *args, **kwargs)
        self.colorbar = None

    def update(self, wavefront, new_title):
        if self.colorbar is not None:
            self.colorbar.remove()
        self.canvas.axes.set_title(new_title)
        self.data = wavefront
        image = self.canvas.axes.imshow(wavefront, cmap=viridis)
        image.set_clim(0, 1)
        self.colorbar = self.canvas.fig.colorbar(
            image, ax=self.canvas.axes, orientation="vertical"
        )
        self.canvas.draw()


class PhaseMaskDialog(MatplotlibDialog):

    def __init__(self, title, *args, **kwargs) -> None:
        super().__init__(title, *args, **kwargs)
        self.colorbar = None

    def update(self, wavefront, new_title):
        if self.colorbar is not None:
            self.colorbar.remove()
        self.canvas.axes.set_title(new_title)
        self.data = wavefront
        image = self.canvas.axes.imshow(wavefront, cmap=turbo)
        image.set_clim(-np.pi, np.pi)
        self.colorbar = self.canvas.fig.colorbar(
            image, ax=self.canvas.axes, orientation="vertical"
        )
        self.canvas.draw()
