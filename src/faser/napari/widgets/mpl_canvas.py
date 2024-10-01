import sys
import typing
from PyQt5.QtWidgets import QWidget
import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.cm import coolwarm, viridis, turbo

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=300):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MatplotlibDialog(QtWidgets.QDialog):

    def __init__(self, title, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle(title)
        self.title=title

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setSpacing(1)
        self.setLayout(self.layout)

        self.again = QtWidgets.QPushButton("Popup")

        self.canvas = MplCanvas(self, width=5, height=5, dpi=100)
        self.canvas.axes.set_aspect('auto')
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
        self.colorbar = self.canvas.fig.colorbar(image, ax=self.canvas.axes, orientation='vertical')
        self.canvas.draw()
    

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
        self.colorbar = self.canvas.fig.colorbar(image, ax=self.canvas.axes, orientation='vertical')
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
        self.colorbar = self.canvas.fig.colorbar(image, ax=self.canvas.axes, orientation='vertical')
        self.canvas.draw()