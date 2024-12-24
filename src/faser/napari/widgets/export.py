import sys

import matplotlib

matplotlib.use("Qt5Agg")

import napari
from matplotlib.backends.backend_qt import FigureCanvasQT
from matplotlib.figure import Figure
from qtpy import QtCore, QtWidgets


class MplCanvas(FigureCanvasQT):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class TopBar(QtWidgets.QWidget):
    def __init__(self, viewer: napari.Viewer, *args, **kwargs):
        super(TopBar, self).__init__(*args, **kwargs)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.layout = QtWidgets.QHBoxLayout()
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])

        self.layout.addWidget(sc)
        self.setLayout(self.layout)
        self.show()
