import sys
import typing
from PyQt5.QtWidgets import QWidget
import matplotlib

matplotlib.use("Qt5Agg")

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)




class MatplotlibDialog(QtWidgets.QDialog):

    def __init__(self, title, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle(title)
        self.title=title

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.again = QtWidgets.QPushButton("Popup")

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
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
        self.another = MatplotlibDialog(parent=self, title="Copy of " + self.title)
        self.another.update(self.data, "Copy of " + self.title)
        self.another.show()




    


