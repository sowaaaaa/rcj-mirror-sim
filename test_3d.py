import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pyvista as pv
from PySide6 import QtWidgets
from rcjmirror.gui import MainWindow
app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
w._read(); w.view_3d()
w.plotter.screenshot("scene3d.png")
w.tabs.setCurrentIndex(3); w.refresh(); 
w.t_cal.fig.savefig("cal.png", dpi=80)
