import sys, traceback
sys.excepthook = lambda *a: traceback.print_exception(*a)
from PySide6 import QtWidgets, QtCore
from PySide6.QtTest import QTest
from rcjmirror.gui import MainWindow, POINTS
app = QtWidgets.QApplication(sys.argv)
w = MainWindow(); w.show()
def pump(ms=300): QTest.qWait(ms)
pump(800)
print("shown; geometry", w.geometry())
w.preset.setCurrentText(POINTS); pump(500); print("points selected")
ed = w.editor
# simulate mouse drag on the first inner point via matplotlib events
from matplotlib.backend_bases import MouseEvent
tr = ed.ax.transData.transform(ed.pts[1])
c = ed.canvas
for name, x, y in (("button_press_event", tr[0], tr[1]), ("motion_notify_event", tr[0]+10, tr[1]+10), ("button_release_event", tr[0]+10, tr[1]+10)):
    ev = MouseEvent(name, c, x, y, button=1)
    c.callbacks.process(name, ev)
pump(500); print("dragged", ed.pts[1])
ed.add_point(); pump(300); print("added")
ed.table.setCurrentCell(2, 0); ed.delete_point(); pump(300); print("deleted")
ed.table.item(1, 1).setText("-12"); pump(300); print("cell edited")
ed.chk_smooth.setChecked(False); pump(300); print("smooth off")
ed.calculate(); pump(500); print("calculated")
for i in range(w.tabs.count()):
    w.tabs.setCurrentIndex(i); pump(500); print("tab", i)
print("DONE")
w.dock_prof.close(); pump(300); print("editor hidden:", not w.dock_prof.isVisible())
w.dock_prof.toggleViewAction().trigger(); pump(300); print("editor shown:", w.dock_prof.isVisible())
w.preset.setCurrentIndex(1); pump(300); print("still works")
