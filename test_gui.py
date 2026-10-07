import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6 import QtWidgets
from rcjmirror.gui import MainWindow, POINTS
app = QtWidgets.QApplication(sys.argv)
w = MainWindow(); w.show()
for i in range(w.tabs.count()):
    w.tabs.setCurrentIndex(i); w.refresh()
w.tabs.setCurrentIndex(0)
for k in range(w.preset.count()):
    w.preset.setCurrentIndex(k); w.refresh()
    print(w.preset.currentText(), "ok")
w.editor.calculate(); w.refresh(); print("calc ->", w.preset.currentText())
w.editor.add_point(); w.editor.table.setCurrentCell(2,0); w.editor.delete_point(); w.refresh()
w.grab().save("layout.png"); w.editor.fig.savefig("editor.png", dpi=90)
