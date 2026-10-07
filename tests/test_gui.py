import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def win():
    from rcjmirror.gui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    yield w
    w.close()
    app.processEvents()


def test_main_window_all_tabs_and_presets(win):
    from rcjmirror.gui import POINTS

    w = win
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i)
        w.refresh()
    w.tabs.setCurrentIndex(0)
    for k in range(w.preset.count()):
        w.preset.setCurrentIndex(k)
        w.refresh()
    w.editor.calculate()
    w.refresh()
    assert w.preset.currentText() == POINTS
    assert "img" in w._last and w._last["img"].shape[:2] == (w.p.camera.height, w.p.camera.width)


def test_params_round_trip(win, tmp_path):
    from rcjmirror.gui import POINTS

    w = win
    w.preset.setCurrentText(POINTS)
    w.editor.pts = [(0.0, -18.0), (12.0, -11.0), (24.0, -4.0), (32.0, 0.0)]
    w.editor.changed.emit()
    w.league.setCurrentIndex(1)
    for name, v in (("h_cam", 95.0), ("d_lm", 52.0), ("hfov", 70.8), ("body_h", 75.0),
                    ("rx", 120.0), ("bx", -300.0)):
        getattr(w, name).setValue(v)
    w.editor.d_far.setValue(1800.0)
    w.refresh()
    saved = w.params_dict()
    img_before = w.sim.render(w.p, 80, 60)
    path = tmp_path / "params.json"
    w.save_params(path)
    assert json.loads(path.read_text(encoding="utf-8"))["h_cam"] == 95.0

    # scramble, then load
    w.preset.setCurrentIndex(0)
    w.league.setCurrentIndex(0)
    for name in ("h_cam", "d_lm", "body_h", "rx", "bx"):
        getattr(w, name).setValue(getattr(w, name).value() + 13)
    w.editor.d_far.setValue(900.0)
    w.load_params(path)
    w.refresh()

    assert w.params_dict() == saved
    assert w.h_mir.value() == pytest.approx(95.0 + 52.0)
    assert w.m_r.isEnabled() is False
    assert np.array_equal(w.sim.render(w.p, 80, 60), img_before)


def test_missing_step_file_falls_back_to_profile(win, monkeypatch):
    w = win
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: shown.append(a))
    d = w.params_dict()
    d["step_path"] = "Z:/no/such/mirror.step"
    w.apply_params(d)
    assert shown and w.p.mirror.step_path == ""
