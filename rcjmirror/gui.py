"""Qt GUI: parameters on the left, views on the right."""
import sys

import matplotlib
matplotlib.use("QtAgg")
import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from . import analysis as A
from . import config as C
from . import field as F
from .mirror import PRESETS, profile_curve
from .profile_editor import ProfileEditor
from .render import Simulator

POINTS = "По точкам (редактор)"
RESOLUTIONS = {"160 x 120": (160, 120), "320 x 240 (QVGA)": (320, 240), "640 x 480 (VGA)": (640, 480)}


def _arr_to_pixmap(arr, scale_to=None):
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    img = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(img)


class ImageView(QtWidgets.QLabel):
    """Shows an image scaled to the widget (no smoothing, so real pixels stay visible)."""

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self._pm = None

    def set_array(self, arr):
        self._pm = _arr_to_pixmap(arr)
        self._rescale()

    def resizeEvent(self, e):
        self._rescale()

    def _rescale(self):
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))


class MplTab(QtWidgets.QWidget):
    def __init__(self, nrows=1, ncols=1):
        super().__init__()
        self.fig = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.axes = self.fig.subplots(nrows, ncols, squeeze=False)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RCJ Soccer: симулятор зеркальной камеры")
        geo = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(int(geo.width() * 0.96), int(geo.height() * 0.92))
        from .config import SimParams
        self.p = SimParams()
        self.sim = Simulator()
        self.sim.set_mirror(profile=("Конус",) + PRESETS["Конус"][:2])
        self._busy = False
        self._last = {}

        self.timer = QtCore.QTimer(singleShot=True, interval=150)
        self.timer.timeout.connect(self.refresh)

        self._build_ui()
        self._push_preset_to_editor()
        self._ready = True
        self._sync_heights(origin="cam")
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _spin(self, form, label, lo, hi, val, step=1.0, dec=1, suffix=" мм", cb=None):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setSingleStep(step)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        s.valueChanged.connect(cb or self.changed)
        form.addRow(label, s)
        return s

    def _build_ui(self):
        panel = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(panel)
        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        left.setWidget(panel)
        left.setFixedWidth(420)
        left.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # robot / league
        g = QtWidgets.QGroupBox("Робот")
        f = QtWidgets.QFormLayout(g)
        self.league = QtWidgets.QComboBox()
        self.league.addItems(list(C.ROBOT_PRESETS))
        self.league.currentIndexChanged.connect(self.changed)
        f.addRow("Лига", self.league)
        self.body_h = self._spin(f, "Высота корпуса (сплошной)", 0, 250, self.p.robot.body_height)
        self.rx = self._spin(f, "Робот X", -900, 900, 0, 10)
        self.ry = self._spin(f, "Робот Y", -1200, 1200, 0, 10)
        self.rh = self._spin(f, "Курс", -360, 360, 0, 5, suffix="°")
        lv.addWidget(g)

        # the three main distances
        g = QtWidgets.QGroupBox("Главные расстояния")
        f = QtWidgets.QFormLayout(g)
        self.h_cam = self._spin(f, "Камера (конец объектива) над полем", 0, 250,
                                self.p.camera.height_above_field, cb=lambda: self._sync_heights("cam"))
        self.d_lm = self._spin(f, "Объектив → 1-я плоскость зеркала", 1, 250,
                               self.p.mirror.lens_to_mirror, cb=lambda: self._sync_heights("d"))
        self.h_mir = self._spin(f, "1-я плоскость зеркала над полем", 1, 400,
                                self.p.mirror_plane_height, cb=lambda: self._sync_heights("mir"))
        lv.addWidget(g)

        # camera
        g = QtWidgets.QGroupBox("Камера OpenMV H7 Plus")
        f = QtWidgets.QFormLayout(g)
        self.res = QtWidgets.QComboBox()
        self.res.addItems(list(RESOLUTIONS))
        self.res.setCurrentIndex(1)
        self.res.currentIndexChanged.connect(self.changed)
        f.addRow("Разрешение", self.res)
        self.hfov = self._spin(f, "HFOV (проверьте для вашего объектива)", 20, 170, self.p.camera.hfov_deg, 1, suffix="°")
        lv.addWidget(g)

        # ball
        g = QtWidgets.QGroupBox("Мяч (оранжевый, 42 мм)")
        f = QtWidgets.QFormLayout(g)
        self.bx = self._spin(f, "Мяч X", -790, 790, self.p.ball_x, 10)
        self.by = self._spin(f, "Мяч Y", -1095, 1095, self.p.ball_y, 10)
        self.det_px = self._spin(f, "Мин. размер для детекции", 1, 50, 6, 1, 0, " px")
        lv.addWidget(g)

        # mirror
        g = QtWidgets.QGroupBox("Зеркало (полированный алюминий)")
        f = QtWidgets.QFormLayout(g)
        self.refl = self._spin(f, "Коэффициент отражения", 0.3, 1.0, self.p.mirror.reflectance, 0.01, 2, "")
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(PRESETS) + [POINTS])
        self.preset.currentIndexChanged.connect(self.on_preset)
        f.addRow("Профиль зеркала", self.preset)
        self.m_r = self._spin(f, "Радиус зеркала", 5, 100, PRESETS["Конус"][0], 0.5, cb=self.on_profile_param)
        self.m_depth = self._spin(f, "Глубина (вершина ниже плоскости)", 1, 100, PRESETS["Конус"][1], 0.5,
                                  cb=self.on_profile_param)
        self.btn_step = QtWidgets.QPushButton("Загрузить STEP зеркала…")
        self.btn_step.clicked.connect(self.load_step)
        self.btn_cone = QtWidgets.QPushButton("Вернуться к профилю из списка")
        self.btn_cone.clicked.connect(self.reset_mirror)
        f.addRow(self.btn_step)
        f.addRow(self.btn_cone)
        self.scale = self._spin(f, "Масштаб файла → мм", 0.001, 1000, 1.0, 1, 3, "", cb=self.reload_mirror)
        self.chk_flip = QtWidgets.QCheckBox("Перевернуть (если вершина смотрит не вниз)")
        self.chk_flip.toggled.connect(self.reload_mirror)
        f.addRow(self.chk_flip)
        self.chk_ref = QtWidgets.QCheckBox("Задать 1-ю плоскость вручную (z в файле)")
        self.chk_ref.toggled.connect(self.reload_mirror)
        f.addRow(self.chk_ref)
        self.ref_z = self._spin(f, "z плоскости в системе файла", -500, 500, 0, 0.5, 2, " мм", cb=self.reload_mirror)
        self.mirror_lbl = QtWidgets.QLabel("")
        self.mirror_lbl.setWordWrap(True)
        f.addRow(self.mirror_lbl)
        lv.addWidget(g)

        self.checks = QtWidgets.QLabel()
        self.checks.setWordWrap(True)
        self.checks.setTextFormat(QtCore.Qt.RichText)
        lv.addWidget(self.checks)
        lv.addStretch(1)

        # right: tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(lambda *_: self.refresh())

        self.v_cam = ImageView()
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(self.v_cam, 1)
        b = QtWidgets.QPushButton("Сохранить кадр в PNG…")
        b.clicked.connect(self.save_png)
        lay.addWidget(b)
        self.tabs.addTab(w, "Камера")

        self.v_top = ImageView()
        self.tabs.addTab(self.v_top, "Вид сверху")
        self.v_pol = ImageView()
        self.tabs.addTab(self.v_pol, "Панорама")
        self.t_cal = MplTab(1, 2)
        self.tabs.addTab(self.t_cal, "Калибровка")
        self.t_ball = MplTab(1, 1)
        self.tabs.addTab(self.t_ball, "Мяч в пикселях")
        self.t_heat = MplTab(1, 1)
        self.tabs.addTab(self.t_heat, "Карта видимости")
        self.t_map = MplTab(1, 1)
        self.t_map.canvas.mpl_connect("button_press_event", self.on_map_click)
        self.dock_map = QtWidgets.QDockWidget("Поле: ЛКМ — мяч, ПКМ — робот", self)
        self.dock_map.setWidget(self.t_map)
        self.dock_map.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
                                  | QtWidgets.QDockWidget.DockWidgetClosable)
        self.t_map.setMinimumWidth(380)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_map)
        self.editor = ProfileEditor()
        self.editor.changed.connect(self.on_editor_changed)
        self.dock_prof = QtWidgets.QDockWidget("Редактор профиля зеркала (тяните красные точки)", self)
        self.dock_prof.setWidget(self.editor)
        self.dock_prof.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
                                   | QtWidgets.QDockWidget.DockWidgetClosable)
        self.editor.setMinimumHeight(200)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock_prof)
        view = self.menuBar().addMenu("Вид")
        act = self.dock_prof.toggleViewAction()
        act.setText("Редактор профиля зеркала")
        act.setShortcut("Ctrl+E")
        view.addAction(act)
        act = self.dock_map.toggleViewAction()
        act.setText("Поле: расстановка робота и мяча")
        act.setShortcut("Ctrl+M")
        view.addAction(act)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#202428")
        self.tabs.addTab(self.plotter.interactor, "3D сцена")

        central = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(central)
        h.addWidget(left)
        h.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.status = self.statusBar()

    # ---------------------------------------------------------- parameters
    def _sync_heights(self, origin):
        """The three heights are linked: mirror plane = camera height + lens-to-mirror distance."""
        for s in (self.h_cam, self.d_lm, self.h_mir):
            s.blockSignals(True)
        if origin == "mir":
            self.d_lm.setValue(max(1.0, self.h_mir.value() - self.h_cam.value()))
        self.h_mir.setValue(self.h_cam.value() + self.d_lm.value())
        for s in (self.h_cam, self.d_lm, self.h_mir):
            s.blockSignals(False)
        self.changed()

    def changed(self, *_):
        self.timer.start()

    def _read(self):
        p = self.p
        p.robot.league = self.league.currentText()
        p.robot.body_height = self.body_h.value()
        p.robot.x, p.robot.y, p.robot.heading_deg = self.rx.value(), self.ry.value(), self.rh.value()
        p.camera.height_above_field = self.h_cam.value()
        p.mirror.lens_to_mirror = self.d_lm.value()
        p.camera.width, p.camera.height = RESOLUTIONS[self.res.currentText()]
        p.camera.hfov_deg = self.hfov.value()
        p.ball_x, p.ball_y = self.bx.value(), self.by.value()
        p.mirror.reflectance = self.refl.value()

    # -------------------------------------------------------------- mirror
    def load_step(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "STEP зеркала", "", "STEP (*.step *.stp)")
        if path:
            self.p.mirror.step_path = path
            self.reload_mirror()

    def _profile_key(self):
        name = self.preset.currentText()
        if name == POINTS:
            return ("points", self.editor.points(), self.editor.smooth)
        return (name, round(self.m_r.value(), 3), round(self.m_depth.value(), 3))

    def _push_preset_to_editor(self):
        name = self.preset.currentText()
        if name != POINTS:
            self.editor.set_curve(*profile_curve(name, self.m_r.value(), self.m_depth.value()))

    def on_editor_changed(self):
        self.preset.blockSignals(True)
        self.preset.setCurrentText(POINTS)
        self.preset.blockSignals(False)
        self.m_r.setEnabled(False)
        self.m_depth.setEnabled(False)
        self.p.mirror.step_path = ""
        self.reload_mirror()

    def on_preset(self, *_):
        name = self.preset.currentText()
        is_pts = name == POINTS
        self.m_r.setEnabled(not is_pts)
        self.m_depth.setEnabled(not is_pts)
        if not is_pts:
            r, d, _ = PRESETS[name]
            for sp, v in ((self.m_r, r), (self.m_depth, d)):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
            self._push_preset_to_editor()
        self.p.mirror.step_path = ""
        self.reload_mirror()

    def on_profile_param(self, *_):
        self._push_preset_to_editor()
        self.p.mirror.step_path = ""
        self.reload_mirror()

    def reset_mirror(self):
        self.p.mirror.step_path = ""
        self.reload_mirror()

    def reload_mirror(self, *_):
        m = self.p.mirror
        ref = self.ref_z.value() if self.chk_ref.isChecked() else None
        try:
            self.sim.set_mirror(m.step_path, self.scale.value(), self.chk_flip.isChecked(), ref, self._profile_key())
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Ошибка зеркала", str(e))
            return
        info = self.sim.mirror_info
        sz = info["size"]
        src = m.step_path or f"профиль: {self.preset.currentText()}"
        txt = f"{src}<br>размер {sz[0]:.1f} × {sz[1]:.1f} × {sz[2]:.1f} мм, плоскость z={info['ref_plane_z']:.2f}"
        if info.get("axis", "Z") != "Z":
            txt += f"<br>ось вращения в файле: {info['axis']} (повёрнуто в Z)"
        if info.get("auto_flipped"):
            txt += "<br>(автоматически перевёрнуто: широкая часть сверху)"
        self.mirror_lbl.setText(txt)
        self.changed()

    # --------------------------------------------------------------- views
    def refresh(self):
        if not getattr(self, "_ready", False):
            return
        if self._busy:
            self.timer.start()
            return
        self._busy = True
        try:
            self._read()
            c = self.p.camera
            half = np.degrees(np.arctan(np.tan(np.radians(c.hfov_deg) / 2) * min(c.width, c.height) / c.width))
            self.editor.set_geometry(self.p.mirror.lens_to_mirror, self.p.mirror_plane_height,
                                     c.height_above_field, half)
            self._update_checks()
            i = self.tabs.currentIndex()
            [self.view_cam, self.view_top, self.view_pol, self.view_cal, self.view_ball,
             self.view_heat, self.view_3d][i]()
            self.view_map()
        finally:
            self._busy = False

    def _render(self):
        p = self.p
        img = self.sim.render(p)
        self._last["img"] = img
        return img

    def view_cam(self):
        self.v_cam.set_array(self._render())

    def _floor(self, z):
        return self.sim.floor_map(self.p, z)

    def view_top(self):
        p = self.p
        W, H = 640, 480
        img = self.sim.render(p, W, H)
        X, Y, valid, _ = self.sim.floor_map(p, 0.0, W, H)
        td, ext = A.top_down(img, X, Y, valid, p)
        self.v_top.set_array(td)

    def view_pol(self):
        img = self.sim.render(self.p, 640, 480)
        self.v_pol.set_array(A.polar_unwrap(img))

    def _profile(self, z):
        X, Y, valid, _ = self.sim.floor_map(self.p, z, 640, 480, walls=False)
        return A.radial_profile(X, Y, valid, self.p), (X, Y, valid)

    def view_cal(self):
        t = self.t_cal
        prof, _ = self._profile(0.0)
        for ax in t.axes.ravel():
            ax.clear()
        a1, a2 = t.axes[0]
        if prof is None:
            a1.set_title("Пол не виден")
        else:
            s = 640 / self.p.camera.width           # report in units of the selected resolution
            a1.plot(prof["r_px"] / s, prof["dist"] / 10)
            a1.set_xlabel("радиус от центра кадра, px")
            a1.set_ylabel("расстояние до точки пола, см")
            a1.set_title("Калибровка: радиус пикселя → расстояние")
            a1.grid(True)
            a2.plot(prof["dist"] / 10, prof["rad"] * s, label="вдоль радиуса")
            a2.plot(prof["dist"] / 10, prof["tan"] * s, label="вдоль кольца")
            a2.set_xlabel("расстояние, см")
            a2.set_ylabel("мм на пиксель")
            a2.set_title("Разрешение пола")
            a2.set_ylim(0, min(60, np.nanmax(prof["tan"] * s) * 1.1))
            a2.legend()
            a2.grid(True)
            self.status.showMessage(f"Слепая зона: {prof['blind']/10:.1f} см, дальний обзор: {prof['far']/10:.0f} см")
        t.canvas.draw_idle()

    def view_ball(self):
        t = self.t_ball
        ax = t.axes[0, 0]
        ax.clear()
        prof, _ = self._profile(C.BALL_D / 2)
        if prof is not None:
            s = 640 / self.p.camera.width
            bs = A.ball_size_curve(prof)
            ax.plot(bs["dist"] / 10, bs["radial"] / s, label="вдоль радиуса")
            ax.plot(bs["dist"] / 10, bs["tangential"] / s, label="вдоль кольца")
            ax.axhline(self.det_px.value(), color="r", ls="--", label="порог детекции")
            ax.set_ylim(0, 40)
            ax.set_xlabel("расстояние до мяча, см")
            ax.set_ylabel("размер мяча, px")
            ax.set_title("Видимый размер мяча 42 мм")
            ax.legend()
            ax.grid(True)
        t.canvas.draw_idle()

    def view_heat(self):
        t = self.t_heat
        ax = t.axes[0, 0]
        t.fig.clf()
        ax = t.fig.subplots(1, 1)
        t.axes = np.array([[ax]])
        W, H = 640, 480
        X, Y, valid, _ = self.sim.floor_map(self.p, C.BALL_D / 2, W, H)
        xs, ys, size = A.ball_heatmap(X, Y, valid, self.p)
        s = W / self.p.camera.width
        size = size / s
        im = ax.imshow(size, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="viridis", vmin=0, vmax=30)
        det = self.det_px.value()
        ax.contour(xs, ys, np.nan_to_num(size, nan=0), levels=[det], colors="r", linewidths=1)
        ax.plot(self.p.robot.x, self.p.robot.y, "wo")
        ax.set_aspect("equal")
        ax.set_title(f"Размер мяча, px (красная линия = порог {det:.0f} px; серое = мяч не виден)")
        ax.set_facecolor("#555")
        t.fig.colorbar(im, ax=ax, shrink=0.8, label="px")
        t.canvas.draw_idle()

    def view_map(self):
        t = self.t_map
        ax = t.axes[0, 0]
        ax.clear()
        tex = self.sim.floor.tex
        ax.imshow(tex, origin="lower", extent=(-C.TOTAL_W / 2, C.TOTAL_W / 2, -C.TOTAL_L / 2, C.TOTAL_L / 2))
        p = self.p
        from matplotlib.patches import Circle
        ax.add_patch(Circle((p.robot.x, p.robot.y), p.robot.size / 2, fill=False, ec="cyan", lw=2))
        a = np.radians(p.robot.heading_deg + 90)
        ax.arrow(p.robot.x, p.robot.y, 80 * np.cos(a), 80 * np.sin(a), color="cyan", width=8)
        ax.add_patch(Circle((p.ball_x, p.ball_y), C.BALL_D / 2, color="orange"))
        ax.set_aspect("equal")
        t.canvas.draw_idle()

    def on_map_click(self, e):
        if e.inaxes is None or e.xdata is None:
            return
        if e.button == 1:
            self.bx.setValue(float(np.clip(e.xdata, -790, 790)))
            self.by.setValue(float(np.clip(e.ydata, -1095, 1095)))
        elif e.button == 3:
            self.rx.setValue(float(np.clip(e.xdata, -900, 900)))
            self.ry.setValue(float(np.clip(e.ydata, -1200, 1200)))

    # ---------------------------------------------------------------- 3D
    def view_3d(self):
        p, pl = self.p, self.plotter
        # floor with texture
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=C.TOTAL_W, j_size=C.TOTAL_L)
        tex = pv.Texture((np.clip(self.sim.floor.tex, 0, 1) * 255).astype(np.uint8))
        pl.add_mesh(plane, texture=tex, name="floor", lighting=False)

        mesh, cols = self.sim.static_mb.build()
        pl.add_mesh(self._pv(mesh, cols), scalars="rgb", rgb=True, name="field")

        r = p.robot
        body = F.robot_body_mesh(r.x, r.y, r.heading_deg, r.size, r.body_height)
        pl.add_mesh(self._pv(body, np.tile(C.C_ROBOT, (len(body.faces), 1))), scalars="rgb", rgb=True, name="robot")
        ball = F.ball_mesh(p.ball_x, p.ball_y)
        pl.add_mesh(self._pv(ball, np.tile(C.C_BALL, (len(ball.faces), 1))), scalars="rgb", rgb=True, name="ball")

        self.sim._place_mirror(p.mirror_plane_height)
        mm = self.sim._placed_mirror.copy()
        mm.apply_transform(_rz(r.heading_deg))
        mm.apply_translation((r.x, r.y, 0))
        pl.add_mesh(pv.wrap(mm), color="#c8ccd2", smooth_shading=True, specular=0.8, name="mirror")

        # sample rays: camera -> mirror -> floor
        cam = p.camera
        cx, cy = r.x, r.y
        cam_pos = np.array([cx, cy, cam.height_above_field])
        pl.add_mesh(pv.Sphere(radius=6, center=cam_pos), color="lime", name="cam")
        d_local = self.sim.camera_rays(cam, 16, 12)
        ro, rd, hit = self.sim._reflect(p, d_local)
        end = ro + rd * np.where(rd[:, 2] < 0, -ro[:, 2] / np.minimum(rd[:, 2], -1e-9), 300.0)[:, None]
        segs = []
        for o, e in zip(ro[::3], end[::3]):
            segs += [cam_pos, o, o, e]
        if segs:
            pl.add_mesh(pv.line_segments_from_points(np.array(segs)), color="yellow", line_width=1, name="rays")
        if not getattr(self, "_cam_set", False):
            pl.camera_position = [(1500, -1800, 1500), (0, 0, 0), (0, 0, 1)]
            self._cam_set = True
        pl.render()

    @staticmethod
    def _pv(mesh, cols):
        pd = pv.wrap(mesh)
        pd.cell_data["rgb"] = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
        return pd

    # ----------------------------------------------------------- the rest
    def _update_checks(self):
        p = self.p
        lim = p.robot.size
        info = self.sim.mirror_info
        top = p.mirror_plane_height + float(info["bounds"][1][2])
        rad = max(abs(info["bounds"][0][0]), info["bounds"][1][0]) if info else 0
        rows = []
        ok = "<span style='color:#2a2'>OK</span>"
        bad = "<span style='color:#c33'><b>ПРЕВЫШЕНИЕ</b></span>"
        rows.append(f"Высота робота: верх зеркала {top:.0f} мм из {lim:.0f} — {ok if top <= lim else bad}")
        rows.append(f"Диаметр зеркала {2*rad:.0f} мм, диаметр корпуса {lim:.0f} — {ok if 2*rad <= lim else bad}")
        apex = p.camera.height_above_field + p.mirror.lens_to_mirror + float(info["bounds"][0][2])
        rows.append(f"Низшая точка зеркала над полем: {apex:.0f} мм"
                    + ("" if apex > p.camera.height_above_field else " — <b>ниже объектива!</b>"))
        try:
            _, _, v, dist = self.sim.floor_map(p, C.BALL_D / 2, 160, 120, walls=False)
            near = float(np.nanmin(dist[v])) if v.any() else float("nan")
            touch = lim / 2 + C.BALL_D / 2
            good = near <= touch + 5
            rows.append(f"Ближайший видимый мяч: {near:.0f} мм от оси (касание корпуса — {touch:.0f}) — "
                        + (ok if good else "<span style='color:#c33'><b>слепая зона шире корпуса</b></span>"))
        except Exception:  # noqa: BLE001
            pass
        self.checks.setText("<br>".join(rows))

    def save_png(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить кадр", "frame.png", "PNG (*.png)")
        if path and "img" in self._last:
            from PIL import Image
            Image.fromarray(self._last["img"]).save(path)


def _rz(deg):
    a = np.radians(deg)
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return m


def _install_excepthook():
    import traceback
    from pathlib import Path

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        try:
            Path(__file__).resolve().parent.parent.joinpath("error.log").write_text(text, encoding="utf-8")
        except OSError:
            pass
        sys.stderr.write(text)
        QtWidgets.QMessageBox.critical(None, "Ошибка", text[-1500:])

    sys.excepthook = hook


def main():
    app = QtWidgets.QApplication(sys.argv)
    _install_excepthook()
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
