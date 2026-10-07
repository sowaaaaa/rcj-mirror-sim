"""Profile editor: drag control points of the mirror cross-section and see the rays."""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from .mirror import curve_from_points, design_profile, resample_points

RIM_W = 6.0


class ProfileEditor(QtWidgets.QWidget):
    """Emits `changed` after any edit. Points are (r, z) in the mirror frame, z=0 at the first plane."""

    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.pts = [(0.0, -20.0), (10.0, -14.0), (20.0, -8.0), (30.0, 0.0)]
        self.smooth = True
        self.geom = dict(d=70.0, h_m=115.0, h_cam=45.0, half_fov=26.0)
        self._drag = None
        self._lock = False

        self.fig = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.subplots()
        self.canvas.mpl_connect("button_press_event", self._press)
        self.canvas.mpl_connect("motion_notify_event", self._move)
        self.canvas.mpl_connect("button_release_event", self._release)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["r, мм", "z, мм"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.cellChanged.connect(self._cell_edited)
        self.table.setMaximumWidth(200)

        self.chk_smooth = QtWidgets.QCheckBox("Плавная кривая")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.toggled.connect(self._smooth_toggled)
        b_add = QtWidgets.QPushButton("+ точка")
        b_add.clicked.connect(self.add_point)
        b_del = QtWidgets.QPushButton("− точка")
        b_del.clicked.connect(self.delete_point)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)

        # computed profile
        box = QtWidgets.QGroupBox("Рассчитать профиль (постоянные мм/пиксель)")
        f = QtWidgets.QFormLayout(box)
        self.d_near = self._spin(f, "Ближняя точка пола", 50, 600, 110, 10)
        self.d_far = self._spin(f, "Дальняя точка пола", 300, 4000, 1500, 100)
        self.th_in = self._spin(f, "Угол внутр. края, °", 1, 20, 6, 0.5)
        self.th_max = self._spin(f, "Угол внешн. края, °", 10, 60, 26, 0.5)
        self.gamma = self._spin(f, "Закон γ (1 = линейный, >1 дальше чётче)", 0.5, 3.0, 1.0, 0.1)
        self.n_pts = QtWidgets.QSpinBox()
        self.n_pts.setRange(4, 30)
        self.n_pts.setValue(10)
        f.addRow("Контрольных точек", self.n_pts)
        b_calc = QtWidgets.QPushButton("Рассчитать и загрузить")
        b_calc.clicked.connect(self.calculate)
        f.addRow(b_calc)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(b_add)
        row.addWidget(b_del)
        side.addLayout(row)
        side.addWidget(self.chk_smooth)
        side.addWidget(self.info)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(side)
        lay.addWidget(box)
        self._fill_table()
        self.redraw()

    @staticmethod
    def _spin(form, label, lo, hi, val, step):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        s.setDecimals(1)
        form.addRow(label, s)
        return s

    # -------------------------------------------------------------- data
    def points(self):
        return tuple((round(r, 3), round(z, 3)) for r, z in self.pts)

    def set_curve(self, r, z, n=None, emit=False):
        n = n or self.n_pts.value()
        self.pts = resample_points(np.asarray(r), np.asarray(z), n)
        self._fill_table()
        self.redraw()
        if emit:
            self.changed.emit()

    def set_geometry(self, d, h_m, h_cam, half_fov):
        self.geom = dict(d=d, h_m=h_m, h_cam=h_cam, half_fov=half_fov)
        self.redraw()

    def calculate(self):
        g = self.geom
        try:
            r, z = design_profile(g["h_cam"], g["d"], self.d_near.value(), self.d_far.value(),
                                  self.th_in.value(), self.th_max.value(), gamma=self.gamma.value())
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Расчёт профиля", str(e))
            return
        self.set_curve(r, z, emit=True)

    # ------------------------------------------------------------- table
    def _fill_table(self):
        self._lock = True
        self.table.setRowCount(len(self.pts))
        for i, (r, z) in enumerate(self.pts):
            for j, v in enumerate((r, z)):
                it = QtWidgets.QTableWidgetItem(f"{v:.2f}")
                if (j == 0 and i == 0) or (j == 1 and i == len(self.pts) - 1):
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)     # r=0 at apex, z=0 at rim
                self.table.setItem(i, j, it)
        self._lock = False

    def _cell_edited(self, i, j):
        if self._lock:
            return
        try:
            v = float(self.table.item(i, j).text().replace(",", "."))
        except ValueError:
            self._fill_table()
            return
        r, z = self.pts[i]
        if j == 0:
            lo = self.pts[i - 1][0] + 0.1 if i > 0 else 0.0
            hi = self.pts[i + 1][0] - 0.1 if i < len(self.pts) - 1 else 200.0
            r = float(np.clip(v, lo, hi))
        else:
            z = v
        self.pts[i] = (r, z)
        self._fill_table()
        self.redraw()
        self.changed.emit()

    def _smooth_toggled(self, on):
        self.smooth = on
        self.redraw()
        self.changed.emit()

    def add_point(self):
        """Insert a point in the middle of the longest gap."""
        gaps = [self.pts[i + 1][0] - self.pts[i][0] for i in range(len(self.pts) - 1)]
        i = int(np.argmax(gaps))
        (r0, z0), (r1, z1) = self.pts[i], self.pts[i + 1]
        r, z = curve_from_points(self.pts, self.smooth)
        rm = (r0 + r1) / 2
        self.pts.insert(i + 1, (rm, float(np.interp(rm, r, z))))
        self._fill_table()
        self.redraw()
        self.changed.emit()

    def delete_point(self):
        i = self.table.currentRow()
        if len(self.pts) <= 3 or i <= 0 or i >= len(self.pts) - 1:
            return                                           # keep apex, rim and at least one inner point
        del self.pts[i]
        self._fill_table()
        self.redraw()
        self.changed.emit()

    # -------------------------------------------------------------- mouse
    def _nearest(self, event):
        if event.inaxes is not self.ax:
            return None
        tr = self.ax.transData.transform
        best, bd = None, 14.0
        for i, p in enumerate(self.pts):
            d = np.hypot(*(tr(p) - np.array([event.x, event.y])))
            if d < bd:
                best, bd = i, d
        return best

    def _press(self, e):
        if e.button == 1:
            self._drag = self._nearest(e)

    def _move(self, e):
        if self._drag is None or e.inaxes is not self.ax or e.xdata is None:
            return
        i, n = self._drag, len(self.pts)
        r, z = e.xdata, e.ydata
        if i == 0:
            r = 0.0
        else:
            lo = self.pts[i - 1][0] + 0.2
            hi = self.pts[i + 1][0] - 0.2 if i < n - 1 else 200.0
            r = float(np.clip(r, lo, hi))
        if i == n - 1:
            z = 0.0
        self.pts[i] = (float(r), float(z))
        self.redraw()

    def _release(self, e):
        if self._drag is not None:
            self._drag = None
            self._fill_table()
            self.changed.emit()

    # ------------------------------------------------------------- drawing
    def redraw(self):
        ax = self.ax
        ax.clear()
        g = self.geom
        r, z = curve_from_points(self.pts, self.smooth)
        R = r[-1]
        full_r = np.concatenate([r, [R + RIM_W]])
        full_z = np.concatenate([z, [0.0]])
        ax.plot(full_r, full_z, "-", color="#556", lw=2)
        ax.plot(-full_r, full_z, "-", color="#aab", lw=1)
        ax.plot(*zip(*self.pts), "o", color="#d33", ms=7)
        ax.plot([0], [-g["d"]], "s", color="green", ms=7)
        ax.text(1, -g["d"] - 3, "объектив", color="green", fontsize=8)
        ax.axhline(0, color="#999", lw=0.6, ls=":")

        # rays in the section plane
        hf = np.radians(g["half_fov"])
        for th in np.linspace(0.0, hf, 12)[1:]:
            end = self._trace(th, full_r, full_z, g)
            if end is None:
                continue
            (hx, hz), (fx, fz), dist = end
            ax.plot([0, hx], [-g["d"], hz], color="#e0b000", lw=0.7)
            ax.plot([hx, fx], [hz, fz], color="#e07000", lw=0.7)
            if dist is not None:
                ax.text(fx, fz, f"{dist/1000:.2f} м", fontsize=6, color="#a40")
        lim = max(R * 1.8, 40)
        ax.set_xlim(-R * 0.4, lim)
        ax.set_ylim(-g["d"] - 12, max(15, -z.min() + 5))
        ax.set_aspect("equal")
        ax.set_xlabel("r, мм (ось слева)")
        ax.set_ylabel("z, мм (0 = 1-я плоскость)")
        ax.grid(True, alpha=0.3)
        self.info.setText(f"Радиус {R:.1f} мм, глубина {-z.min():.1f} мм. "
                          f"Верх зеркала над полем ≈ {g['h_m'] + 3:.0f} мм.")
        self.canvas.draw_idle()

    def _trace(self, th, rr, zz, g):
        """One ray from the lens at angle th to the curve and on to the floor (section plane)."""
        o = np.array([0.0, -g["d"]])
        dvec = np.array([np.sin(th), np.cos(th)])
        best_t, best_i = np.inf, -1
        for i in range(len(rr) - 1):
            a, b = np.array([rr[i], zz[i]]), np.array([rr[i + 1], zz[i + 1]])
            e = b - a
            m = np.array([[dvec[0], -e[0]], [dvec[1], -e[1]]])
            if abs(np.linalg.det(m)) < 1e-12:
                continue
            t, u = np.linalg.solve(m, a - o)
            if t > 0 and 0 <= u <= 1 and t < best_t:
                best_t, best_i = t, i
        if best_i < 0:
            return None
        P = o + best_t * dvec
        e = np.array([rr[best_i + 1] - rr[best_i], zz[best_i + 1] - zz[best_i]])
        n = np.array([-e[1], e[0]])
        n /= np.linalg.norm(n)
        if n @ dvec > 0:
            n = -n
        out = dvec - 2 * (dvec @ n) * n
        floor_z = -g["h_m"]
        if out[1] < -1e-6:
            tf = (floor_z - P[1]) / out[1]
            F = P + out * tf
            dist = F[0]
            seg = P + out * min(tf, 40.0 / max(np.hypot(*out), 1e-9))
            return (P[0], P[1]), (seg[0], seg[1]), dist
        seg = P + out * 40.0
        return (P[0], P[1]), (seg[0], seg[1]), None
