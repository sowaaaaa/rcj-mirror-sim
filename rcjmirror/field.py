"""Field geometry: analytic floor texture (lines) + triangle meshes for solids."""
from pathlib import Path

import numpy as np
import trimesh

from . import config as C

TEX_RES = 2.0                       # mm per texture pixel
_TEX_PATH = Path(__file__).with_name("_floor_tex.npy")


# --------------------------------------------------------------------------
# Floor texture
# --------------------------------------------------------------------------
def _rrect_mask(X, Y, x0, x1, y0, y1, r):
    """Rounded-rectangle mask (all four corners rounded with radius r)."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 - r, (y1 - y0) / 2 - r
    qx = np.abs(X - cx) - hx
    qy = np.abs(Y - cy) - hy
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    return d <= 0


def build_floor_texture():
    """Render the field markings at 1 mm and box-filter to TEX_RES. Returns HxWx3."""
    hw, hl = C.TOTAL_W / 2, C.TOTAL_L / 2
    xs = np.arange(-hw + 0.5, hw, 1.0, dtype=np.float32)
    ys = np.arange(-hl + 0.5, hl, 1.0, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)               # shape (L, W)

    pw, pl, lw = C.PLAY_W / 2, C.PLAY_L / 2, C.LINE_W
    white = (np.abs(X) <= pw) & (np.abs(Y) <= pl) & ~((np.abs(X) <= pw - lw) & (np.abs(Y) <= pl - lw))
    black = np.zeros_like(white)

    for s in (1, -1):                         # penalty areas at both ends
        ys_ = s * Y
        outer = _rrect_mask(X, ys_, -C.PEN_W / 2, C.PEN_W / 2, pl - C.PEN_D, pl + 300, C.PEN_R)
        inner = _rrect_mask(X, ys_, -C.PEN_W / 2 + lw, C.PEN_W / 2 - lw,
                            pl - C.PEN_D + lw, pl + 300, C.PEN_R - lw)
        white |= outer & ~inner & (ys_ <= pl)

    r = np.hypot(X, Y)
    black |= np.abs(r - C.CENTER_CIRCLE_D / 2) <= 2.0
    spots = [(0, 0)] + [(sx * C.SPOT_X, sy * C.SPOT_Y) for sx in (1, -1) for sy in (1, -1)]
    for sx, sy in spots:
        black |= np.hypot(X - sx, Y - sy) <= C.SPOT_D / 2

    img = np.empty(X.shape + (3,), np.float32)
    img[:] = C.C_CARPET
    img[white] = C.C_LINE
    img[black & ~white] = C.C_MARK
    k = int(TEX_RES)
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    img = img[:h, :w].reshape(h // k, k, w // k, k, 3).mean(axis=(1, 3))
    return img


def get_floor_texture():
    if _TEX_PATH.exists():
        return np.load(_TEX_PATH)
    tex = build_floor_texture()
    np.save(_TEX_PATH, tex)
    return tex


class FloorTexture:
    def __init__(self):
        self.tex = get_floor_texture()
        self.h, self.w = self.tex.shape[:2]

    def sample(self, x, y):
        """Bilinear colour lookup at floor coordinates (mm)."""
        u = (x + C.TOTAL_W / 2) / TEX_RES - 0.5
        v = (y + C.TOTAL_L / 2) / TEX_RES - 0.5
        u = np.clip(u, 0, self.w - 1.001)
        v = np.clip(v, 0, self.h - 1.001)
        u0, v0 = u.astype(int), v.astype(int)
        fu, fv = (u - u0)[:, None], (v - v0)[:, None]
        t = self.tex
        top = t[v0, u0] * (1 - fu) + t[v0, u0 + 1] * fu
        bot = t[v0 + 1, u0] * (1 - fu) + t[v0 + 1, u0 + 1] * fu
        return top * (1 - fv) + bot * fv


# --------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------
class MeshBuilder:
    def __init__(self):
        self.v, self.f, self.c = [], [], []
        self.n = 0

    def add(self, verts, faces, colors):
        verts = np.asarray(verts, float)
        faces = np.asarray(faces, int)
        self.v.append(verts)
        self.f.append(faces + self.n)
        self.c.append(np.broadcast_to(np.asarray(colors, float), (len(faces), 3)).copy())
        self.n += len(verts)

    def add_trimesh(self, m, color):
        self.add(m.vertices, m.faces, color)

    def add_box(self, lo, hi, color_of_normal):
        b = trimesh.creation.box(extents=np.subtract(hi, lo))
        b.apply_translation((np.add(lo, hi)) / 2)
        cols = np.array([color_of_normal(n) for n in b.face_normals])
        self.add(b.vertices, b.faces, cols)

    def build(self):
        if not self.v:
            return None, np.zeros((0, 3))
        m = trimesh.Trimesh(np.vstack(self.v), np.vstack(self.f), process=False)
        return m, np.vstack(self.c)


def _quad(p0, p1, p2, p3):
    return np.array([p0, p1, p2, p3]), np.array([[0, 1, 2], [0, 2, 3]])


def build_field_mesh():
    """Walls, wedges and goals (static)."""
    mb = MeshBuilder()
    hw, hl, H = C.TOTAL_W / 2, C.TOTAL_L / 2, C.WALL_H
    t = C.WALL_T
    black = lambda n: C.C_WALL
    # walls (inner face at +-hw / +-hl)
    mb.add_box((-hw - t, -hl - t, 0), (-hw, hl + t, H), black)
    mb.add_box((hw, -hl - t, 0), (hw + t, hl + t, H), black)
    mb.add_box((-hw, -hl - t, 0), (hw, -hl, H), black)
    mb.add_box((-hw, hl, 0), (hw, hl + t, H), black)

    # wedges: 100 mm wide ramps rising 20 mm towards the wall, none behind goals
    b, r = C.WEDGE_BASE, C.WEDGE_RISE
    ix, iy = hw - b, hl - b                       # inner edge of ramp (z=0)
    c = C.C_CARPET
    for sx in (1, -1):                            # long sides
        mb.add(*_quad((sx * ix, -iy, 0), (sx * hw, -hl, r), (sx * hw, hl, r), (sx * ix, iy, 0)), c)
    gx = C.GOAL_W / 2 + C.GOAL_T                  # ramp stops at the goal sides
    for sy in (1, -1):                            # short sides, left and right of goal
        for sx in (1, -1):
            x_in_a, x_in_b = sx * gx, sx * ix
            x_out_a, x_out_b = sx * gx, sx * hw
            mb.add(*_quad((x_in_a, sy * iy, 0), (x_in_b, sy * iy, 0),
                          (x_out_b, sy * hl, r), (x_out_a, sy * hl, r)), c)

    # goals: interior coloured, exterior black
    for sy, col in ((1, C.C_BLUE), (-1, C.C_YELLOW)):
        y0 = C.PLAY_L / 2 - C.LINE_W / 2         # posts sit on the white line
        y1 = y0 + C.GOAL_D
        gw, gt, gh = C.GOAL_W / 2, C.GOAL_T, C.GOAL_H

        def mk(lo_y, hi_y, interior_dir):
            def f(n):
                return col if np.dot(n, interior_dir) > 0.5 else C.C_WALL
            return f

        ys = sorted((sy * y0, sy * y1))
        yb = sorted((sy * y1, sy * (y1 + gt)))
        # back wall: interior normal faces the field (-sy in y)
        mb.add_box((-gw - gt, yb[0], 0), (gw + gt, yb[1], gh + gt), mk(0, 0, np.array([0, -sy, 0])))
        # side panels: interior faces +-x towards centre
        mb.add_box((-gw - gt, ys[0], 0), (-gw, ys[1], gh + gt), mk(0, 0, np.array([1, 0, 0])))
        mb.add_box((gw, ys[0], 0), (gw + gt, ys[1], gh + gt), mk(0, 0, np.array([-1, 0, 0])))
        # roof: interior faces down
        mb.add_box((-gw - gt, ys[0], gh), (gw + gt, ys[1], gh + gt), mk(0, 0, np.array([0, 0, -1])))
    return mb


def ball_mesh(x, y, diameter=C.BALL_D):
    s = trimesh.creation.icosphere(subdivisions=3, radius=diameter / 2)
    s.apply_translation((x, y, diameter / 2))
    return s


def robot_body_mesh(x, y, heading_deg, size, body_h):
    cyl = trimesh.creation.cylinder(radius=size / 2, height=body_h, sections=64)
    cyl.apply_translation((0, 0, body_h / 2))
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.radians(heading_deg), [0, 0, 1]))
    cyl.apply_translation((x, y, 0))
    return cyl
