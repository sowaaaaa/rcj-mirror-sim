"""Ray tracer: camera -> mirror (one specular bounce) -> field scene."""
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from . import config as C
from . import field as F
from .mirror import load_mirror

LIGHT = np.array([0.3, 0.2, 0.93])
LIGHT /= np.linalg.norm(LIGHT)
EPS = 1e-3


def _intersect(inter, o, d):
    """First hit per ray: returns (t, tri_index) with t = inf on miss."""
    n = len(o)
    t = np.full(n, np.inf)
    tri = np.full(n, -1)
    if inter is None or n == 0:
        return t, tri
    idx_tri, idx_ray, loc = inter.intersects_id(o, d, multiple_hits=False, return_locations=True)
    t[idx_ray] = np.linalg.norm(loc - o[idx_ray], axis=1)
    tri[idx_ray] = idx_tri
    return t, tri


def _corner_normals(mesh, angle_deg=30.0):
    """Per-face-corner smooth normals; neighbours further than angle_deg are ignored (sharp edges)."""
    fn = mesh.face_normals
    area = mesh.area_faces
    vf = mesh.vertex_faces[mesh.faces]                  # (F, 3, maxdeg), -1 padded
    valid = vf >= 0
    nb = fn[np.where(valid, vf, 0)]                     # (F, 3, maxdeg, 3)
    close = (nb * fn[:, None, None, :]).sum(-1) > np.cos(np.radians(angle_deg))
    w = np.where(valid & close, area[np.where(valid, vf, 0)], 0.0)
    cn = (nb * w[..., None]).sum(2)
    ln = np.linalg.norm(cn, axis=-1, keepdims=True)
    return np.where(ln > 1e-12, cn / np.maximum(ln, 1e-12), fn[:, None, :])


class Simulator:
    def __init__(self):
        self.field = F.load_field()
        self.field_occluders = F.occluders(*self.field)
        self.field_top, self.field_top_extent = F.top_view(*self.field)
        self._mirror_key = None
        self._mirror_placed_key = None
        self.mirror_mesh = None
        self.mirror_info = {}
        self.mirror_inter = None

    # -- mirror ---------------------------------------------------------
    def set_mirror(self, step_path="", scale=1.0, flip=False, ref_plane_z=None, profile=None):
        key = (step_path, scale, flip, ref_plane_z, profile if not step_path else None)
        if key == self._mirror_key:
            return
        self.mirror_mesh, self.mirror_info = load_mirror(step_path, scale, flip, ref_plane_z, profile)
        self._mirror_key = key
        self._mirror_placed_key = None

    def _place_mirror(self, plane_h):
        """Mirror in the robot frame: shifted so its first plane is at plane_h."""
        if self._mirror_placed_key == plane_h:
            return
        m = self.mirror_mesh.copy()
        m.apply_translation((0, 0, plane_h))
        self._placed_mirror = m
        self._corner_normals = _corner_normals(m)
        self.mirror_inter = RayMeshIntersector(m)
        self._mirror_placed_key = plane_h

    # -- scene ----------------------------------------------------------
    def _scene(self, p, field, ball=True):
        """field = (mesh, colours) of the field part to include, or None."""
        mb = F.MeshBuilder()
        if field is not None:
            mb.add(field[0].vertices, field[0].faces, field[1])
        if ball:
            mb.add_trimesh(F.ball_mesh(p.ball_x, p.ball_y), C.C_BALL)
        r = p.robot
        mb.add_trimesh(F.robot_body_mesh(r.x, r.y, r.heading_deg, r.size, r.body_height), C.C_ROBOT)
        mesh, cols = mb.build()
        return RayMeshIntersector(mesh), mesh, cols

    def _shade_scene(self, o, d, inter, mesh, cols, bg=C.C_BACKGROUND):
        """Colour of the first scene hit along each ray."""
        t_mesh, tri = _intersect(inter, o, d)
        out = np.empty((len(o), 3))
        out[:] = bg
        hit = np.isfinite(t_mesh)
        if hit.any():
            nrm = mesh.face_normals[tri[hit]]
            lam = np.abs(nrm @ LIGHT)
            out[hit] = cols[tri[hit]] * (0.55 + 0.45 * lam)[:, None]
        return out

    # -- camera rays ------------------------------------------------------
    @staticmethod
    def camera_rays(cam, width=None, height=None):
        W, H = width or cam.width, height or cam.height
        f = (W / 2) / np.tan(np.radians(cam.hfov_deg) / 2)
        i, j = np.meshgrid(np.arange(W), np.arange(H))
        u, v = i + 0.5 - W / 2, j + 0.5 - H / 2
        # camera looks up (+Z): image-right = -X, image-up = +Y (robot forward)
        d = np.stack([-u / f, -v / f, np.ones_like(u)], axis=-1).reshape(-1, 3)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d

    def _rot(self, deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _reflect(self, p, d_local):
        """Camera rays -> reflected rays in WORLD frame. Returns (o, d, hit_mask, refl)."""
        self._place_mirror(p.mirror_plane_height)
        n = len(d_local)
        o_local = np.zeros((n, 3))
        o_local[:, 2] = p.camera.height_above_field
        t, tri = _intersect(self.mirror_inter, o_local, d_local)
        hit = np.isfinite(t)
        hp = o_local[hit] + d_local[hit] * t[hit][:, None]
        m = self._placed_mirror
        bary = trimesh.triangles.points_to_barycentric(m.triangles[tri[hit]], hp)
        nrm = (self._corner_normals[tri[hit]] * bary[..., None]).sum(1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        flip = (nrm * d_local[hit]).sum(1) > 0
        nrm[flip] *= -1
        rd = d_local[hit] - 2 * (d_local[hit] * nrm).sum(1, keepdims=True) * nrm
        ro = hp + nrm * EPS
        R = self._rot(p.robot.heading_deg)
        ro_w = ro @ R.T + np.array([p.robot.x, p.robot.y, 0.0])
        rd_w = rd @ R.T
        return ro_w, rd_w, hit

    # -- public -----------------------------------------------------------
    def render(self, p, width=None, height=None):
        W, H = width or p.camera.width, height or p.camera.height
        d_local = self.camera_rays(p.camera, W, H)
        ro, rd, hit = self._reflect(p, d_local)
        inter, mesh, cols = self._scene(p, self.field)
        img = np.zeros((W * H, 3))
        col = self._shade_scene(ro, rd, inter, mesh, cols)
        img[hit] = col * p.mirror.reflectance
        return (np.clip(img, 0, 1).reshape(H, W, 3) * 255).astype(np.uint8)

    def floor_map(self, p, plane_z=0.0, width=None, height=None, walls=True):
        """For each pixel: world (x, y) where the reflected ray crosses z = plane_z.

        valid = ray hits the mirror and goes downward; with walls=True also not blocked by a
        wall/goal and inside the field (walls=False = unbounded floor, for calibration curves).
        Returns X, Y, valid, dist (distance from robot axis), each shaped (H, W).
        """
        W, H = width or p.camera.width, height or p.camera.height
        d_local = self.camera_rays(p.camera, W, H)
        ro, rd, hit = self._reflect(p, d_local)
        # occluders: the robot's own body always; walls/goals only when walls=True
        inter, mesh, cols = self._scene(p, self.field_occluders if walls else None, ball=False)
        n = len(ro)
        X = np.full(W * H, np.nan)
        Y = np.full(W * H, np.nan)
        valid = np.zeros(W * H, bool)
        down = rd[:, 2] < -1e-9
        t = np.full(len(ro), np.inf)
        t[down] = (plane_z - ro[down, 2]) / rd[down, 2]
        t_wall, _ = _intersect(inter, ro, rd)
        ok = down & (t > 0) & (t < t_wall)
        hp = ro + rd * np.where(ok, t, 0)[:, None]
        okh = ok
        if walls:
            okh = ok & (np.abs(hp[:, 0]) <= C.TOTAL_W / 2) & (np.abs(hp[:, 1]) <= C.TOTAL_L / 2)
        idx = np.flatnonzero(hit)
        X[idx[okh]] = hp[okh, 0]
        Y[idx[okh]] = hp[okh, 1]
        valid[idx[okh]] = True
        dist = np.hypot(X - p.robot.x, Y - p.robot.y)
        return (X.reshape(H, W), Y.reshape(H, W), valid.reshape(H, W), dist.reshape(H, W))
