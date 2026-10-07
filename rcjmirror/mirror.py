"""Mirror loading (STEP via OpenCASCADE) and normalisation.

Normalised mirror frame: rotation axis = +Z through the origin, apex pointing
down (-Z, towards the camera), and z = 0 on the mirror's first (nearest to the
camera) flat plane.  In the scene the frame is shifted up by the lens-to-mirror
distance.
"""
import numpy as np
import trimesh


def load_step_mesh(path, linear_deflection=0.01, angular_deflection=0.05):
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"cannot read STEP file: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    verts, faces, n0 = [], [], 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        to_face = getattr(TopoDS, "Face_s", None) or TopoDS.Face   # OCP < 8 / OCP 8
        face = to_face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            pts = []
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                pts.append((p.X(), p.Y(), p.Z()))
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                t = (a - 1, c - 1, b - 1) if rev else (a - 1, b - 1, c - 1)
                faces.append([n0 + t[0], n0 + t[1], n0 + t[2]])
            verts.extend(pts)
            n0 += len(pts)
        exp.Next()
    if not faces:
        raise ValueError("STEP file contains no surfaces")
    m = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    m.update_faces(m.nondegenerate_faces())          # zero-area slivers from the tessellator
    return m


def default_cone(radius=30.0, alpha_deg=35.0, rim=3.0):
    """Stand-in mirror: cone, apex down, flat rim ring. Used until a STEP is loaded."""
    depth = radius * np.tan(np.radians(alpha_deg))
    profile = np.array([(0, -depth), (radius, 0), (radius + 6, 0), (radius + 6, rim), (0, rim)])
    m = trimesh.creation.revolve(profile, sections=128)
    return m


PRESETS = {
    # name: (default radius mm, default depth mm, short description)
    "Конус": (30.0, 21.0, "прямой конус, вершина вниз"),
    "Конус 40°": (30.0, 25.0, "угол к горизонту 40°, обзор ближе к горизонту"),
    "Гиперболоид": (30.0, 20.0, "выпуклая гипербола (классика омнивизора)"),
    "Параболоид": (30.0, 20.0, "выпуклая парабола"),
    "Сфера": (30.0, 20.0, "сферическая выпуклая шапка"),
}


def profile_curve(name, radius, depth, n=60):
    """Mirror surface z(r) for r in [0, radius]; z=0 at the rim, z=-depth at the apex."""
    r = np.linspace(0.0, radius, n)
    u = r / radius
    if name.startswith("Конус"):
        z = -depth * (1 - u)
    elif name == "Параболоид":
        z = -depth * (1 - u ** 2)
    elif name == "Гиперболоид":
        a = radius * 0.6
        g = np.sqrt(1 + (r / a) ** 2) - 1
        z = -depth * (1 - g / g[-1])
    elif name == "Сфера":
        rho = (radius ** 2 + depth ** 2) / (2 * depth)
        z = -depth + (rho - np.sqrt(np.maximum(rho ** 2 - r ** 2, 0)))
    else:
        raise ValueError(name)
    return r, z


def _revolve_with_rim(r, z, rim=3.0, rim_w=6.0):
    radius = r[-1]
    pts = list(zip(r, z)) + [(radius + rim_w, 0.0), (radius + rim_w, rim), (0.0, rim)]
    return trimesh.creation.revolve(np.array(pts), sections=360)


def preset_mirror(name, radius, depth, rim=3.0, rim_w=6.0):
    r, z = profile_curve(name, radius, depth)
    return _revolve_with_rim(r, z, rim, rim_w)


def curve_from_points(pts, smooth=True, n=120):
    """Control points [(r, z), ...] (r ascending, first r=0, last z=0) -> dense curve."""
    from scipy.interpolate import PchipInterpolator
    pts = np.array(sorted(pts), float)
    rr = np.linspace(pts[0, 0], pts[-1, 0], n)
    if smooth and len(pts) >= 3:
        zz = PchipInterpolator(pts[:, 0], pts[:, 1])(rr)
    else:
        zz = np.interp(rr, pts[:, 0], pts[:, 1])
    return rr, zz


def resample_points(r, z, n):
    """Pick n control points along a dense curve (uniform in r)."""
    rr = np.linspace(r[0], r[-1], n)
    zz = np.interp(rr, r, z)
    zz[-1] = 0.0
    return list(zip(rr.tolist(), zz.tolist()))


def design_profile(h_cam, d, d_near, d_far, theta_in_deg, theta_max_deg, n=80, gamma=1.0):
    """Computed mirror: floor distance D grows with image radius by a power law.

    gamma = 1: D linear in tan(theta) (constant mm/px along the radius).
    gamma > 1: more pixels for the far floor, fewer near; gamma < 1: the opposite.

    Camera at the origin looking up, the mirror's first plane is d above the lens, the
    field is h_cam below the lens.  Image ray angle theta (from the axis) in
    [theta_in, theta_max] is mapped to the floor distance D in [d_near, d_far] linearly in
    tan(theta).  The surface is integrated from the rim inwards so that the normal
    bisects the incoming and outgoing rays.  Inside theta_in the surface continues as a cone.
    Returns dense (r, z) in the mirror frame (z = 0 on the first plane).
    """
    t_in, t_max = np.radians(theta_in_deg), np.radians(theta_max_deg)
    zf = -h_cam

    def D_of(th):
        u = (np.tan(th) - np.tan(t_in)) / (np.tan(t_max) - np.tan(t_in))
        return d_near * (1 + u * ((d_far / d_near) ** gamma - 1)) ** (1 / gamma)

    def f(th, s):
        i = np.array([np.sin(th), np.cos(th)])
        ip = np.array([np.cos(th), -np.sin(th)])
        P = s * i
        o = np.array([D_of(th) - P[0], zf - P[1]])
        o /= np.linalg.norm(o)
        nrm = o - i
        return -s * (ip @ nrm) / (i @ nrm)

    ths = np.linspace(t_max, t_in, n)
    s = np.empty(n)
    s[0] = d / np.cos(t_max)
    for k in range(n - 1):
        h = ths[k + 1] - ths[k]
        t0, s0 = ths[k], s[k]
        k1 = f(t0, s0)
        k2 = f(t0 + h / 2, s0 + h / 2 * k1)
        k3 = f(t0 + h / 2, s0 + h / 2 * k2)
        k4 = f(t0 + h, s0 + h * k3)
        s[k + 1] = s0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    r = s * np.sin(ths)
    z = s * np.cos(ths) - d
    # continue as a straight cone to the axis
    slope = (z[-1] - z[-2]) / (r[-1] - r[-2])
    r = np.concatenate([[0.0], r[::-1]])
    z = np.concatenate([[z[-1] - slope * r[1]], z[::-1]])
    z[-1] = 0.0
    return r, z


def _radius_at(mesh, z_lo, z_hi):
    v = mesh.vertices
    sel = v[(v[:, 2] >= z_lo) & (v[:, 2] <= z_hi)]
    return np.hypot(sel[:, 0], sel[:, 1]).max() if len(sel) else 0.0


def normalise_mirror(mesh, scale=1.0, flip=False, ref_plane_z=None):
    """Return (mesh_in_mirror_frame, info dict). Axis assumed to be Z of the file."""
    m = mesh.copy()
    m.apply_scale(scale)
    # axis of revolution = the extent that differs from the other two (SolidWorks exports Y-up)
    e = m.extents
    axis = 2
    for k in (0, 1):
        a, b = [i for i in range(3) if i != k]
        if abs(e[a] - e[b]) <= 0.01 * max(e[a], e[b]) and abs(e[k] - e[a]) > 0.01 * e[a]:
            axis = k
    if axis != 2:
        to_z = np.eye(4)
        to_z[:3, :3] = np.roll(np.eye(3), 2 - axis, axis=0)   # cyclic permutation, keeps handedness
        m.apply_transform(to_z)
    if flip:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    lo, hi = m.bounds
    m.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0))

    # auto-orient: the wide flat part must be on top
    zr = hi[2] - lo[2]
    r_bot = _radius_at(m, lo[2], lo[2] + 0.1 * zr)
    r_top = _radius_at(m, hi[2] - 0.1 * zr, hi[2])
    auto_flipped = False
    if r_bot > r_top * 1.05 and not flip:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        auto_flipped = True

    if ref_plane_z is None:
        n = m.face_normals
        down = n[:, 2] < -0.999
        area = m.area_faces
        big = down & (area > 0.0)
        if big.any():
            zc = m.triangles_center[:, 2]
            # cluster by z (0.05 mm), keep clusters with >= 2 % of total area
            zs = np.round(zc[big] / 0.05) * 0.05
            best = None
            for z in np.unique(zs):
                a = area[big][zs == z].sum()
                if a >= 0.02 * area.sum() and (best is None or z < best):
                    best = z
            ref_plane_z = best
        if ref_plane_z is None:
            ref_plane_z = m.bounds[1][2]
    m.apply_translation((0, 0, -ref_plane_z))
    info = dict(auto_flipped=auto_flipped, ref_plane_z=float(ref_plane_z), axis="XYZ"[axis],
                bounds=m.bounds.copy(), size=m.extents.copy())
    return m, info


def load_mirror(step_path="", scale=1.0, flip=False, ref_plane_z=None, profile=None):
    """profile = (preset name, radius, depth) is used when no STEP path is given."""
    if not step_path and profile:
        if profile[0] == "points":                    # ("points", control points, smooth)
            raw = _revolve_with_rim(*curve_from_points([tuple(q) for q in profile[1]], profile[2]))
        else:
            raw = preset_mirror(*profile)
        return raw, dict(auto_flipped=False, ref_plane_z=0.0, bounds=raw.bounds.copy(), size=raw.extents.copy())
    if step_path:
        raw = load_step_mesh(step_path)
    else:
        raw = default_cone()
        return normalise_mirror(raw, 1.0, False, 0.0)[0], dict(
            auto_flipped=False, ref_plane_z=0.0, bounds=raw.bounds.copy(), size=raw.extents.copy())
    return normalise_mirror(raw, scale, flip, ref_plane_z)
