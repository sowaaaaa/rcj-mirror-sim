"""Field geometry: the RCJ field STEP model with face colours, plus ball and robot meshes."""
from pathlib import Path

import numpy as np
import trimesh

from . import config as C

FIELD_STEP = Path(__file__).resolve().parent.parent / "ref" / "SoccerField_202605.step"
# STEP frame: origin at the outer corner, long side along +X with the blue goal at small X,
# carpet at z = 20 on a base plate.  Sim frame: centre of the carpet, blue goal towards +Y.
_STEP_CENTRE = np.array([1235.0, 930.0, 20.0])
_DEFAULT_COLOUR = (0.63, 0.63, 0.63)
FLOOR_LEVEL = 2.0          # faces entirely below this are carpet / lines / marks (not occluders)
TOP_RES = 2.0              # mm per pixel of the top view used for the field map


def load_field(path=FIELD_STEP):
    """Field STEP -> (trimesh in the sim frame, per-triangle sRGB colours (F, 3))."""
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.OCP.collections import Sequence_TDF_Label
    from OCP.Quantity import Quantity_Color, Quantity_TypeOfColor
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool

    from .mirror import triangulate_faces

    doc = TDocStd_Document(TCollection_ExtendedString("field"))
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.Transfer(doc):
        raise ValueError(f"cannot read STEP file: {path}")
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    colours = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    labels = Sequence_TDF_Label()
    shapes.GetFreeShapes(labels)

    def colour(item, default):
        c = Quantity_Color()
        for kind in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
            if colours.GetColor(item, kind, c):
                return c.Values(Quantity_TypeOfColor.Quantity_TOC_sRGB)
        return default

    mb = MeshBuilder()
    for k in range(1, labels.Length() + 1):
        shape = shapes.GetShape_s(labels.Value(k))
        base = colour(shape, _DEFAULT_COLOUR)
        for face, pts, tris in triangulate_faces(shape, 0.5, 0.1):
            mb.add(pts, tris, colour(face, base))
    mesh, cols = mb.build()
    if mesh is None:
        raise ValueError("STEP file contains no surfaces")
    # rotate -90 deg about Z (STEP +X -> sim -Y) and move the carpet centre to the origin
    v = mesh.vertices - _STEP_CENTRE
    mesh = trimesh.Trimesh(np.column_stack([v[:, 1], -v[:, 0], v[:, 2]]), mesh.faces, process=False)
    return mesh, cols


def occluders(mesh, cols):
    """Field without the floor-level faces: what can block a ray before it reaches the floor."""
    keep = mesh.vertices[mesh.faces][:, :, 2].max(axis=1) >= FLOOR_LEVEL
    sub = trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=False)
    return sub, cols[keep]


def top_view(mesh, cols, res=TOP_RES):
    """Orthographic picture of the field from above. Returns (image HxWx3 float, extent)."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    lo, hi = mesh.bounds
    xs = np.arange(lo[0] + res / 2, hi[0], res)
    ys = np.arange(lo[1] + res / 2, hi[1], res)
    X, Y = np.meshgrid(xs, ys)                         # row 0 = smallest y (imshow origin="lower")
    o = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, hi[2] + 10.0)])
    d = np.tile([0.0, 0.0, -1.0], (len(o), 1))
    tri, ray = RayMeshIntersector(mesh).intersects_id(o, d, multiple_hits=False)
    img = np.zeros((X.size, 3))
    img[ray] = cols[tri]
    return img.reshape(X.shape + (3,)), (lo[0], hi[0], lo[1], hi[1])


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

    def build(self):
        if not self.v:
            return None, np.zeros((0, 3))
        m = trimesh.Trimesh(np.vstack(self.v), np.vstack(self.f), process=False)
        return m, np.vstack(self.c)


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
