import numpy as np
import pytest

from rcjmirror import analysis as A
from rcjmirror import config as C
from rcjmirror import field as F
from rcjmirror.config import SimParams
from rcjmirror.mirror import PRESETS
from rcjmirror.render import Simulator


@pytest.fixture(scope="module")
def sim():
    s = Simulator()
    s.set_mirror(profile=("Конус",) + PRESETS["Конус"][:2])
    return s


BLUE, YELLOW = (0.34, 0.43, 1.0), (1.0, 0.7, 0.0)
GREEN, WHITE, BLACK = (0.27, 0.59, 0.28), (0.96, 0.96, 0.95), (0.1, 0.1, 0.1)


def _faces_of(mesh, cols, colour):
    return mesh.triangles_center[np.all(np.abs(cols - colour) < 0.02, axis=1)]


def test_field_step_in_sim_frame(sim):
    mesh, cols = sim.field
    lo, hi = mesh.bounds
    assert lo[:2] == pytest.approx([-930.0, -1235.0], abs=0.5)     # outer face of the 20 mm walls
    assert hi[:2] == pytest.approx([930.0, 1235.0], abs=0.5)
    carpet = _faces_of(mesh, cols, GREEN)
    assert np.median(carpet[:, 2]) == pytest.approx(0.0, abs=0.5)  # carpet at z = 0
    blue, yellow = _faces_of(mesh, cols, BLUE), _faces_of(mesh, cols, YELLOW)
    assert len(blue) and len(yellow)
    edge = C.PLAY_L / 2 - 20.0                                     # goal posts stand on the 20 mm line
    assert np.all(blue[:, 1] > edge) and np.all(yellow[:, 1] < -edge)
    assert np.all(np.abs(blue[:, 0]) <= 300.0 + 1)                 # goal 600 mm wide, centred


def test_field_top_view_markings(sim):
    img, (x0, _, y0, _) = sim.field_top, sim.field_top_extent

    def at(x, y):
        return img[int((y - y0) / F.TOP_RES), int((x - x0) / F.TOP_RES)]

    assert at(200, 200) == pytest.approx(GREEN, abs=0.02)
    assert at(C.PLAY_W / 2 - 10, 0) == pytest.approx(WHITE, abs=0.02)    # side boundary line
    assert at(0, C.PLAY_L / 2 - 10) == pytest.approx(WHITE, abs=0.02)    # goal line
    assert at(0, 0) == pytest.approx(BLACK, abs=0.02)                    # centre spot
    assert at(-925, 0) == pytest.approx(BLACK, abs=0.02)                 # wall


def test_occluders_exclude_floor(sim):
    occ, _ = sim.field_occluders
    assert len(occ.faces) < len(sim.field[0].faces)
    assert occ.vertices[occ.faces][:, :, 2].max(axis=1).min() >= F.FLOOR_LEVEL


def test_render_image(sim):
    p = SimParams()
    img = sim.render(p, 160, 120)
    assert img.shape == (120, 160, 3) and img.dtype == np.uint8
    green = (img[..., 1] > img[..., 0] + 20) & (img[..., 1] > img[..., 2] + 20)
    assert green.mean() > 0.05                           # the carpet is seen in the mirror


def test_floor_map_rotates_with_robot_heading(sim):
    p = SimParams()
    X0, Y0, v0, _ = sim.floor_map(p, 0.0, 80, 60, walls=False)
    p.robot.heading_deg = 90.0
    X1, Y1, v1, _ = sim.floor_map(p, 0.0, 80, 60, walls=False)
    both = v0 & v1
    assert both.sum() > 0.9 * v0.sum()
    # heading +90 deg rotates every floor point by +90 deg around the robot: (x, y) -> (-y, x)
    assert X1[both] == pytest.approx(-Y0[both], abs=1.0)
    assert Y1[both] == pytest.approx(X0[both], abs=1.0)


def test_floor_map_follows_robot_position(sim):
    p = SimParams()
    X0, Y0, v0, d0 = sim.floor_map(p, 0.0, 80, 60, walls=False)
    p.robot.x, p.robot.y = 200.0, -300.0
    X1, Y1, v1, d1 = sim.floor_map(p, 0.0, 80, 60, walls=False)
    both = v0 & v1
    assert X1[both] == pytest.approx(X0[both] + 200.0, abs=1e-6)
    assert Y1[both] == pytest.approx(Y0[both] - 300.0, abs=1e-6)
    assert d1[both] == pytest.approx(d0[both], abs=1e-6)


def test_walls_limit_the_view(sim):
    p = SimParams()
    _, _, _, d_free = sim.floor_map(p, 0.0, 160, 120, walls=False)
    X, Y, v, d = sim.floor_map(p, 0.0, 160, 120, walls=True)
    assert np.all(np.abs(X[v]) <= C.TOTAL_W / 2) and np.all(np.abs(Y[v]) <= C.TOTAL_L / 2)
    assert np.nanmax(d) < np.nanmax(d_free)


def test_blind_zone_is_outside_robot_body(sim):
    p = SimParams()
    _, _, v, d = sim.floor_map(p, 0.0, 160, 120, walls=False)
    assert d[v].min() >= p.robot.size / 2               # the robot body hides the floor under it


def test_radial_profile_is_monotonic(sim):
    p = SimParams()
    X, Y, v, _ = sim.floor_map(p, 0.0, 320, 240, walls=False)
    prof = A.radial_profile(X, Y, v, p)
    assert prof is not None and len(prof["r_px"]) > 10
    assert np.all(np.diff(prof["dist"]) >= -1.0)
    assert prof["blind"] < prof["far"]


def test_render_with_step_mirror():
    s = Simulator()
    s.set_mirror(str(F.FIELD_STEP.parent / "Gyperbolic_mirror.STEP"))
    p = SimParams()
    img = s.render(p, 160, 120)
    green = (img[..., 1] > img[..., 0] + 20) & (img[..., 1] > img[..., 2] + 20)
    assert green.mean() > 0.05
    _, _, v, d = s.floor_map(p, 0.0, 160, 120)
    assert v.any() and np.nanmax(d) > 1000.0                # sees far across the field
