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


def test_floor_texture_markings():
    tex = F.FloorTexture()
    carpet, line = np.array(C.C_CARPET), np.array(C.C_LINE)
    x_line = C.PLAY_W / 2 - C.LINE_W / 2                  # middle of the side boundary line
    got = tex.sample(np.array([200.0, x_line]), np.array([200.0, 0.0]))
    assert got[0] == pytest.approx(carpet, abs=1e-3)
    assert got[1] == pytest.approx(line, abs=1e-3)


def test_texture_cache_depends_on_field_constants(monkeypatch):
    a = F._tex_path()
    monkeypatch.setattr(C, "PLAY_W", C.PLAY_W + 10)
    assert F._tex_path() != a


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
