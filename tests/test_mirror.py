from pathlib import Path

import numpy as np
import pytest

from rcjmirror.mirror import PRESETS, curve_from_points, design_profile, load_mirror, profile_curve

STEP = Path(__file__).resolve().parent.parent / "ref" / "Gyperbolic_mirror.STEP"


@pytest.mark.parametrize("name", list(PRESETS))
def test_preset_profile_apex_and_rim(name):
    r, z = profile_curve(name, 30.0, 20.0)
    assert r[0] == 0 and r[-1] == pytest.approx(30.0)
    assert z[0] == pytest.approx(-20.0)
    assert z[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(z) >= -1e-9)                 # convex mirror rises from apex to rim


def test_curve_from_points_passes_through_control_points():
    pts = [(0.0, -20.0), (10.0, -14.0), (20.0, -8.0), (30.0, 0.0)]
    for smooth in (True, False):
        r, z = curve_from_points(pts, smooth)
        # the dense curve is sampled every 0.25 mm, so a kink between samples is cut slightly
        assert np.interp([p[0] for p in pts], r, z) == pytest.approx([p[1] for p in pts], abs=0.02)


def test_design_profile_maps_rays_to_requested_floor_distances():
    """Reflect lens rays off the computed surface and check where they land on the floor."""
    h_cam, d, d_near, d_far, th_in, th_max = 80.0, 60.0, 110.0, 1500.0, 6.0, 26.0
    r, z = design_profile(h_cam, d, d_near, d_far, th_in, th_max, n=400)
    r, z = r[1:], z[1:]                                # drop the cone tip added at the axis
    tz = np.gradient(z, r)
    tang = np.stack([np.ones_like(tz), tz], 1)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)

    P = np.stack([r, z], 1)
    lens = np.array([0.0, -d])
    i = P - lens
    i /= np.linalg.norm(i, axis=1, keepdims=True)
    o = i - 2 * (i * nrm).sum(1, keepdims=True) * nrm
    floor_z = -d - h_cam
    D = P[:, 0] + o[:, 0] * (floor_z - P[:, 1]) / o[:, 1]

    th = np.arctan2(P[:, 0], P[:, 1] + d)
    u = (np.tan(th) - np.tan(np.radians(th_in))) / (np.tan(np.radians(th_max)) - np.tan(np.radians(th_in)))
    expected = d_near + u * (d_far - d_near)            # gamma = 1: linear in tan(theta)
    inner = slice(5, -5)                                # np.gradient is one-sided at the ends
    assert np.all(o[inner, 1] < 0)                      # every reflected ray goes down to the floor
    assert D[inner] == pytest.approx(expected[inner], rel=0.02)
    assert np.all(np.diff(D[inner]) > 0)


def test_step_mirror_is_normalised():
    m, info = load_mirror(str(STEP))
    assert info["axis"] == "Y"                          # SolidWorks Y-up export rotated to Z
    assert info["size"][0] == pytest.approx(54.0, abs=0.1)
    assert info["size"][1] == pytest.approx(54.0, abs=0.1)
    lo, hi = m.bounds
    assert lo[0] == pytest.approx(-hi[0], abs=0.1)      # centred on the axis
    assert lo[2] < 0 < hi[2]                            # apex below the first plane, rim above


def test_points_profile_builds_a_mirror():
    pts = ((0.0, -20.0), (10.0, -14.0), (20.0, -8.0), (30.0, 0.0))
    m, info = load_mirror(profile=("points", pts, True))
    assert m.bounds[0][2] == pytest.approx(-20.0, abs=0.2)
    assert info["size"][0] == pytest.approx(2 * (30.0 + 6.0), abs=0.5)    # radius + rim ring
