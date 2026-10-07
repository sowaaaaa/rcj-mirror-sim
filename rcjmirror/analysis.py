"""Derived views and metrics computed from the traced pixel -> floor mapping."""
import warnings

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from . import config as C


def pixel_scale(X, Y, valid):
    """Local mm-per-pixel along image x and y (NaN where undefined)."""
    dXdi = np.gradient(X, axis=1)
    dYdi = np.gradient(Y, axis=1)
    dXdj = np.gradient(X, axis=0)
    dYdj = np.gradient(Y, axis=0)
    sx, sy = np.hypot(dXdi, dYdi), np.hypot(dXdj, dYdj)
    bad = ~valid
    # discard gradients across validity boundaries / wall edges
    for a in (sx, sy):
        a[bad] = np.nan
    return sx, sy


def _max_scale(sx, sy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.stack([sx, sy]), axis=0)


def top_down(img, X, Y, valid, p, radius=1000.0, mm_per_px=4.0):
    """Bird's-eye view around the robot: each floor cell takes the nearest camera pixel."""
    n = int(2 * radius / mm_per_px)
    xs = np.linspace(p.robot.x - radius, p.robot.x + radius, n)
    ys = np.linspace(p.robot.y + radius, p.robot.y - radius, n)        # +Y is up on screen
    GX, GY = np.meshgrid(xs, ys)
    pts = np.column_stack([X[valid], Y[valid]])
    out = np.full((n, n, 3), 40, np.uint8)
    if len(pts) == 0:
        return out, (xs[0], xs[-1], ys[-1], ys[0])
    tree = cKDTree(pts)
    d, idx = tree.query(np.column_stack([GX.ravel(), GY.ravel()]))
    sx, sy = pixel_scale(X, Y, valid)
    spacing = _max_scale(sx, sy)[valid]
    ok = d < np.fmax(2.0 * spacing[idx], mm_per_px * 1.5)
    colors = img[valid][idx]
    flat = out.reshape(-1, 3)
    flat[ok] = colors[ok]
    return flat.reshape(n, n, 3), (xs[0], xs[-1], ys[-1], ys[0])


def polar_unwrap(img, n_theta=720, r_max=None, center=None):
    H, W = img.shape[:2]
    cx, cy = (W / 2 - 0.5, H / 2 - 0.5) if center is None else center
    r_max = r_max or min(W, H) / 2
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rr = np.arange(int(r_max))
    R, T = np.meshgrid(rr, th, indexing="ij")
    xs, ys = cx + R * np.cos(T), cy - R * np.sin(T)
    chans = [map_coordinates(img[..., c].astype(float), [ys, xs], order=1, mode="constant") for c in range(3)]
    return np.stack(chans, -1).astype(np.uint8)[::-1]       # far ring at the top


def radial_profile(X, Y, valid, p, max_dist=3000.0):
    """Calibration data: pixel radius -> floor distance, plus mm/px along the radius."""
    H, W = X.shape
    j, i = np.mgrid[0:H, 0:W]
    r_px = np.hypot(i - (W / 2 - 0.5), j - (H / 2 - 0.5))
    dist = np.hypot(X - p.robot.x, Y - p.robot.y)
    m = valid & np.isfinite(dist)
    if not m.any():
        return None
    bins = np.arange(0, r_px[m].max() + 1, 1.0)
    which = np.digitize(r_px[m], bins)
    d_med = np.array([np.median(dist[m][which == k]) if (which == k).any() else np.nan
                      for k in range(1, len(bins) + 1)])
    r_c = bins + 0.5
    ok = np.isfinite(d_med)
    r_c, d_med = r_c[ok], d_med[ok]
    # keep the monotonic branch that belongs to the mirror surface: stop at the horizon
    # (distance cap) or where the mapping turns back (mirror rim / frame edge)
    run_max = np.maximum.accumulate(d_med)
    bad = (d_med > max_dist) | (d_med < 0.8 * run_max)
    cut = int(np.argmax(bad)) if bad.any() else len(d_med)
    r_c, d_med = r_c[:cut], d_med[:cut]
    mm_per_px_rad = np.abs(np.gradient(d_med, r_c)) if len(r_c) > 2 else np.full_like(d_med, np.nan)
    mm_per_px_tan = d_med / np.maximum(r_c, 1e-9)
    return dict(r_px=r_c, dist=d_med, rad=mm_per_px_rad, tan=mm_per_px_tan,
                blind=float(d_med.min()) if len(d_med) else float("nan"),
                far=float(d_med.max()) if len(d_med) else float("nan"))


def ball_size_curve(prof, ball_d=C.BALL_D):
    """Apparent ball size in pixels vs distance (radial = stretched axis, tangential = along ring)."""
    return dict(dist=prof["dist"],
                radial=ball_d / np.maximum(prof["rad"], 1e-9),
                tangential=ball_d / np.maximum(prof["tan"], 1e-9))


def ball_heatmap(X, Y, valid, p, step=30.0, ball_d=C.BALL_D):
    """Smallest apparent ball dimension (px) for a ball placed at every field point.

    X, Y, valid must come from floor_map at plane z = ball radius.  NaN = ball not visible.
    """
    xs = np.arange(-C.PLAY_W / 2, C.PLAY_W / 2 + 1, step)
    ys = np.arange(-C.PLAY_L / 2, C.PLAY_L / 2 + 1, step)
    GX, GY = np.meshgrid(xs, ys)
    pts = np.column_stack([X[valid], Y[valid]])
    out = np.full(GX.shape, np.nan)
    if len(pts) == 0:
        return xs, ys, out
    sx, sy = pixel_scale(X, Y, valid)
    scale = _max_scale(sx, sy)[valid]
    tree = cKDTree(pts)
    d, idx = tree.query(np.column_stack([GX.ravel(), GY.ravel()]))
    ok = d < np.fmax(2.0 * scale[idx], step)
    size = np.where(ok, ball_d / scale[idx], np.nan)
    return xs, ys, size.reshape(GX.shape)
