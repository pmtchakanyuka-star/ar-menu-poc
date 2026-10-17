"""Build the real-size, photo-textured tiramisu for the AR menu.

Pipeline:
  1. Shape: a slice of tiramisu (cocoa-dusted top with soft rounded edges, slightly
     uneven cream sides) on a black ribbed takeaway tray, with a sprig of mint.
     The sizes come from the photos, with the tray as the yardstick (see SIZES).
  2. Texture: the top is the cocoa from the straight-down photo; the sides are the
     front face of the slice from two photos, rectified flat and wrapped round, so
     every pixel is a real one. The photos' lighting is evened out first, so the
     AR viewer's own lights do the shading. Bump and roughness maps come from the
     same pixels: matte cocoa, moist cream.
  3. Export: models/tiramisu.glb (web / Android) and models/tiramisu.usdz (iPhone).

World frame: Y up, metres, origin at the centre of the tray's underside, +Z = front.
The reference photos are not in the repo (they show more than the dessert): put
them in tools/tiramisu/ref/ (see README).
Run:  .venv/Scripts/python tools/build_tiramisu.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from model_io import write_glb, write_usdz  # noqa: E402

OUT_DIR = ROOT / "models"
REF = ROOT / "tools" / "tiramisu" / "ref"

# SIZES. The tray is ~10x as wide as it is deep, and the slice spans ~70% of its width
# and stands ~1.7x the tray's depth (low.jpg); the slice is ~0.3x as tall as wide
# (front.jpg, corner.jpg, corrected for the camera's tilt).
SLICE_W, SLICE_D, SLICE_H = 0.108, 0.090, 0.038   # width (x), depth (z), height at the centre
DOME = 0.002                 # the top is that much higher in the middle than at its edge
EDGE_RH, EDGE_RV = 0.008, 0.010   # rounded top edge: horizontal / vertical radius
CORNER_R = 0.003             # plan-view corner radius of the flat top (sides: + EDGE_RH)
TRAY_W, TRAY_H = 0.160, 0.017     # tray: outer size at the rim, rim height
TRAY_WALL, TRAY_LIP = 0.018, 0.004  # horizontal run of the ribbed wall, rim flange
TRAY_FLOOR_R = 0.008         # plan-view corner radius of the tray floor
FLOOR_Y = 0.0022             # tray floor surface above the table
N_RIBS = 9

N_U = 256                    # mesh columns around the slice
TEX_W, TEX_H = 2048, 1024    # slice texture: u = around (seam at the back), v = top centre -> foot
PX_CM = 80                   # resolution of the rectified photo crops

# Rectified photo crops: (file, corner points TL, TR, BR, BL in photo pixels).
# The side faces run from the crest of the top edge down to the tray.
SIDE_FACES = {
    "corner": ("corner.jpg", [(1380, 1415), (2640, 1350), (2622, 1745), (1390, 1935)]),
    "front": ("front.jpg", [(1200, 1545), (2785, 1560), (2790, 1965), (1200, 2010)]),
}
# Cocoa top from the straight-down photo: rows of the crop run along the slice's width.
TOP_CROP = ("top.jpg", (1590, 0, 2256, 860))
TOP_BLEMISH = (2037, 288, 46)          # a dark crumb in the cocoa: x, y, radius (px)


def srgb(r: int, g: int, b: int) -> list[float]:
    c = np.array([r, g, b], dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return [round(float(v), 5) for v in lin]


MATERIALS = {
    "Tiramisu": dict(color=[1.0, 1.0, 1.0], texture="tiramisu_basecolor.jpg", normal="tiramisu_normal.jpg",
                     rough_tex="tiramisu_roughness.jpg", roughness=1.0, clearcoat=0.0, clearcoat_roughness=0.0),
    "Mint": dict(color=[1.0, 1.0, 1.0], texture="mint_basecolor.jpg", normal="mint_normal.jpg",
                 roughness=0.55, clearcoat=0.0, clearcoat_roughness=0.0),
    "Tray": dict(color=srgb(28, 28, 30), roughness=0.55, clearcoat=0.2, clearcoat_roughness=0.35),
}


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- #
# Rounded-rectangle sweeps: X = s * (C0(u) + d * n(u)) + y * up
# --------------------------------------------------------------------------- #
def rrect(a, b, r, d_ref, n):
    """n points on a rounded rectangle C0 (half sizes a, b, corner radius r) and their
    outward normals, spaced evenly by arc length along C0 offset outward by d_ref.
    Starts at the middle of the back edge (-z), so the texture seam is at the back."""
    ax, bz = a - r, b - r
    segs = [  # (kind, start, end, length at offset d_ref)
        ("e", (0.0, -1.0), (ax, -b), (-ax, -b)),
        ("a", (-ax, -bz), -np.pi / 2, -np.pi),
        ("e", (-1.0, 0.0), (-a, -bz), (-a, bz)),
        ("a", (-ax, bz), np.pi, np.pi / 2),
        ("e", (0.0, 1.0), (-ax, b), (ax, b)),
        ("a", (ax, bz), np.pi / 2, 0.0),
        ("e", (1.0, 0.0), (a, bz), (a, -bz)),
        ("a", (ax, -bz), 0.0, -np.pi / 2),
    ]
    lens = [np.hypot(s[3][0] - s[2][0], s[3][1] - s[2][1]) if s[0] == "e" else (r + d_ref) * np.pi / 2
            for s in segs]
    # start half-way along the back edge
    total = sum(lens)
    t = (np.arange(n) / n * total + lens[0] / 2) % total
    P, N = np.zeros((n, 2)), np.zeros((n, 2))
    edges = np.concatenate([[0], np.cumsum(lens)])
    for i, s in enumerate(segs):
        m = (t >= edges[i]) & (t < edges[i + 1])
        f = (t[m] - edges[i]) / lens[i]
        if s[0] == "e":
            p0, p1 = np.array(s[2]), np.array(s[3])
            P[m] = p0 + f[:, None] * (p1 - p0)
            N[m] = s[1]
        else:
            ang = s[2] + f * (s[3] - s[2])
            N[m] = np.stack([np.cos(ang), np.sin(ang)], -1)
            P[m] = np.array(s[1]) + r * N[m]
    return P, N, t, total


def sweep(C0, N0, S, D, Y):
    """Grid (n_u, n_rows, 3) for the closed loop C0/N0 and per-row (s, d, y), x/z order."""
    xz = S[None, :, None] * (C0[:, None, :] + D[None, :, None] * N0[:, None, :])
    return np.stack([xz[..., 0], np.broadcast_to(Y, xz.shape[:2]), xz[..., 1]], -1)


def grid_normals(X, up_rows=(), down_rows=()):
    """Vertex normals of a grid closed around axis 0, oriented away from the y axis/up."""
    du = np.roll(X, -1, 0) - np.roll(X, 1, 0)
    dv = np.gradient(X, axis=1)
    N = np.cross(dv, du)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
    for j in up_rows:
        N[:, j] = [0, 1, 0]
    for j in down_rows:
        N[:, j] = [0, -1, 0]
    return N


def grid_mesh(X, N, UV=None, T=None):
    """Triangulate a closed (n_u, n_rows) grid; adds the seam column copy for UVs."""
    X, N = np.concatenate([X, X[:1]]), np.concatenate([N, N[:1]])
    if T is not None:
        T = np.concatenate([T, T[:1]])
    nc, nr = X.shape[:2]
    idx = np.arange(nc * nr).reshape(nc, nr)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    Vf, Nf = X.reshape(-1, 3), N.reshape(-1, 3)
    fn = np.cross(Vf[F[:, 1]] - Vf[F[:, 0]], Vf[F[:, 2]] - Vf[F[:, 0]])
    F = F[np.linalg.norm(fn, axis=1) > 1e-14]           # drop the degenerate pole triangles
    fn = np.cross(Vf[F[:, 1]] - Vf[F[:, 0]], Vf[F[:, 2]] - Vf[F[:, 0]])
    if (fn * Nf[F].mean(axis=1)).sum(axis=1).mean() < 0:
        F = F[:, ::-1].copy()
    mesh = {"V": Vf.astype(np.float32), "N": Nf.astype(np.float32), "F": F}
    if UV is not None:
        UV = np.concatenate([UV, UV[:1]])
        UV[-1, :, 0] = 1.0
        mesh["UV"] = UV.reshape(-1, 2).astype(np.float32)
    if T is not None:
        mesh["T"] = T.reshape(-1, 4).astype(np.float32)
    return mesh


# --------------------------------------------------------------------------- #
# The slice
# --------------------------------------------------------------------------- #
A0, B0 = SLICE_W / 2 - EDGE_RH, SLICE_D / 2 - EDGE_RH      # the flat top's half sizes
H_EDGE = SLICE_H - DOME                                    # height where the top meets the edge


def slice_profile():
    """Per-row (s, d, y) from the top centre over the rounded edge, down the side, under."""
    s_top = np.linspace(0, 1, 34) ** 0.8
    y_top = H_EDGE + DOME * (1 - s_top**2)
    phi = np.linspace(0, np.pi / 2, 21)[1:]
    d_edge, y_edge = EDGE_RH * np.sin(phi), H_EDGE - EDGE_RV * (1 - np.cos(phi))
    y_side = np.linspace(H_EDGE - EDGE_RV, 0.0, 30)[1:]
    S = np.concatenate([s_top, np.ones(len(phi) + len(y_side) + 1)])
    D = np.concatenate([np.zeros(len(s_top)), d_edge, np.full(len(y_side), EDGE_RH), [EDGE_RH - 0.0012]])
    Y = np.concatenate([y_top, y_edge, y_side, [-0.0008]])
    return S, D, Y, len(s_top)


SP, DP, YP, N_TOP = slice_profile()
ROWS = len(SP)
C0, N0, T_SIDE, P_SIDE = rrect(A0, B0, CORNER_R, EDGE_RH, N_U)   # T_SIDE: arc length round the side


def periodic_noise(rng, u, v, kmax, vfreq, amp):
    """Smooth noise, periodic in u (0..1), at heights v (any units, scaled by vfreq)."""
    out = np.zeros(np.broadcast(u, v).shape)
    for k in range(1, kmax + 1):
        for _ in range(2):
            out += (rng.normal() / k) * np.cos(2 * np.pi * k * u + rng.uniform(0, 2 * np.pi)
                                                + rng.normal() * vfreq * v)
    return amp * out / np.abs(out).max()


def build_slice():
    rng = np.random.default_rng(7)
    X = sweep(C0, N0, SP, DP, YP)
    u = (np.arange(N_U) / N_U)[:, None]
    # Soft, uneven cream: the sides bulge and slump a little, the top undulates.
    side_w = smoothstep(0.0, 0.5, (np.arange(ROWS) - N_TOP + 6) / 12.0)[None, :]
    bulge = periodic_noise(rng, u, YP[None, :], 9, 90.0, 0.0018)
    foot = 0.0012 * smoothstep(0.012, 0.0, YP)[None, :] * (0.6 + 0.4 * periodic_noise(rng, u, 0 * YP, 14, 0, 1.0))
    nh = np.stack([N0[:, 0], np.zeros(N_U), N0[:, 1]], -1)[:, None, :]
    X = X + ((bulge + foot) * side_w)[..., None] * nh
    tx, tz = X[..., 0] / SLICE_W, X[..., 2] / SLICE_D
    lump = 0.0010 * (np.sin(9.0 * tx + 1.3) * np.cos(7.0 * tz - 0.4) + 0.6 * np.sin(17 * tx * tz + 2.0))
    X[..., 1] += lump * (1 - side_w) * (np.arange(ROWS) < N_TOP + 8)[None, :]
    X[..., 1] += FLOOR_Y
    N = grid_normals(X, up_rows=[0])
    # UVs: u around, v = row (the texture's rows are baked the same way)
    UV = np.stack(np.broadcast_arrays(u, (np.arange(ROWS) / (ROWS - 1))[None, :]), -1).astype(float)
    # Tangents point round the slice (+u); w makes cross(N, T) * w point toward -v.
    Tv = np.roll(X, -1, 0) - np.roll(X, 1, 0)
    Tv -= (Tv * N).sum(-1, keepdims=True) * N
    Tv /= np.maximum(np.linalg.norm(Tv, axis=-1, keepdims=True), 1e-12)
    Tv[:, 0] = Tv[:, 1]                                   # the pole has no 'around' of its own
    dv = np.gradient(X, axis=1)
    w = -np.sign((np.cross(N, Tv) * dv).sum(-1))
    w[w == 0] = 1.0
    T = np.concatenate([Tv, w[..., None]], -1)
    return grid_mesh(X, N, UV, T)


# --------------------------------------------------------------------------- #
# Slice textures
# --------------------------------------------------------------------------- #
def load(name):
    path = REF / name
    if not path.exists():
        sys.exit(f"missing reference photo {path} (see README)")
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def rectify(photo, quad, w_px, h_px):
    """Warp the photo quad (TL, TR, BR, BL) to a w_px x h_px image (projective)."""
    src = np.array(quad, float)
    dst = np.array([(0, 0), (w_px, 0), (w_px, h_px), (0, h_px)], float)
    A = []
    for (x, y), (X, Y) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -X * x, -X * y, -X], [0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y]]
    Hm = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    yy, xx = np.mgrid[0:h_px, 0:w_px] + 0.5
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    q = Hm @ p
    sx, sy = q[0] / q[2] - 0.5, q[1] / q[2] - 0.5
    return np.stack([map_coordinates(photo[..., c], [sy, sx], order=1, mode="nearest").reshape(h_px, w_px)
                     for c in range(3)], -1)


def even_light(img, sigma, axis=None):
    """Divide out slow brightness changes (along `axis` only, or both), keeping the mean."""
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    if axis is None:
        lp = gaussian_filter(lum, sigma, mode="reflect")
        mean = lum.mean()
    else:
        lp = gaussian_filter1d(lum, sigma, axis=axis, mode="reflect")
        mean = lum.mean(axis=axis, keepdims=True)
    return img * np.clip(mean / np.maximum(lp, 1.0), 0.6, 1.7)[..., None]


def side_atlas():
    """The cream sides all the way round: (H, P) image, x = arc length round the side
    (from the back middle, like the mesh), y = height (row 0 = SLICE_H, last = tray)."""
    h_face = int(round(H_EDGE * 100 * PX_CM))
    faces = {}
    for k, (f, quad) in SIDE_FACES.items():
        img = rectify(load(f), quad, int(round(SLICE_W * 100 * PX_CM)), h_face)
        faces[k] = even_light(img, 2.0 * PX_CM, axis=1)
    pw = int(round(P_SIDE * 100 * PX_CM))
    x = (np.arange(pw) + 0.5) / PX_CM / 100                     # metres round the side
    # Face intervals (in arc length from the back middle): back, left, front, right.
    corner = (CORNER_R + EDGE_RH) * np.pi / 2
    e_w, e_d = 2 * (A0 - CORNER_R), 2 * (B0 - CORNER_R)
    b0 = e_w / 2 + corner / 2
    spans = [(-b0, b0, "front", 0.00),                          # back: the other photo's face
             (b0, b0 + e_d + corner, "front", -0.9),
             (b0 + e_d + corner, 3 * b0 + e_d + corner, "corner", 0.0),
             (3 * b0 + e_d + corner, P_SIDE - b0, "corner", 0.9)]
    fade = 0.006
    acc, wsum = np.zeros((h_face, pw, 3), np.float32), np.zeros(pw, np.float32)
    for lo, hi, k, off in spans:
        src = faces[k]
        for shift in (-P_SIDE, 0.0, P_SIDE):                    # wrap round the seam
            xs = x - shift
            w = smoothstep(lo - fade, lo + fade, xs) * smoothstep(hi + fade, hi - fade, xs)
            if not w.any():
                continue
            # centre the span on the face, shifted by `off` (-1..1) of the room left over
            fx = (xs - (lo + hi) / 2) / SLICE_W + 0.5 + off * 0.5 * (1 - (hi - lo + 2 * fade) / SLICE_W)
            cols = np.clip(fx, 0.01, 0.99) * src.shape[1] - 0.5
            rr, cc = np.mgrid[0:h_face, 0:pw]
            samp = np.stack([map_coordinates(src[..., c], [rr, np.broadcast_to(cols, (h_face, pw))], order=1)
                             for c in range(3)], -1)
            acc += samp * w[None, :, None]
            wsum += w
    side = acc / np.maximum(wsum, 1e-6)[None, :, None]
    # extend upward over the dome with the face's top row (the top texture covers it)
    top_rows = int(round(DOME * 100 * PX_CM))
    return np.concatenate([np.repeat(side[:1], top_rows, 0), side], 0)


def top_image():
    """Cocoa top: (rows along x, cols along z) image covering SLICE_W x SLICE_D."""
    f, (x0, y0, x1, y1) = TOP_CROP
    photo = load(f).copy()
    bx, by, br = TOP_BLEMISH                  # cover it with clean cocoa from beside it
    yy, xx = np.mgrid[by - 2 * br:by + 2 * br, bx - 2 * br:bx + 2 * br]
    m = gaussian_filter((np.hypot(xx - bx, yy - by) < br).astype(np.float32), br / 4)[..., None]
    clean = photo[by + 3 * br:by + 7 * br, bx - 2 * br:bx + 2 * br]
    photo[by - 2 * br:by + 2 * br, bx - 2 * br:bx + 2 * br] = photo[by - 2 * br:by + 2 * br, bx - 2 * br:bx + 2 * br] * (1 - m) + clean * m
    img = even_light(photo[y0:y1, x0:x1], 120)
    return img                     # rows (photo y) -> slice x, cols (photo x) -> slice z


def bake_textures():
    rows_f = (np.arange(TEX_H) + 0.5) / TEX_H * (ROWS - 1)
    S = np.interp(rows_f, np.arange(ROWS), SP)
    D = np.interp(rows_f, np.arange(ROWS), DP)
    Y = np.interp(rows_f, np.arange(ROWS), YP)
    uu = (np.arange(TEX_W) + 0.5) / TEX_W
    C, Nn, Ts, _ = rrect(A0, B0, CORNER_R, EDGE_RH, TEX_W)
    xz = S[:, None, None] * (C[None] + D[:, None, None] * Nn[None])      # (H, W, 2)
    # upward-ness of each texel row (from the profile), for the cocoa / cream blend
    prof = np.stack([S * A0 + D, Y], -1)                                  # (distance out from the middle, height)
    dprof = np.gradient(prof, axis=0)
    ny = np.abs(dprof[:, 0]) / np.maximum(np.hypot(dprof[:, 0], dprof[:, 1]), 1e-12)

    side = side_atlas()
    top = top_image()
    rng = np.random.default_rng(3)
    # side colour by (arc length round the side, height)
    sx = np.broadcast_to(Ts[None, :], (TEX_H, TEX_W)) * 100 * PX_CM - 0.5
    sy = np.broadcast_to(((SLICE_H - Y) * 100 * PX_CM - 0.5)[:, None], (TEX_H, TEX_W))
    side_c = np.stack([map_coordinates(side[..., c], [sy, sx], order=1, mode="nearest") for c in range(3)], -1)
    # top colour by position (x along the crop's rows, z along its columns)
    ty = (xz[..., 0] / SLICE_W + 0.5) * top.shape[0] - 0.5
    tx = (xz[..., 1] / SLICE_D + 0.5) * top.shape[1] - 0.5
    top_c = np.stack([map_coordinates(top[..., c], [ty, tx], order=1, mode="nearest") for c in range(3)], -1)
    # cocoa settles on what faces up: blend across the rounded edge with a ragged line
    rag = periodic_noise(rng, uu[None, :], rows_f[:, None] / ROWS, 40, 4.0, 0.22)
    rag += 0.08 * gaussian_filter(rng.normal(size=(TEX_H, TEX_W)), 2.0, mode="wrap") / 0.14
    w_top = smoothstep(0.35, 0.72, ny[:, None] + rag)
    base = top_c * w_top[..., None] + side_c * (1 - w_top[..., None])
    base = np.clip(base, 0, 255)

    # Relief from the photo detail (lighter cream lumps stand out), metres per texel
    lum = base @ np.array([0.299, 0.587, 0.114], np.float32)
    detail = (lum - gaussian_filter(lum, 3.0, mode=("nearest", "wrap"))) / 255.0
    seglen = np.hypot(np.diff(np.interp(np.arange(TEX_H + 1) / TEX_H * (ROWS - 1), np.arange(ROWS), SP) * A0
                              + np.interp(np.arange(TEX_H + 1) / TEX_H * (ROWS - 1), np.arange(ROWS), DP)),
                      np.diff(np.interp(np.arange(TEX_H + 1) / TEX_H * (ROWS - 1), np.arange(ROWS), YP)))
    dv = np.maximum(seglen, 2e-5)[:, None]
    du = np.maximum(S * (P_SIDE - 2 * np.pi * EDGE_RH + 2 * np.pi * D) / TEX_W, 2e-5)[:, None]
    h = detail * (0.0016 + 0.0024 * (1 - w_top))                    # cream is lumpier than cocoa
    h *= smoothstep(0.03, 0.12, S)[:, None]       # texels shrink to nothing at the top's centre
    dhdu = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * du)
    dhdv = (np.vstack([h[1:], h[-1:]]) - np.vstack([h[:1], h[:-1]])) / (2 * dv)
    n = np.stack([-dhdu, dhdv, np.ones_like(h)], -1)                 # +x = +u, +y = up = -v (glTF)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal = ((n * 0.5 + 0.5) * 255).astype(np.uint8)

    # Matte cocoa, moist cream, biscuit at the bottom in between.
    biscuit = smoothstep(0.006, 0.002, Y)[:, None]
    rough = 0.93 * w_top + (0.62 * (1 - biscuit) + 0.8 * biscuit) * (1 - w_top) - 0.5 * detail * (1 - w_top)
    rough_img = np.zeros((TEX_H, TEX_W, 3), np.uint8)
    rough_img[..., 0] = 255
    rough_img[..., 1] = (np.clip(rough, 0.3, 0.97) * 255).astype(np.uint8)   # glTF: G = roughness
    return (Image.fromarray(base.astype(np.uint8)), Image.fromarray(normal),
            Image.fromarray(rough_img).resize((TEX_W // 2, TEX_H // 2), Image.BILINEAR))


# --------------------------------------------------------------------------- #
# Mint sprig
# --------------------------------------------------------------------------- #
LEAF_TOP, LEAF_UNDER = (92, 132, 36), (128, 158, 76)    # sRGB, from the photos' leaves (shaded side)


def mint_textures(n=512):
    """Leaf texture (u = along the midrib, v = across): veins, mottling, and a bump map."""
    rng = np.random.default_rng(5)
    s = (np.arange(n) + 0.5) / n
    t = ((np.arange(n // 2) + 0.5) / (n // 2)) * 2 - 1
    T, S_ = np.meshgrid(t, s, indexing="ij")             # rows = across, cols = along
    mid = np.exp(-(T / 0.035) ** 2)
    # side veins leave the midrib and sweep toward the tip
    phase = (S_ - 0.55 * np.abs(T) ** 1.2) * 9.0
    veins = np.exp(-((phase - np.round(phase)) / 0.07) ** 2) * (np.abs(T) < 0.92)
    puff = gaussian_filter(rng.normal(size=T.shape), 6) * 6
    height = -0.6 * veins - 0.9 * mid + 0.5 * puff               # veins sit in grooves, the blade puffs up
    shade = 1.0 + 0.10 * puff - 0.05 * np.abs(T)
    top = np.array(LEAF_TOP, np.float32)[None, None] * shade[..., None]
    top = top * (1 - 0.25 * veins - 0.2 * mid)[..., None] + np.array([175, 190, 90])[None, None] * (0.25 * veins + 0.2 * mid)[..., None]
    gy, gx = np.gradient(height * 0.06)
    nrm = np.stack([-gx * 90, gy * 90, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    # the underside is stored in the bottom half
    under = np.array(LEAF_UNDER, np.float32)[None, None] * (1 + 0.06 * puff)[..., None]
    under = under * (1 - 0.1 * veins)[..., None] + np.array([200, 205, 150]) * (0.1 * veins)[..., None]
    color = np.clip(np.concatenate([top, under], 0), 0, 255).astype(np.uint8)
    nimg = ((np.concatenate([nrm, nrm], 0) * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(color), Image.fromarray(nimg)


def leaf(length, width, rng, nu=28, nv=13):
    """One mint leaf in local coords: x along the midrib, y up, z across. Two-sided."""
    s = np.linspace(0, 1, nu)
    t = np.linspace(-1, 1, nv)
    S_, T = np.meshgrid(s, t, indexing="ij")
    half = width / 2 * np.sin(np.pi * np.clip(S_, 0, 1) ** 0.75) ** 0.9
    teeth = 1 + 0.07 * np.abs(np.sin(np.pi * 9 * S_))
    z = T * half * teeth
    x = S_ * length
    y = 0.28 * np.abs(z) + 0.10 * x**2 / length - 0.25 * z**2 / max(width, 1e-6)  # fold + droop + cupping
    y = y + 0.0004 * gaussian_filter(rng.normal(size=x.shape), 1.2)                # crinkles
    P = np.stack([x, y, z], -1)
    du, dv = np.gradient(P, axis=0), np.gradient(P, axis=1)
    N = np.cross(dv, du)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
    N[N[..., 1] < 0] *= -1
    idx = np.arange(nu * nv).reshape(nu, nv)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    V, Nf = P.reshape(-1, 3), N.reshape(-1, 3)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    keep = np.linalg.norm(fn, axis=1) > 1e-14
    F, fn = F[keep], fn[keep]
    if (fn * Nf[F].mean(1)).sum(1).mean() < 0:
        F = F[:, ::-1]
    uv_top = np.stack([S_, (T + 1) / 4], -1).reshape(-1, 2)              # top half of the texture
    uv_under = uv_top + [0.0, 0.5]
    off = 0.00025
    V2 = np.concatenate([V, V - off * Nf])
    N2 = np.concatenate([Nf, -Nf])
    F2 = np.concatenate([F, F[:, ::-1] + len(V)])
    UV2 = np.concatenate([uv_top, uv_under])
    return V2, N2, F2, UV2


def build_mint(top_y):
    """A sprig of three leaves sitting on the cocoa just behind the middle (as served)."""
    rng = np.random.default_rng(9)
    base = np.array([0.003, top_y + 0.0012, -0.010])
    # (length, width, heading round y in deg (0 = toward +x), tilt up in deg, roll)
    leaves = [(0.026, 0.017, 100, 40, 0), (0.023, 0.015, -20, 14, -10), (0.020, 0.013, 205, 18, 12),
              (0.010, 0.006, 60, 55, 0), (0.009, 0.0055, 150, 60, 0)]
    Vs, Ns, Fs, UVs, Ts = [], [], [], [], []
    count = 0
    for L, W, head, tilt, roll in leaves:
        V, N, F, UV = leaf(L, W, rng)
        def rot(v, ang, ax):
            c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            i, j = [(1, 2), (2, 0), (0, 1)][ax]
            out = v.copy()
            out[:, i], out[:, j] = c * v[:, i] - s * v[:, j], s * v[:, i] + c * v[:, j]
            return out
        for (ang, ax) in ((roll, 0), (tilt, 2), (head, 1)):
            V, N = rot(V, ang, ax), rot(N, ang, ax)
        T = rot(rot(rot(np.tile([1.0, 0, 0], (len(V), 1)), roll, 0), tilt, 2), head, 1)
        Vs.append(V + base)
        Ns.append(N)
        Fs.append(F + count)
        UVs.append(UV)
        Ts.append(T)
        count += len(V)
    V, N = np.concatenate(Vs), np.concatenate(Ns)
    T = np.concatenate(Ts)
    T -= (T * N).sum(-1, keepdims=True) * N
    T /= np.maximum(np.linalg.norm(T, axis=-1, keepdims=True), 1e-12)
    T4 = np.concatenate([T, np.ones((len(T), 1))], -1)
    return {"V": V.astype(np.float32), "N": N.astype(np.float32), "F": np.concatenate(Fs),
            "UV": np.concatenate(UVs).astype(np.float32), "T": T4.astype(np.float32)}


# --------------------------------------------------------------------------- #
# The tray
# --------------------------------------------------------------------------- #
def build_tray():
    """Thin black thermoformed tray: flat floor, ribbed sloping walls, rolled rim."""
    af = TRAY_W / 2 - TRAY_WALL - TRAY_LIP
    C, Nn, _, _ = rrect(af, af, TRAY_FLOOR_R, TRAY_WALL, 192)
    S, D, Y = [], [], []

    def add(s, d, y):
        s, d, y = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, float)) for a in (s, d, y)))
        S.extend(s); D.extend(d); Y.extend(y)

    add(np.linspace(0, 1, 14)[:-1] ** 0.7, 0.0, FLOOR_Y)
    # inner wall: rising and flaring, with ribs running round it
    q = np.linspace(0, 1, 9 * N_RIBS + 1)
    wall_d = TRAY_WALL * (0.12 * q + 0.88 * q**1.15)
    wall_y = FLOOR_Y + (TRAY_H - FLOOR_Y - 0.0015) * smoothstep(-0.25, 1.0, q) / smoothstep(-0.25, 1.0, 1.0)
    slope = np.arctan2(np.gradient(wall_y), np.gradient(wall_d))
    rib = 0.00045 * np.sin(2 * np.pi * N_RIBS * q) * smoothstep(0, 0.08, q) * smoothstep(1, 0.92, q)
    add(1.0, wall_d - rib * np.sin(slope), wall_y + rib * np.cos(slope))
    # rolled rim: flange out, then a bead curling down and under
    d0, y0 = TRAY_WALL, TRAY_H - 0.0015
    ang = np.linspace(np.pi, 0, 12)[1:]
    add(1.0, d0 + 0.0015 + 0.0015 * np.cos(ang), y0 + 0.0015 * np.sin(ang))
    rb = 0.0016
    ang = np.linspace(np.pi / 2, -np.pi * 0.75, 16)[1:]
    add(1.0, d0 + 0.003 + rb * np.cos(ang) * 0.9, y0 - rb + 0.0015 + rb * np.sin(ang))
    # outer wall: back down, a plastic thickness outside the inner one
    q2 = np.linspace(1, 0, 40)
    od = TRAY_WALL * (0.12 * q2 + 0.88 * q2**1.15) + 0.0007
    oy = (TRAY_H - 0.003) * smoothstep(-0.25, 1.0, q2) / smoothstep(-0.25, 1.0, 1.0)
    rib2 = 0.00045 * np.sin(2 * np.pi * N_RIBS * q2) * smoothstep(0, 0.08, q2) * smoothstep(1, 0.92, q2)
    sl2 = np.arctan2(np.gradient(oy), np.gradient(od))
    add(1.0, od + rib2 * np.sin(sl2), np.maximum(oy - rib2 * np.cos(sl2), 0.0))
    # underside back to the middle
    add(np.linspace(1, 0, 10)[1:] ** 0.7, 0.0007, 0.0)
    S, D, Y = map(np.array, (S, D, Y))
    X = sweep(C, Nn, S, D, Y)
    N = grid_normals(X, up_rows=[0], down_rows=[len(S) - 1])
    return grid_mesh(X, N)


# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(exist_ok=True)
    base, normal, rough = bake_textures()
    base.save(REF.parent / "tiramisu_basecolor_preview.png")
    mint_c, mint_n = mint_textures()
    textures = {}
    for name, im, q in (("tiramisu_basecolor.jpg", base, 90), ("tiramisu_normal.jpg", normal, 92),
                        ("tiramisu_roughness.jpg", rough, 90), ("mint_basecolor.jpg", mint_c, 90),
                        ("mint_normal.jpg", mint_n, 92)):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True)
        textures[name] = buf.getvalue()

    slab = build_slice() | {"name": "Tiramisu", "material": "Tiramisu"}
    top_y = slab["V"][:, 1].max()
    mint = build_mint(top_y - 0.0006) | {"name": "Mint", "material": "Mint"}
    tray = build_tray() | {"name": "Tray", "material": "Tray"}
    meshes = [tray, slab, mint]
    write_glb(OUT_DIR / "tiramisu.glb", meshes, textures, MATERIALS, "Tiramisu", generator="ar-menu-poc/build_tiramisu.py")
    write_usdz(OUT_DIR / "tiramisu.usdz", meshes, textures, MATERIALS, "Tiramisu")

    allV = np.vstack([m["V"] for m in meshes])
    lo, hi = allV.min(axis=0), allV.max(axis=0)
    sv = slab["V"]
    print(f"triangles      : {sum(len(m['F']) for m in meshes):,}   vertices: {sum(len(m['V']) for m in meshes):,}")
    print(f"overall        : {(hi - lo)[0] * 100:.1f} x {(hi - lo)[2] * 100:.1f} cm, {(hi - lo)[1] * 100:.1f} cm tall")
    print(f"slice          : {np.ptp(sv[:, 0]) * 100:.1f} x {np.ptp(sv[:, 2]) * 100:.1f} cm, "
          f"{(sv[:, 1].max() - FLOOR_Y) * 100:.1f} cm tall")
    print("textures       : " + ", ".join(f"{k} {len(v) / 1e6:.2f} MB" for k, v in textures.items()))
    for f in ("tiramisu.glb", "tiramisu.usdz"):
        print(f"{f:<15}: {(OUT_DIR / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
