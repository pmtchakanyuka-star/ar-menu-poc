"""Build the real-size, photo-textured pumpkin gnocchi for the AR menu.

Pipeline:
  1. Camera: the bowl's rim is found in the overhead photo and a pinhole camera
     (focal length from the EXIF: 24 mm on the EOS M50 II's 22.3 mm-wide sensor)
     is fitted to it, so every point of the dish maps to its exact photo pixel.
  2. Shape: the bowl is a surface of revolution with the profile measured from the
     side-on photo. The food is a height field: solid food (gnocchi, walnuts) is
     separated from thin sauce by its colour and shape in the overhead photo,
     piled to the height the side photos show, with each gnocchi's relief taken
     from the photo's light and shade.
  3. Texture: the inside of the bowl and all the food use the overhead photo
     itself through the fitted camera (UVs = projected pixels), at the photo's
     native resolution, so every crumb of walnut and fleck of pepper is real.
     The bump and roughness maps come from the same pixels.
  4. Export: models/gnocchi.glb (web / Android) and models/gnocchi.usdz (iPhone).

World frame: Y up, metres, origin at the centre of the bowl's foot, on the table.
The reference photos are not in the repo; put them in tools/gnocchi/ref/ (see README).
Run:  .venv/Scripts/python tools/build_gnocchi.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from model_io import write_glb, write_usdz  # noqa: E402

OUT_DIR = ROOT / "models"
REF = ROOT / "tools" / "gnocchi" / "ref"
PHOTO = REF / "overhead.jpg"
CAMERA_JSON = ROOT / "tools" / "gnocchi" / "camera.json"

# SIZES. No yardstick in the photos: the bowl is taken as a standard 23 cm pasta
# bowl. Its proportions come from the side-on photo (side.jpg): 0.23x as tall as
# wide, foot 0.47x the rim. The food peaks at about rim height there.
BOWL_R = 0.115               # rim radius
BOWL_H = 0.053               # rim height above the table
WALL = 0.0045                # glaze-to-glaze wall thickness
FLOOR_Y = 0.009              # inside floor above the table
FOOD_EDGE, FOOD_PEAK = 0.012, 0.030   # solid food height above the floor: at its edge, extra in the middle
RELIEF = 0.006               # depth of the dark creases between gnocchi
FOCAL_PX = 24.0 / 22.3 * 6000

N_THETA, N_R = 320, 150      # inside surface grid (polar)
TEX = 4096                   # colour texture: the photo crop around the bowl
WORK = 4                     # masks and heights are computed on a 1/4-size photo


def srgb(r: int, g: int, b: int) -> list[float]:
    c = np.array([r, g, b], dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return [round(float(v), 5) for v in lin]


MATERIALS = {
    "Dish": dict(color=[1.0, 1.0, 1.0], texture="gnocchi_basecolor.jpg", normal="gnocchi_normal.jpg",
                 rough_tex="gnocchi_roughness.jpg", roughness=1.0, clearcoat=0.0, clearcoat_roughness=0.0),
    "Glaze": dict(color=srgb(236, 236, 232), roughness=0.14, clearcoat=0.5, clearcoat_roughness=0.08),
}


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- #
# Camera fitted to the rim
# --------------------------------------------------------------------------- #
CAM_AXES = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # world x -> image right, z -> image down, -y -> forward


def _rot(rx, rz, ry):
    c, s = np.cos, np.sin
    Rx = np.array([[1, 0, 0], [0, c(rx), -s(rx)], [0, s(rx), c(rx)]])
    Rz = np.array([[c(rz), -s(rz), 0], [s(rz), c(rz), 0], [0, 0, 1]])
    Ry = np.array([[c(ry), 0, s(ry)], [0, 1, 0], [-s(ry), 0, c(ry)]])
    return Rz @ Rx @ Ry


def project_rim(P, cam, size):
    """Photo pixels of points P given in rim units (rim = unit circle at y = 0)."""
    tx, ty, tz, rx, rz, ry = cam
    Q = (P - [tx, ty, tz]) @ (_rot(rx, rz, ry) @ CAM_AXES).T
    w, h = size
    return np.stack([w / 2 + FOCAL_PX * Q[..., 0] / Q[..., 2], h / 2 + FOCAL_PX * Q[..., 1] / Q[..., 2]], -1)


def rim_edge_points(gray):
    """Outer edge of the white rim against the grey table, along rays from the middle."""
    g = ndi.gaussian_filter(gray, 2.0)
    h, w = g.shape
    # rough centre: middle of the bright, unsaturated bowl blob
    cy0, cx0 = h * 0.47, w * 0.62
    pts = []
    for a in np.radians(np.arange(0, 360, 0.5)):
        r = np.arange(0.43 * h, 0.56 * h, 0.5)
        x, y = cx0 + r * np.cos(a), cy0 + r * np.sin(a)
        ok = (x > 3) & (x < w - 4) & (y > 3) & (y < h - 4)
        if ok.sum() < 300:
            continue
        x, y = x[ok], y[ok]
        v = ndi.map_coordinates(g, [y, x], order=1)
        gr = np.convolve(np.gradient(v), np.ones(7) / 7, mode="same")
        cands = []
        for i in np.argsort(gr)[:40]:
            if 40 <= i <= len(v) - 40:
                b, af = v[i - 40:i - 8].mean(), v[i + 8:i + 40].mean()
                if b > 165 and b - af > 22 and v[i - 40:i - 8].std() < 12:
                    cands.append(i)
        if cands:
            pts.append((x[max(cands)], y[max(cands)]))
    return np.array(pts)


def fit_camera(photo_gray):
    pts = rim_edge_points(photo_gray)
    th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    circ = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1)
    size = photo_gray.shape[::-1]

    def resid(p):
        proj = project_rim(circ, p, size)
        return np.linalg.norm(pts[:, None] - proj[None], axis=-1).min(1)

    c = pts.mean(0)
    p0 = [(c[0] - size[0] / 2) / FOCAL_PX * 3.3, 3.3, (c[1] - size[1] / 2) / FOCAL_PX * 3.3, 0, 0, 0]
    sol = least_squares(resid, p0, loss="soft_l1", f_scale=3)
    res = resid(sol.x)
    print(f"camera: {len(pts)} rim points, fit {np.median(res):.1f} px median / {np.percentile(res, 90):.1f} px p90, "
          f"{sol.x[1] * BOWL_R * 100:.0f} cm above the rim")
    return sol.x.tolist()


def load_camera(photo_gray):
    if CAMERA_JSON.exists():
        return json.loads(CAMERA_JSON.read_text())["cam"]
    cam = fit_camera(photo_gray)
    CAMERA_JSON.write_text(json.dumps({"cam": cam, "focal_px": FOCAL_PX, "rim_radius_m": BOWL_R,
                                       "note": "rim units: rim = unit circle at y = 0; see build_gnocchi.py"},
                                      indent=1))
    return cam


# --------------------------------------------------------------------------- #
# Bowl profile
# --------------------------------------------------------------------------- #
# Outer surface (r / BOWL_R, y / BOWL_H), measured from side.jpg, foot to rim.
OUTER = [(0.0, 0.0), (0.44, 0.0), (0.47, 0.004), (0.49, 0.05), (0.52, 0.075), (0.63, 0.28), (0.76, 0.43),
         (0.86, 0.55), (0.93, 0.68), (0.96, 0.80), (0.985, 0.90), (0.997, 0.955)]


def outer_profile(n=120):
    """Outer surface as a polyline (r, y) in metres, from the middle underneath to the rim lip."""
    p = np.array(OUTER) * [BOWL_R, BOWL_H]
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    t = np.linspace(0, s[-1], n)
    return np.stack([PchipInterpolator(s, p[:, 0])(t), PchipInterpolator(s, p[:, 1])(t)], -1)


def lip_profile():
    """Rounded rim lip from the outer surface's top over to the inside."""
    rr = WALL / 2
    c = np.array([BOWL_R - rr, BOWL_H - rr])
    a = np.linspace(0, np.pi, 17)
    return c + rr * np.stack([np.cos(a), np.sin(a)], -1)


def inner_y():
    """Inside surface height y(r) for r in [0, BOWL_R - WALL]: the outer wall offset by
    the wall thickness, on a flat floor."""
    o = outer_profile(400)
    d = np.gradient(o, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    inner = o + WALL * n                                  # inward/up
    keep = (inner[:, 0] > 0.3 * BOWL_R) & (inner[:, 1] > FLOOR_Y + 0.003)
    r = np.concatenate([[0.0, 0.40 * BOWL_R], inner[keep, 0], [BOWL_R - WALL]])
    y = np.concatenate([[FLOOR_Y, FLOOR_Y], inner[keep, 1], [BOWL_H - WALL / 2]])
    order = np.argsort(r)
    r, y = r[order], np.maximum.accumulate(y[order])
    r, idx = np.unique(r, return_index=True)
    f = PchipInterpolator(r, y[idx])
    return lambda q: f(np.clip(q, 0, BOWL_R - WALL))


Y_IN = inner_y()


# --------------------------------------------------------------------------- #
# Food from the overhead photo
# --------------------------------------------------------------------------- #
def food_maps(photo):
    """1/WORK-scale maps in photo space: solid food, sauce film, height above the floor (m)."""
    sm = photo[::WORK, ::WORK].astype(np.float32)
    R, B = sm[..., 0], sm[..., 2]
    mx = sm.max(-1)
    sat = (mx - sm.min(-1)) / np.maximum(mx, 1)
    food = ndi.binary_opening(R - B > 38, iterations=2)
    lab, n = ndi.label(food)
    food = lab == (np.argmax(ndi.sum(food, lab, range(1, n + 1))) + 1)
    food = ndi.binary_fill_holes(ndi.binary_closing(food, iterations=6))
    # px per metre at food height (rim is ~BOWL_R from the middle in the photo)
    ppm = 16000.0 / WORK
    # solid food = what survives an opening ~6 mm across; the rest is a film of sauce
    disk = int(round(0.003 * ppm))
    yy, xx = np.mgrid[-disk:disk + 1, -disk:disk + 1]
    solid = ndi.binary_opening(food, structure=(xx**2 + yy**2) <= disk**2)
    solid = ndi.binary_fill_holes(solid)
    d = ndi.distance_transform_edt(solid) / ppm                        # metres inside the solid food
    lum = sm @ np.array([0.299, 0.587, 0.114], np.float32)
    # the dark creases between gnocchi sink in; the lit tops stay full and pillowy
    crease = ndi.gaussian_filter(lum, 0.0008 * ppm) - ndi.gaussian_filter(lum, 0.008 * ppm)
    crease = ndi.gaussian_filter(np.clip(-crease / 35.0, 0, 1), 0.0022 * ppm)
    walnut = (mx < 135) & (R - B > 40) & (sat > 0.5) & solid
    walnut = ndi.gaussian_filter(ndi.binary_opening(walnut).astype(np.float32), 0.0012 * ppm)
    height = (FOOD_EDGE + FOOD_PEAK * smoothstep(0.0, 0.055, d) - RELIEF * crease + 0.003 * np.clip(walnut * 2, 0, 1))
    soft = ndi.gaussian_filter(solid.astype(np.float32), 0.0012 * ppm)
    edge = np.sqrt(smoothstep(0.0, 0.008, d)) * 0.9 + 0.1     # rounded, like a gnocchi's side
    height = ndi.gaussian_filter(height, 0.0006 * ppm) * soft * edge
    film = ndi.gaussian_filter((food & ~solid).astype(np.float32), 0.001 * ppm)
    return {"solid": soft, "film": film, "height": height, "food": food}


def sample(img, uv_px):
    """Bilinear sample of a 1/WORK map at full-resolution photo pixels."""
    q = uv_px / WORK - 0.5
    return ndi.map_coordinates(img, [q[..., 1], q[..., 0]], order=1, mode="nearest")


# --------------------------------------------------------------------------- #
# Meshes
# --------------------------------------------------------------------------- #
def grid_normals(X):
    du = np.roll(X, -1, 0) - np.roll(X, 1, 0)
    dv = np.gradient(X, axis=1)
    N = np.cross(dv, du)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
    N *= np.sign(N[..., 1:2].sum(0, keepdims=True) + 1e-9)
    return N


def triangulate(nc, nr, closed=True):
    idx = np.arange(nc * nr).reshape(nc, nr)
    if closed:
        idx = np.concatenate([idx, idx[:1]])
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    return np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])


def orient(F, V, N):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    keep = np.linalg.norm(fn, axis=1) > 1e-14
    F, fn = F[keep], fn[keep]
    if (fn * N[F].mean(1)).sum(1).mean() < 0:
        F = F[:, ::-1].copy()
    return F


def tangents(V, N, UV, F):
    """Per-vertex tangents from UV gradients; w makes cross(N, T) * w point toward -v."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    t0, t1, t2 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-20, 1.0 / np.where(det == 0, 1, det), 0.0)[:, None]
    T = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    Bv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    Tv, Bs = np.zeros_like(V), np.zeros_like(V)
    for k in range(3):
        np.add.at(Tv, F[:, k], T)
        np.add.at(Bs, F[:, k], Bv)
    Tv -= (Tv * N).sum(-1, keepdims=True) * N
    bad = np.linalg.norm(Tv, axis=-1) < 1e-12
    Tv[bad] = np.cross(N[bad], [0, 0, 1.0]) + 1e-6
    Tv /= np.linalg.norm(Tv, axis=-1, keepdims=True)
    w = -np.sign((np.cross(N, Tv) * Bs).sum(-1))
    w[w == 0] = 1.0
    return np.concatenate([Tv, w[:, None]], -1)


def build_inside(cam, maps, size, crop):
    """Inside of the bowl + the food, one height field on a polar grid, UVs from the photo."""
    r_max = BOWL_R - WALL / 2
    rr = r_max * np.linspace(0, 1, N_R) ** 0.85
    th = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    T, Rr = np.meshgrid(th, rr, indexing="ij")
    x, z = Rr * np.cos(T), Rr * np.sin(T)
    y_bowl = Y_IN(Rr)
    y_bowl = np.where(Rr > BOWL_R - WALL, BOWL_H - WALL / 2 + 0 * Rr, y_bowl)

    def to_px(x, y, z):
        P = np.stack(np.broadcast_arrays(x, y - BOWL_H, z), -1) / BOWL_R
        return project_rim(P, cam, size)

    guess = to_px(x, FLOOR_Y + 0.025, z)
    h = sample(maps["height"], guess)
    film = sample(maps["film"], guess)
    solid = sample(maps["solid"], guess)
    y_food = FLOOR_Y + h
    y = np.maximum(y_bowl + 0.0012 * film + 0.002 * solid, np.where(solid > 0.02, y_food, 0))
    X = np.stack([x, y, z], -1)
    N = grid_normals(X)
    N[:, 0] = [0, 1, 0] if solid[0, 0] < 0.5 else N[:, 1].mean(0)
    px = to_px(x, y, z)
    (x0, y0, x1, y1) = crop
    UV = np.stack([(px[..., 0] - x0) / (x1 - x0), (px[..., 1] - y0) / (y1 - y0)], -1)
    F = triangulate(N_THETA, N_R)
    V, Nf, UVf = X.reshape(-1, 3), N.reshape(-1, 3), UV.reshape(-1, 2)
    F = orient(F, V, Nf)
    Tg = tangents(V, Nf, UVf, F)
    return {"V": V.astype(np.float32), "N": Nf.astype(np.float32), "F": F,
            "UV": UVf.astype(np.float32), "T": Tg.astype(np.float32)}


def build_outside():
    """Glazed outside: underneath, foot, wall and the rim lip (plain white)."""
    prof = np.concatenate([outer_profile(70), lip_profile()[1:]])
    n_th = 192
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    X = np.stack([prof[None, :, 0] * np.cos(th)[:, None], np.broadcast_to(prof[:, 1], (n_th, len(prof))),
                  prof[None, :, 0] * np.sin(th)[:, None]], -1)
    d = np.gradient(prof, axis=0)
    nr = np.stack([d[:, 1], -d[:, 0]], -1)
    nr /= np.maximum(np.linalg.norm(nr, axis=-1, keepdims=True), 1e-12)
    nr[0] = [0, -1]
    N = np.stack([nr[None, :, 0] * np.cos(th)[:, None], np.broadcast_to(nr[:, 1], (n_th, len(prof))),
                  nr[None, :, 0] * np.sin(th)[:, None]], -1)
    F = triangulate(n_th, len(prof))
    V, Nf = X.reshape(-1, 3), N.reshape(-1, 3)
    return {"V": V.astype(np.float32), "N": Nf.astype(np.float32), "F": orient(F, V, Nf)}


# --------------------------------------------------------------------------- #
# Textures
# --------------------------------------------------------------------------- #
def textures(photo, maps, crop):
    x0, y0, x1, y1 = crop
    h, w = photo.shape[:2]
    pad = max(0, -y0, -x0, y1 - h, x1 - w)
    src = np.pad(photo, ((pad, pad), (pad, pad), (0, 0)), mode="edge") if pad else photo
    region = src[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    base = Image.fromarray(region).resize((TEX, TEX), Image.LANCZOS)

    # bump + roughness at half the colour resolution, from the same pixels
    half = np.asarray(base.resize((TEX // 2, TEX // 2), Image.LANCZOS)).astype(np.float32)
    lum = half @ np.array([0.299, 0.587, 0.114], np.float32)
    sx = (x1 - x0) / (TEX // 2)
    food_m = np.asarray(Image.fromarray((maps["food"] * 255).astype(np.uint8)).resize(
        (w // WORK, h // WORK))).astype(np.float32) / 255
    yy, xx = np.mgrid[0:TEX // 2, 0:TEX // 2]
    fx, fy = (x0 + (xx + 0.5) * sx) / WORK - 0.5, (y0 + (yy + 0.5) * sx) / WORK - 0.5
    food = ndi.map_coordinates(ndi.gaussian_filter(food_m, 2), [fy, fx], order=1, mode="nearest")
    detail = (lum - ndi.gaussian_filter(lum, 4.0)) / 255.0
    m_per_texel = sx / 16000.0
    hgt = detail * 0.0006 * food
    gy, gx = np.gradient(hgt, m_per_texel)
    n = np.stack([-gx, gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal = Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8))

    R, B = half[..., 0], half[..., 2]
    mx = half.max(-1)
    sat = (mx - half.min(-1)) / np.maximum(mx, 1)
    walnut = smoothstep(150, 110, mx) * smoothstep(0.4, 0.6, sat)
    sauce = smoothstep(0.75, 0.45, sat) * smoothstep(20, 60, R - B)
    # the photo already holds the dish's highlights: keep the food fairly matte
    rough = 0.12 * (1 - food) + food * (0.68 - 0.12 * sauce + 0.2 * walnut)
    rough_img = np.zeros((TEX // 2, TEX // 2, 3), np.uint8)
    rough_img[..., 0] = 255
    rough_img[..., 1] = (np.clip(rough, 0.08, 0.9) * 255).astype(np.uint8)
    return base, normal, Image.fromarray(rough_img).resize((TEX // 4, TEX // 4), Image.BILINEAR)


# --------------------------------------------------------------------------- #
def main():
    if not PHOTO.exists():
        sys.exit(f"missing reference photo {PHOTO} (see README)")
    photo = np.asarray(Image.open(PHOTO).convert("RGB"))
    gray = photo @ np.array([0.299, 0.587, 0.114])
    cam = load_camera(gray.astype(np.float32))
    size = (photo.shape[1], photo.shape[0])

    # square photo crop holding the whole rim
    th = np.linspace(0, 2 * np.pi, 720)
    rim = project_rim(np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1) * 1.01, cam, size)
    cx, cy = rim.mean(0)
    half = np.abs(rim - [cx, cy]).max() + 8
    crop = tuple(int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))

    maps = food_maps(photo)
    inside = build_inside(cam, maps, size, crop) | {"name": "Dish", "material": "Dish"}
    outside = build_outside() | {"name": "Bowl", "material": "Glaze"}
    base, normal, rough = textures(photo, maps, crop)

    OUT_DIR.mkdir(exist_ok=True)
    tex = {}
    for name, im, q in (("gnocchi_basecolor.jpg", base, 88), ("gnocchi_normal.jpg", normal, 90),
                        ("gnocchi_roughness.jpg", rough, 90)):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True)
        tex[name] = buf.getvalue()
    meshes = [outside, inside]
    # Turn the dish so its front (+z, where viewers start) faces the photo's camera:
    # that side was seen best, so it looks best.
    al = np.arctan2(cam[0], cam[2])
    rot = np.array([[np.cos(al), 0, -np.sin(al)], [0, 1, 0], [np.sin(al), 0, np.cos(al)]])
    for m in meshes:
        m["V"], m["N"] = (m["V"] @ rot.T).astype(np.float32), (m["N"] @ rot.T).astype(np.float32)
        if "T" in m:
            m["T"][:, :3] = m["T"][:, :3] @ rot.T
    write_glb(OUT_DIR / "gnocchi.glb", meshes, tex, MATERIALS, "Gnocchi", generator="ar-menu-poc/build_gnocchi.py")
    write_usdz(OUT_DIR / "gnocchi.usdz", meshes, tex, MATERIALS, "Gnocchi")

    allV = np.vstack([m["V"] for m in meshes])
    lo, hi = allV.min(0), allV.max(0)
    print(f"triangles      : {sum(len(m['F']) for m in meshes):,}   vertices: {sum(len(m['V']) for m in meshes):,}")
    print(f"overall        : {(hi - lo)[0] * 100:.1f} x {(hi - lo)[2] * 100:.1f} cm, {(hi - lo)[1] * 100:.1f} cm tall "
          f"(food peak {(inside['V'][:, 1].max() - BOWL_H) * 1000:+.0f} mm vs the rim)")
    print("textures       : " + ", ".join(f"{k} {len(v) / 1e6:.2f} MB" for k, v in tex.items()))
    for f in ("gnocchi.glb", "gnocchi.usdz"):
        print(f"{f:<15}: {(OUT_DIR / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
