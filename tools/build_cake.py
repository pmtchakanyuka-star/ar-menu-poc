"""Build the real-size, photo-textured 3D cake for the AR menu.

Pipeline (run tools/calib/detect_edges.py + fit_camera.py first):
  1. Shape: the cake profile and camera fitted to the reference photo
     (tools/calib/calibration.json), scaled so the cake is 26 cm across the base.
     The glaze pool's wavy outline on the plate is measured from the photo too.
  2. Texture: the photo is projected onto the shape through the fitted camera.
     Every surface point the camera saw gets its real pixels; points it could not
     see (the back, the near wall of the hole) are mirrored in from the closest
     well-seen part of the same height band - the cake is round, so that is
     the same kind of surface under similar light.
  3. Export: models/cake.glb (web / Android) and models/cake.usdz (iPhone).

World frame: Y up, metres, origin at the centre of the plate's underside.
Run:  .venv/Scripts/python tools/build_cake.py
"""
from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "tools" / "calib"
sys.path.insert(0, str(CALIB))
import fit_camera as fc  # noqa: E402  (shared profile + camera model)

OUT_DIR = ROOT / "models"
CAL = json.loads((CALIB / "calibration.json").read_text())
EDGES = json.loads((CALIB / "edges.json").read_text())
PHOTO = CALIB / "reference.jpg"

PLATE_TOP = 0.012            # plate well surface above the table
N_THETA = 192                # mesh columns around
N_CAKE = 170                 # mesh rows, hole centre -> glaze foot
N_POOL = 16                  # mesh rows, glaze pooled on the plate
TEX_W, TEX_H = 2048, 1024    # base-colour texture (u = around, v = along the profile)
GOOD_COS = 0.42              # photo pixels count as "well seen" below ~65 deg incidence
SEG_DEG, BLEND_DEG = 40.0, 2.5  # unseen arcs get 40-deg patches of seen ones, 2.5-deg crossfades


def srgb(r: int, g: int, b: int) -> list[float]:
    c = np.array([r, g, b], dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return [round(float(v), 5) for v in lin]


MATERIALS = {
    "Cake": dict(color=[1.0, 1.0, 1.0], texture="cake_basecolor.jpg", normal="cake_normal.jpg",
                 rough_tex="cake_roughness.jpg", roughness=1.0, clearcoat=0.0, clearcoat_roughness=0.0),
    "Plate": dict(color=srgb(236, 230, 220), roughness=0.16, clearcoat=0.6, clearcoat_roughness=0.08),
}


# --------------------------------------------------------------------------- #
# Calibrated camera + shape
# --------------------------------------------------------------------------- #
def params() -> dict:
    p = {k: CAL[k] for k in "Rf Hf Rh Hh Rc Hc Ro Ho Rb Rq rim_h".split()}
    p.update(f=CAL["focal_px"], yaw=CAL["yaw"], pitch=CAL["pitch"], roll=CAL["roll"],
             cam_h=CAL["camera_position"][1], cam_d=CAL["camera_position"][2])
    return p


P = params()
CAM = fc.camera(P)               # calibration frame: origin at the plate's top surface
F_PX, CAM_C, CAM_R = CAM
PLATE_R = CAL["plate_rim_radius"]


def pool_radius_fn():
    """Glaze-pool outline r(theta): photo edge points back-projected onto the plate,
    fitted with a smooth periodic curve (regularised so the unseen back stays tame)."""
    pts = np.array([pt[:2] for pt in EDGES["pool_edge"]])
    d = np.stack([(pts[:, 0] - fc.CX) / F_PX, (pts[:, 1] - fc.CY) / F_PX, np.ones(len(pts))], -1) @ CAM_R
    t = (0.0015 - CAM_C[1]) / d[:, 1]
    X = CAM_C + t[:, None] * d
    th, r = np.arctan2(X[:, 0], X[:, 2]), np.hypot(X[:, 0], X[:, 2])
    K = 7
    k = np.arange(1, K + 1)
    A = np.hstack([np.ones((len(th), 1)), np.cos(np.outer(th, k)), np.sin(np.outer(th, k))])
    lam = 2e-4 * np.concatenate([[0.0], k**2, k**2])
    coef = np.linalg.solve(A.T @ A + np.diag(lam) * len(th), A.T @ r)
    lo, hi = P["Rb"] + 0.012, PLATE_R - 0.009

    def fn(theta):
        th_ = np.asarray(theta, float)
        B = np.concatenate([np.ones(th_.shape + (1,)), np.cos(th_[..., None] * k), np.sin(th_[..., None] * k)], -1)
        return np.clip(B @ coef, lo, hi)

    print(f"pool edge from photo: {len(r)} points, r = {r.min() * 100:.1f}-{r.max() * 100:.1f} cm")
    return fn


POOL_R = pool_radius_fn()
CAKE_PROF = fc.smooth_resample(fc.control_points(P), N_CAKE)  # (r, y) above the plate top


def profiles(theta):
    """Per-angle profile (r, y) for the whole glaze surface: cake + pool. Shape (n, N_CAKE+N_POOL, 2)."""
    theta = np.atleast_1d(theta)
    n = len(theta)
    r_foot, y_foot = CAKE_PROF[-1]
    edge = POOL_R(theta)                           # (n,)
    depth, bead = 0.0034, 0.0045
    n_flat = N_POOL - 8
    t = np.linspace(0, 1, n_flat + 2)[1:-1]
    flat_r = r_foot + np.outer(edge - bead - r_foot, t)
    flat_y = np.broadcast_to(depth + (y_foot - depth) * (1 - t) ** 2, flat_r.shape)
    phi = np.linspace(0, np.pi / 2, 7)
    bead_r = (edge - bead)[:, None] + bead * np.sin(phi)
    bead_y = np.broadcast_to(depth * np.cos(phi), bead_r.shape)
    tuck_r, tuck_y = (edge - 0.0004)[:, None], np.full((n, 1), -0.0005)
    r = np.hstack([np.broadcast_to(CAKE_PROF[:, 0], (n, N_CAKE)), flat_r, bead_r, tuck_r])
    y = np.hstack([np.broadcast_to(CAKE_PROF[:, 1], (n, N_CAKE)), flat_y, bead_y, tuck_y])
    return np.stack([r, y], -1)


def profile_normals(prof):
    """Outward/up unit normals (nr, ny) of profile polylines (..., m, 2)."""
    d = np.gradient(prof, axis=-2)
    n = np.stack([-d[..., 1], d[..., 0]], -1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def revolve(theta, prof, nrm):
    """3D points/normals for angle array theta (n,) and per-angle profiles (n, m, 2)."""
    s, c = np.sin(theta)[:, None], np.cos(theta)[:, None]
    X = np.stack([prof[..., 0] * s, prof[..., 1], prof[..., 0] * c], -1)
    N = np.stack([nrm[..., 0] * s, nrm[..., 1], nrm[..., 0] * c], -1)
    return X, N


# --------------------------------------------------------------------------- #
# Texture: project the photo onto the shape
# --------------------------------------------------------------------------- #
def occluded(X):
    """True where the straight line from X to the camera passes through the cake/pool."""
    pr, py = CAKE_PROF[:, 0], CAKE_PROF[:, 1]
    pr = np.concatenate([pr, [POOL_R(0.0).item(), PLATE_R]])
    py = np.concatenate([py, [0.0034, -0.01]])
    occ = np.zeros(len(X), bool)
    for t in np.linspace(0.004, 0.5, 120):          # the occluders are all near the cake
        Q = X + t * (CAM_C - X)
        surf = np.interp(np.hypot(Q[:, 0], Q[:, 2]), pr, py)
        occ |= Q[:, 1] < surf - 2e-4
    return occ


def seen_arcs(rows):
    """Per profile row, the largest arc [a, a+L] the photo sees head-on enough."""
    gt = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    prof = profiles(gt)
    X, N = revolve(gt, prof, profile_normals(prof))
    Xf, Nf = X.reshape(-1, 3), N.reshape(-1, 3)
    V = CAM_C - Xf
    cosv = (Nf * V).sum(-1) / np.linalg.norm(V, axis=-1)
    cosv = cosv.reshape(len(gt), rows)
    # A row's bar is relative to how well the photo saw that row at best: the crest,
    # seen nearly head-on in front, shouldn't keep its stretched far side.
    bar = np.maximum(GOOD_COS, 0.7 * cosv.max(axis=0))
    good = (cosv > bar) & ~occluded(Xf).reshape(len(gt), rows)
    A, L = np.zeros(rows), np.zeros(rows)
    step = gt[1] - gt[0]
    for j in range(rows):
        g = good[:, j]
        if g.all():
            A[j], L[j] = 0.0, 2 * np.pi
            continue
        if not g.any():
            continue
        start = int(np.argmin(g))                     # begin the run search at a gap
        gg = np.roll(g, -start)
        best, run, best_end = 0, 0, 0
        for i, v in enumerate(gg):
            run = run + 1 if v else 0
            if run > best:
                best, best_end = run, i
        A[j] = gt[(best_end - best + 1 + start) % len(gt)] + 2 * step
        L[j] = max((best - 4) * step, 0.0)
    ok = L >= np.radians(20)        # narrower rows borrow a neighbour; wider ones squeeze patches in
    src_row = np.arange(rows)
    cand = np.nonzero(ok)[0]
    for j in np.nonzero(~ok)[0]:
        src_row[j] = cand[np.argmin(np.abs(cand - j))]
    A, L = A[src_row], L[src_row]
    full = L >= 2 * np.pi - 1e-6
    k = np.ones(9) / 9
    As = np.convolve(np.pad(np.unwrap(A), 4, mode="edge"), k, mode="valid")
    Ls = np.convolve(np.pad(L, 4, mode="edge"), k, mode="valid")
    A = np.where(full, A, As + 0.5 * (L - np.minimum(Ls, L)))   # smoothed arc stays inside the seen one
    L = np.where(full, L, np.minimum(Ls, L))
    print(f"texture: {ok.mean() * 100:.0f}% of profile rows seen head-on; {full.mean() * 100:.0f}% all the way "
          f"round; median seen arc {np.degrees(np.median(L)):.0f} deg")
    return A, L, src_row


def sample_photo(theta, kf, photo):
    """Photo colour at surface points (theta, fractional profile row); both (W, H) arrays."""
    rows = N_CAKE + N_POOL
    k0 = np.clip(np.floor(kf).astype(int), 0, rows - 2)
    w = (kf - k0)[..., None]
    out = np.zeros(theta.shape + (3,), np.float32)
    for c0 in range(0, theta.shape[0], 128):          # column blocks bound the memory use
        th = theta[c0:c0 + 128]
        prof = profiles(th.ravel()).reshape(th.shape + (rows, 2))
        bi, hi = np.indices(th.shape)
        kk = k0[c0:c0 + 128]
        pr = prof[bi, hi, kk] * (1 - w[c0:c0 + 128]) + prof[bi, hi, kk + 1] * w[c0:c0 + 128]
        Xs = np.stack([pr[..., 0] * np.sin(th), pr[..., 1], pr[..., 0] * np.cos(th)], -1).reshape(-1, 3)
        xy = fc.project(Xs, CAM)
        for ch in range(3):
            out[c0:c0 + 128, :, ch] = map_coordinates(photo[..., ch], [xy[:, 1], xy[:, 0]], order=1,
                                                      mode="nearest").reshape(th.shape)
    return out


def bake_textures():
    """Base colour, tangent-space normal and roughness maps for the cake surface."""
    from scipy.ndimage import gaussian_filter, gaussian_filter1d

    photo = np.asarray(Image.open(PHOTO).convert("RGB")).astype(np.float32)
    rows = N_CAKE + N_POOL
    A, L, src_row = seen_arcs(rows)

    u = (np.arange(TEX_W) + 0.5) / TEX_W
    theta = 2 * np.pi * u                                            # (W,)
    kf = ((np.arange(TEX_H) + 0.5) / TEX_H) * (rows - 1)             # (H,)
    rowj = np.clip(np.round(kf).astype(int), 0, rows - 1)
    kf = np.where(src_row[rowj] == rowj, kf, src_row[rowj].astype(float))
    a, Lr = A[rowj][None, :], L[rowj][None, :]                       # (1, H)

    # 1) Every texel straight from the photo (only trusted inside its row's seen arc).
    T = sample_photo(np.broadcast_to(theta[:, None], (TEX_W, TEX_H)), np.broadcast_to(kf, (TEX_W, TEX_H)), photo)
    T = np.ascontiguousarray(np.transpose(T, (1, 0, 2)))             # (H, W, 3)
    delta = np.mod(theta[:, None] - a, 2 * np.pi)                    # (W, H) angle into the arc
    full = Lr >= 2 * np.pi - 1e-6
    inside = (full | (delta <= Lr)).T                                # (H, W)

    # 2) Even out the photo's lighting around each row (keep each row's average):
    #    copied patches then match, and the viewer's own lights do the shading.
    lum = T @ np.array([0.299, 0.587, 0.114], np.float32)
    sig = TEX_W * 14 / 360
    wsum = gaussian_filter1d(inside.astype(np.float32), sig, axis=1, mode="wrap")
    lp = gaussian_filter1d(lum * inside, sig, axis=1, mode="wrap") / np.maximum(wsum, 1e-3)
    rowmean = (lum * inside).sum(1, keepdims=True) / np.maximum(inside.sum(1, keepdims=True), 1)
    gain = np.clip(rowmean / np.maximum(lp, 1.0), 0.6, 1.7)
    T = T * gain[..., None]

    # 3) Fill what the photo didn't see head-on with patches of what it did.
    rng = np.random.default_rng(11)
    nseg = int(round(360 / SEG_DEG))
    seg, bw = 2 * np.pi / nseg, np.radians(BLEND_DEG)
    frac = rng.random(nseg)
    margin = np.radians(3.0)
    room = np.maximum(Lr - 2 * margin - 2 * bw - seg, 0.0)          # (1, H)
    squeeze = np.minimum(1.0, np.maximum(Lr - 2 * margin - 2 * bw, 1e-3) / seg)

    def patch_src(k, ell):
        return a + margin + bw + frac[k % nseg] * room + ell * squeeze

    kseg = np.floor(theta / seg).astype(int)[:, None]                # (W, 1)
    ell = theta[:, None] - kseg * seg                                 # (W, 1)
    w_prev = np.clip(0.5 - ell / (2 * bw), 0, 0.5)                   # crossfade into the previous patch
    w_next = np.clip(0.5 - (seg - ell) / (2 * bw), 0, 0.5)           # ... and the next one

    def take(src_theta):
        cols = np.mod(src_theta, 2 * np.pi) / (2 * np.pi) * TEX_W - 0.5     # (W, H)
        rr = np.broadcast_to(np.arange(TEX_H)[None, :], cols.shape)
        return np.stack([map_coordinates(T[..., c], [rr.T, cols.T], order=1, mode="grid-wrap")
                         for c in range(3)], -1)

    w_mid = (1 - w_prev - w_next)
    patches = (take(patch_src(kseg, ell)) * w_mid.T[..., None]
               + take(patch_src(kseg - 1, ell + seg)) * w_prev.T[..., None]
               + take(patch_src(kseg + 1, ell - seg)) * w_next.T[..., None])
    d_in = np.where(delta <= Lr, np.minimum(delta, Lr - delta), -np.minimum(delta - Lr, 2 * np.pi - delta))
    wd = np.where(full, 1.0, np.clip(0.5 + d_in / (2 * bw), 0, 1)).T[..., None]
    base = T * wd + patches * (1 - wd)

    # 4) The glossy glaze mirrored the green cake carrier: neutralise green casts.
    r, g = base[..., 0], base[..., 1]
    base[..., 1] = np.where(g > r * 0.92, r * 0.92 + 0.15 * (g - r * 0.92), g)
    base = np.clip(base, 0, 255)

    # 5) Relief + finish from the photo detail: sprinkles are lighter than the glaze gaps.
    lum = base @ np.array([0.299, 0.587, 0.114], np.float32)
    detail = (lum - gaussian_filter(lum, 3.0, mode=("nearest", "wrap"))) / 255.0
    prof0 = profiles(np.array([0.0]))[0]
    seglen = np.linalg.norm(np.diff(prof0, axis=0), axis=1)
    s_rows = np.interp(np.arange(TEX_H + 1) / TEX_H * (rows - 1), np.arange(rows),
                       np.concatenate([[0], np.cumsum(seglen)]))
    dv = np.maximum(np.diff(s_rows), 2e-4)[:, None]                  # metres per texel row
    r_tex = np.interp((np.arange(TEX_H) + 0.5) / TEX_H * (rows - 1), np.arange(rows), prof0[:, 0])
    du = np.maximum(2 * np.pi * r_tex / TEX_W, 2e-4)[:, None]       # metres per texel column
    h = detail * 0.0035                                               # ~0.5 mm bumps at full contrast
    dhdu = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * du)
    dhdv = (np.vstack([h[1:], h[-1:]]) - np.vstack([h[:1], h[:-1]])) / (2 * dv)
    n = np.stack([-dhdu, dhdv, np.ones_like(h)], -1)                 # +x = +u, +y = up = -v (glTF)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal = ((n * 0.5 + 0.5) * 255).astype(np.uint8)

    # Glossy where the glaze pools (hole floor, plate), satin over the sprinkles,
    # glossier in the dark glaze gaps between sprinkles.
    rr0 = prof0[:, 0]
    region = np.select([rr0 < CAL["Rf"] * 0.98, rr0 < CAL["Rh"] + 0.002, rr0 > CAL["Rb"] + 0.004],
                       [0.20, 0.40, 0.22], default=0.58)
    region = gaussian_filter1d(region, 2.0)
    reg_tex = np.interp((np.arange(TEX_H) + 0.5) / TEX_H * (rows - 1), np.arange(rows), region)[:, None]
    rough = np.clip(reg_tex + 1.2 * detail, 0.12, 0.85)
    rough_img = np.zeros((TEX_H, TEX_W, 3), np.uint8)
    rough_img[..., 0] = 255
    rough_img[..., 1] = (rough * 255).astype(np.uint8)                # glTF: G = roughness, B = metallic

    return (Image.fromarray(base.astype(np.uint8)), Image.fromarray(normal),
            Image.fromarray(rough_img).resize((TEX_W // 2, TEX_H // 2), Image.BILINEAR))


# --------------------------------------------------------------------------- #
# Meshes
# --------------------------------------------------------------------------- #
def grid_mesh(X, N, U=None, T=None):
    """Triangulate an (n_cols, n_rows) vertex grid (columns already include the seam copy)."""
    nc, nr = X.shape[:2]
    idx = np.arange(nc * nr).reshape(nc, nr)
    a, b = idx[:-1, :-1], idx[1:, :-1]
    c, d = idx[1:, 1:], idx[:-1, 1:]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    Vf, Nf = X.reshape(-1, 3), N.reshape(-1, 3)
    area = np.linalg.norm(np.cross(Vf[F[:, 1]] - Vf[F[:, 0]], Vf[F[:, 2]] - Vf[F[:, 0]]), axis=1)
    F = F[area > 1e-12]                                   # drop the degenerate pole triangles
    fn = np.cross(Vf[F[:, 1]] - Vf[F[:, 0]], Vf[F[:, 2]] - Vf[F[:, 0]])
    if (fn * Nf[F].mean(axis=1)).sum(axis=1).mean() < 0:
        F = F[:, ::-1].copy()
    mesh = {"V": Vf.astype(np.float32), "N": Nf.astype(np.float32), "F": F}
    if U is not None:
        mesh["UV"] = U.reshape(-1, 2).astype(np.float32)
    if T is not None:
        mesh["T"] = T.reshape(-1, 4).astype(np.float32)
    return mesh


def build_cake_mesh():
    theta = np.linspace(0, 2 * np.pi, N_THETA + 1)       # last column = seam copy of the first
    prof = profiles(theta)
    nrm = profile_normals(prof)
    nrm[:, 0] = [0.0, 1.0]                                # pole: straight up
    X, N = revolve(theta, prof, nrm)
    X[..., 1] += PLATE_TOP
    rows = prof.shape[1]
    # u = around; v = fractional profile row (the texture's rows are sampled the same way)
    U = np.stack(np.broadcast_arrays(theta[:, None] / (2 * np.pi), (np.arange(rows) / (rows - 1))[None, :]), -1)
    # Tangents point around the cake (+u); w = +1 makes cross(N, T) point toward -v,
    # i.e. "up" in the normal map, per the glTF convention.
    T = np.zeros(X.shape[:2] + (4,))
    T[..., 0], T[..., 2], T[..., 3] = np.cos(theta)[:, None], -np.sin(theta)[:, None], 1.0
    return grid_mesh(X, N, U, T)


def build_plate_mesh():
    R = PLATE_R
    well = POOL_R(np.linspace(0, 2 * np.pi, 360)).max() + 0.004
    pts = [(0.0, PLATE_TOP), (0.5 * well, PLATE_TOP), (well, PLATE_TOP),
           (well + 0.35 * (R - well), PLATE_TOP + 0.0012), (R - 0.0035, PLATE_TOP + 0.0038),
           (R - 0.0012, PLATE_TOP + 0.0047), (R, PLATE_TOP + 0.0036), (R - 0.0006, PLATE_TOP + 0.0018),
           (R - 0.006, PLATE_TOP - 0.0010), (R - 0.03, PLATE_TOP - 0.0055), (0.62 * R, 0.0035),
           (0.60 * R, 0.0004), (0.59 * R, 0.0), (0.55 * R, 0.0), (0.54 * R, 0.0012), (0.50 * R, 0.0032),
           (0.0, 0.0036)]
    prof = fc.smooth_resample(pts, 96)
    theta = np.linspace(0, 2 * np.pi, 161)
    profs = np.broadcast_to(prof, (len(theta),) + prof.shape)
    nrm = profile_normals(profs).copy()
    nrm[:, 0], nrm[:, -1] = [0.0, 1.0], [0.0, -1.0]
    X, N = revolve(theta, profs, nrm)
    return grid_mesh(X, N)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_glb(path: Path, meshes: list[dict], textures: dict[str, bytes]):
    mat_names = list(MATERIALS)
    binbuf = bytearray()
    views, accessors, gl_meshes, nodes, images = [], [], [], [], []

    def add_view(data: bytes, target: int | None) -> int:
        while len(binbuf) % 4:
            binbuf.append(0)
        view = {"buffer": 0, "byteOffset": len(binbuf), "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        binbuf.extend(data)
        return len(views) - 1

    for m in meshes:
        pos, nrm = m["V"], m["N"]
        idx = m["F"].astype(np.uint16 if len(pos) < 65536 else np.uint32).ravel()
        attrs = {"POSITION": len(accessors)}
        accessors.append({"bufferView": add_view(pos.tobytes(), 34962), "componentType": 5126, "count": len(pos),
                          "type": "VEC3", "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()})
        attrs["NORMAL"] = len(accessors)
        accessors.append({"bufferView": add_view(nrm.tobytes(), 34962), "componentType": 5126,
                          "count": len(nrm), "type": "VEC3"})
        if "T" in m:
            attrs["TANGENT"] = len(accessors)
            accessors.append({"bufferView": add_view(m["T"].tobytes(), 34962), "componentType": 5126,
                              "count": len(m["T"]), "type": "VEC4"})
        if "UV" in m:
            attrs["TEXCOORD_0"] = len(accessors)
            accessors.append({"bufferView": add_view(m["UV"].tobytes(), 34962), "componentType": 5126,
                              "count": len(m["UV"]), "type": "VEC2"})
        accessors.append({"bufferView": add_view(idx.tobytes(), 34963),
                          "componentType": 5123 if idx.dtype == np.uint16 else 5125,
                          "count": len(idx), "type": "SCALAR"})
        gl_meshes.append({"name": m["name"], "primitives": [{
            "attributes": attrs, "indices": len(accessors) - 1,
            "material": mat_names.index(m["material"]), "mode": 4}]})
        nodes.append({"name": m["name"], "mesh": len(gl_meshes) - 1})

    tex_index = {}
    for name, data in textures.items():
        images.append({"name": name, "bufferView": add_view(data, None), "mimeType": "image/jpeg"})
        tex_index[name] = len(images) - 1

    materials = []
    for name, p in MATERIALS.items():
        pbr = {"baseColorFactor": p["color"] + [1.0], "metallicFactor": 0.0, "roughnessFactor": p["roughness"]}
        if p.get("texture"):
            pbr["baseColorTexture"] = {"index": tex_index[p["texture"]]}
        if p.get("rough_tex"):
            pbr["metallicRoughnessTexture"] = {"index": tex_index[p["rough_tex"]]}
        mat = {"name": name, "pbrMetallicRoughness": pbr}
        if p.get("normal"):
            mat["normalTexture"] = {"index": tex_index[p["normal"]], "scale": 1.0}
        if p["clearcoat"] > 0:
            mat["extensions"] = {"KHR_materials_clearcoat": {
                "clearcoatFactor": p["clearcoat"], "clearcoatRoughnessFactor": p["clearcoat_roughness"]}}
        materials.append(mat)

    binbuf.extend(b"\0" * (-len(binbuf) % 4))
    gltf = {
        "asset": {"version": "2.0", "generator": "ar-menu-poc/build_cake.py"},
        "extensionsUsed": ["KHR_materials_clearcoat"],
        "scene": 0, "scenes": [{"name": "Scene", "nodes": [0]}],
        "nodes": [{"name": "ChocolateRingCake", "children": list(range(1, len(nodes) + 1))}] + nodes,
        "meshes": gl_meshes, "materials": materials, "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": len(binbuf)}],
    }
    if images:
        gltf["images"] = images
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 33071}]
        gltf["textures"] = [{"sampler": 0, "source": i} for i in range(len(images))]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binbuf)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binbuf), 0x004E4942) + bytes(binbuf))


def write_usdz(path: Path, meshes: list[dict], textures: dict[str, bytes]):
    import tempfile

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    work = Path(tmp.name)
    (work / "textures").mkdir()
    for name, data in textures.items():
        (work / "textures" / name).write_bytes(data)
    usdc = work / "cake.usdc"
    stage = Usd.Stage.CreateNew(str(usdc))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/ChocolateRingCake")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/ChocolateRingCake/Materials")

    mats = {}
    for name, p in MATERIALS.items():
        mp = f"/ChocolateRingCake/Materials/{name}"
        mat = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp + "/PreviewSurface")
        sh.CreateIdAttr("UsdPreviewSurface")
        if p.get("texture"):
            st = UsdShade.Shader.Define(stage, mp + "/stReader")
            st.CreateIdAttr("UsdPrimvarReader_float2")
            st.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
            tex = UsdShade.Shader.Define(stage, mp + "/BaseColorTexture")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"textures/{p['texture']}")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
            if p.get("normal"):
                nt = UsdShade.Shader.Define(stage, mp + "/NormalTexture")
                nt.CreateIdAttr("UsdUVTexture")
                nt.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"textures/{p['normal']}")
                nt.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
                nt.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                nt.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
                nt.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                nt.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                nt.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                nt.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nt.ConnectableAPI(), "rgb")
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p["color"]))
        if p.get("rough_tex"):
            rt = UsdShade.Shader.Define(stage, mp + "/RoughnessTexture")
            rt.CreateIdAttr("UsdUVTexture")
            rt.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"textures/{p['rough_tex']}")
            rt.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            rt.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            rt.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
            rt.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            rt.CreateOutput("g", Sdf.ValueTypeNames.Float)
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(rt.ConnectableAPI(), "g")
        else:
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(p["roughness"])
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(p["clearcoat"])
        sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(p["clearcoat_roughness"])
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        mats[name] = mat

    for m in meshes:
        mesh = UsdGeom.Mesh.Define(stage, f"/ChocolateRingCake/{m['name']}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(m["V"]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m["F"]), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m["F"].astype(np.int32).ravel()))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(m["N"]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*m["V"].min(axis=0).tolist()),
                                             Gf.Vec3f(*m["V"].max(axis=0).tolist())]))
        if "UV" in m:
            st = m["UV"].copy()
            st[:, 1] = 1.0 - st[:, 1]                     # USD t runs bottom-up
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                    UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(st))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[m["material"]])

    stage.GetRootLayer().Save()
    if path.exists():
        path.unlink()
    if not UsdUtils.CreateNewARKitUsdzPackage(Sdf.AssetPath(str(usdc)), str(path)):
        raise RuntimeError("USDZ packaging failed")
    del stage
    tmp.cleanup()


# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(exist_ok=True)
    base, normal, rough = bake_textures()
    base.save(CALIB / "cake_basecolor_preview.png")
    textures = {}
    for name, im, q in (("cake_basecolor.jpg", base, 90), ("cake_normal.jpg", normal, 92),
                        ("cake_roughness.jpg", rough, 90)):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True)
        textures[name] = buf.getvalue()

    cake = build_cake_mesh() | {"name": "Cake", "material": "Cake"}
    plate = build_plate_mesh() | {"name": "Plate", "material": "Plate"}
    meshes = [plate, cake]
    write_glb(OUT_DIR / "cake.glb", meshes, textures)
    write_usdz(OUT_DIR / "cake.usdz", meshes, textures)

    allV = np.vstack([m["V"] for m in meshes])
    lo, hi = allV.min(axis=0), allV.max(axis=0)
    body = cake["V"][cake["V"][:, 1] > PLATE_TOP + 0.01]
    print(f"triangles      : {sum(len(m['F']) for m in meshes):,}   vertices: {sum(len(m['V']) for m in meshes):,}")
    print(f"overall        : {(hi - lo)[0] * 100:.1f} x {(hi - lo)[2] * 100:.1f} cm, {(hi - lo)[1] * 100:.1f} cm tall")
    print(f"cake           : {np.hypot(body[:, 0], body[:, 2]).max() * 200:.1f} cm across at the base, "
          f"{(cake['V'][:, 1].max() - PLATE_TOP) * 100:.1f} cm tall above the plate")
    print("textures       : " + ", ".join(f"{k} {len(v) / 1e6:.2f} MB" for k, v in textures.items()))
    for f in ("cake.glb", "cake.usdz"):
        print(f"{f:<15}: {(OUT_DIR / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
