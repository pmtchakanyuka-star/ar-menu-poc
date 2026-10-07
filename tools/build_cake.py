"""Build a real-world-scale 3D replica of the chocolate ring cake.

Outputs (Y-up, metres, origin = centre of the plate's underside, so the model
sits on whatever surface AR places it on):

  models/cake.glb   glTF 2.0 binary - web preview, Android (WebXR / Scene Viewer)
  models/cake.usdz  ARKit USDZ      - iPhone / iPad Quick Look

The model is procedural geometry + PBR materials, no textures:
  * a white glazed-ceramic platter (30 cm)
  * ONE continuous glossy ganache surface, lathed from a hand-fitted profile:
    the sauce pooled in the centre hole -> up the hole wall -> over the rounded
    crest -> down the flared sides -> the sauce pooled on the plate (with an
    irregular, rounded edge)
  * thousands of individual chocolate sprinkles (granulado) scattered over that
    surface, densest on the crest, sparse in the pools

Dimensions were estimated from the reference photo (ring cake from a tube pan,
~23 cm across the base, ~9 cm tall, ~9 cm hole). Edit the PROFILE / sizes below
to match a real cake exactly.

Run:  .venv/Scripts/python tools/build_cake.py
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models"
SEED = 7

# Plate top (the well the cake sits in) above the table, metres.
PLATE_TOP = 0.012

# Ganache profile of the cake, (radius, height above plate top), metres.
# Walks from the centre of the sauce pooled in the hole, up the hole wall, over
# the crest and down the outer side to where the glaze flows onto the plate.
CAKE_PROFILE = [
    (0.0000, 0.0480),  # sauce pooled in the hole
    (0.0220, 0.0480),
    (0.0365, 0.0483),
    (0.0410, 0.0496),  # meniscus up the hole wall
    (0.0436, 0.0528),
    (0.0455, 0.0592),  # hole wall (tube-pan funnel: narrows downward)
    (0.0472, 0.0672),
    (0.0488, 0.0748),
    (0.0512, 0.0816),  # inner shoulder
    (0.0565, 0.0870),
    (0.0645, 0.0907),  # rounded crest
    (0.0735, 0.0920),
    (0.0825, 0.0908),
    (0.0905, 0.0870),  # outer shoulder
    (0.0970, 0.0810),
    (0.1012, 0.0730),
    (0.1036, 0.0615),  # side, slightly wider at the base
    (0.1060, 0.0450),
    (0.1085, 0.0280),
    (0.1106, 0.0140),
    (0.1132, 0.0074),  # glaze foot flowing out onto the plate
    (0.1165, 0.0048),
    (0.1210, 0.0038),
]
POOL_DEPTH = 0.0035          # thickness of the sauce pooled on the plate
POOL_EDGE = 0.0045           # width of the pool's rounded bead edge
POOL_RADIUS = (0.1295, 0.1365)  # irregular pool outline stays inside the plate well

N_THETA = 256                # angular resolution of the lathed surfaces
N_CAKE = 132                 # profile samples, cake part
N_POOL = 14                  # profile samples, pool part (per angle)


def srgb(r: int, g: int, b: int) -> list[float]:
    """8-bit sRGB -> linear RGB (glTF baseColorFactor and USD diffuseColor are linear)."""
    c = np.array([r, g, b], dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return [round(float(v), 5) for v in lin]


MATERIALS = {
    # Glossy where the glaze is exposed (pools); darker satin where it sits in the
    # shadow of the sprinkle carpet - stands in for ambient occlusion, which
    # Quick Look / Scene Viewer don't compute.
    "Ganache": dict(color=srgb(44, 22, 13), roughness=0.28, clearcoat=0.5, clearcoat_roughness=0.22),
    "GanacheCovered": dict(color=srgb(30, 15, 9), roughness=0.55, clearcoat=0.0, clearcoat_roughness=0.0),
    "SprinkleDark": dict(color=srgb(40, 26, 19), roughness=0.42, clearcoat=0.0, clearcoat_roughness=0.0),
    "SprinkleMid": dict(color=srgb(50, 32, 23), roughness=0.38, clearcoat=0.0, clearcoat_roughness=0.0),
    "SprinkleLight": dict(color=srgb(60, 39, 28), roughness=0.36, clearcoat=0.0, clearcoat_roughness=0.0),
    "Plate": dict(color=srgb(238, 235, 229), roughness=0.16, clearcoat=0.6, clearcoat_roughness=0.08),
}


# --------------------------------------------------------------------------- #
# Small geometry helpers
# --------------------------------------------------------------------------- #
def smooth_resample(points, n: int, sub: int = 32) -> np.ndarray:
    """Catmull-Rom through `points`, resampled to `n` points evenly by arc length."""
    pts = np.asarray(points, dtype=float)
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    t = np.linspace(0.0, 1.0, sub, endpoint=False)[:, None]
    dense = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1 : i + 3]
        dense.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    dense.append(pts[-1:])
    dense = np.vstack(dense)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    target = np.linspace(0.0, s[-1], n)
    out = np.column_stack([np.interp(target, s, dense[:, 0]), np.interp(target, s, dense[:, 1])])
    out[:, 0] = np.maximum(out[:, 0], 0.0)
    return out


def periodic_noise(theta: np.ndarray, rng, kmax: int = 7, falloff: float = 1.4) -> np.ndarray:
    """Smooth 2*pi-periodic noise in [-1, 1] (random Fourier series)."""
    k = np.arange(1, kmax + 1)
    a = rng.normal(size=kmax) / k**falloff
    ph = rng.uniform(0, 2 * np.pi, kmax)
    v = (a * np.cos(np.outer(theta, k) + ph)).sum(axis=1)
    return v / np.abs(v).max()


def wave_noise(P: np.ndarray, rng, n: int = 40, lam=(0.005, 0.04)) -> np.ndarray:
    """Smooth 3D noise in [-1, 1]: a sum of random plane waves."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    wl = np.exp(rng.uniform(np.log(lam[0]), np.log(lam[1]), n))
    amp = (wl / wl.max()) ** 0.7
    ph = rng.uniform(0, 2 * np.pi, n)
    v = (amp * np.sin(2 * np.pi * (P @ d.T) / wl + ph)).sum(axis=1)
    return v / np.abs(v).max()


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # area-weighted
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def lathe(R: np.ndarray, Y: np.ndarray, theta: np.ndarray, attrs: dict | None = None):
    """Revolve per-angle profiles R/Y (n_theta x n_prof) into a closed-around mesh.

    Profile columns whose radius is 0 for every angle become a single pole vertex.
    Returns V, F and any per-grid-vertex `attrs`, compacted alongside V.
    """
    nt, npf = R.shape
    V = np.stack([R * np.cos(theta)[:, None], Y, R * np.sin(theta)[:, None]], axis=-1).reshape(-1, 3)
    idx = np.arange(nt * npf).reshape(nt, npf)
    for j in (0, npf - 1):
        if np.allclose(R[:, j], 0.0):
            idx[:, j] = idx[0, j]  # collapse to one pole vertex
    i0, j0 = np.meshgrid(np.arange(nt), np.arange(npf - 1), indexing="ij")
    i1 = (i0 + 1) % nt
    a, b, c, d = idx[i0, j0], idx[i1, j0], idx[i1, j0 + 1], idx[i0, j0 + 1]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    used = np.unique(F)
    remap = np.full(nt * npf, -1)
    remap[used] = np.arange(len(used))
    out_attrs = {k: v.reshape(-1)[used] for k, v in (attrs or {}).items()}
    return V[used], remap[F], out_attrs


def orient_outward(V, F, probe_mask: np.ndarray, outward: np.ndarray) -> np.ndarray:
    """Flip all faces if normals at `probe_mask` vertices don't point along `outward`."""
    N = vertex_normals(V, F)
    if (N[probe_mask] * outward[probe_mask]).sum(axis=1).mean() < 0:
        F = F[:, ::-1].copy()
    return F


# --------------------------------------------------------------------------- #
# Plate
# --------------------------------------------------------------------------- #
def build_plate():
    prof = smooth_resample(
        [
            (0.000, PLATE_TOP), (0.070, PLATE_TOP), (0.138, PLATE_TOP),     # flat well
            (0.1425, 0.0133), (0.1462, 0.0155), (0.1492, 0.0180),          # rim rises
            (0.1512, 0.0193), (0.1527, 0.0186), (0.1531, 0.0171),          # rounded lip
            (0.1515, 0.0157), (0.1470, 0.0135), (0.1390, 0.0107),          # rim underside
            (0.1220, 0.0072), (0.1050, 0.0050), (0.0965, 0.0034),
            (0.0950, 0.0008), (0.0930, 0.0000), (0.0880, 0.0000),          # foot ring
            (0.0866, 0.0010), (0.0858, 0.0034), (0.0700, 0.0042), (0.000, 0.0044),
        ],
        80,
    )
    theta = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    R = np.tile(prof[:, 0], (len(theta), 1))
    Y = np.tile(prof[:, 1], (len(theta), 1))
    V, F, _ = lathe(R, Y, theta)
    up = np.zeros_like(V)
    up[:, 1] = 1.0
    top = (V[:, 1] >= PLATE_TOP - 1e-6) & (np.hypot(V[:, 0], V[:, 2]) < 0.13)
    F = orient_outward(V, F, top, up)
    return V, F


# --------------------------------------------------------------------------- #
# Ganache-covered cake + sauce pool (one continuous surface)
# --------------------------------------------------------------------------- #
def build_ganache(rng):
    theta = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    nt = len(theta)
    cake = smooth_resample(CAKE_PROFILE, N_CAKE)
    r0, y0 = cake[:, 0], cake[:, 1]

    # Hand-made irregularity: the cake is slightly out of round, the crest
    # undulates, and the hole sits a touch off-centre.
    wobble_r = 0.010 * periodic_noise(theta, rng, kmax=5)
    wobble_y = 0.0016 * periodic_noise(theta, rng, kmax=6)
    body = np.clip((r0 - 0.039) / 0.02, 0, 1)                  # 0 in hole floor -> 1 on body
    upper = np.clip((y0 - 0.055) / 0.025, 0, 1)                 # weight toward the crest
    holeish = 1 - np.clip((r0 - 0.047) / 0.02, 0, 1)            # hole floor + wall
    in_foot = np.clip((0.1200 - r0) / 0.006, 0, 1)              # fade wobble out at the pool
    R_c = r0 * (1 + np.outer(wobble_r, body * in_foot))
    Y_c = y0 + np.outer(wobble_y, upper)

    # Sauce pooled on the plate: irregular outline, rounded bead edge.
    pool_R = np.interp(periodic_noise(theta, rng, kmax=6, falloff=1.1), [-1, 1], POOL_RADIUS)
    n_flat = N_POOL - 8
    t_flat = np.linspace(0, 1, n_flat + 2)[1:-1]
    phi = np.linspace(0, np.pi / 2, 7)
    R_p = np.empty((nt, N_POOL))
    Y_p = np.empty((nt, N_POOL))
    r_start = cake[-1, 0]
    flat_end = pool_R - POOL_EDGE
    R_p[:, :n_flat] = r_start + np.outer(flat_end - r_start, t_flat)
    Y_p[:, :n_flat] = POOL_DEPTH + (cake[-1, 1] - POOL_DEPTH) * (1 - t_flat) ** 2
    R_p[:, n_flat : n_flat + 7] = flat_end[:, None] + POOL_EDGE * np.sin(phi)
    Y_p[:, n_flat : n_flat + 7] = POOL_DEPTH * np.cos(phi)
    R_p[:, -1] = pool_R - 0.0004                                 # tuck under the plate surface
    Y_p[:, -1] = -0.0005

    R = np.hstack([R_c, R_p])
    Y = np.hstack([Y_c, Y_p]) + PLATE_TOP

    # Per-profile-sample sprinkle density (count per cm^2) and placement style.
    dens_c = np.select(
        [
            (r0 < 0.040) & (y0 < 0.0490),     # sauce in the hole
            r0 < 0.0515,                       # hole wall
            y0 > 0.0700,                       # crest - fully covered
            y0 > 0.0100,                       # sides
        ],
        [2.0, 14.0, 30.0, 16.0],
        default=12.0,                          # glaze foot
    )
    zone_c = np.select([(r0 < 0.040) & (y0 < 0.0490), y0 > 0.0700], [2, 0], default=1)
    t_pool = (R_p - r_start) / (pool_R[:, None] - r_start)
    dens_p = np.interp(t_pool, [0, 0.3, 0.8, 1.0], [9.0, 4.0, 1.0, 0.0])
    dens = np.hstack([np.tile(dens_c, (nt, 1)), dens_p])
    zone = np.hstack([np.tile(zone_c, (nt, 1)), np.full((nt, N_POOL), 2)])  # 0 crest, 1 sides, 2 pools
    disp_amp = np.hstack(
        [np.tile(np.where(r0 < 0.040, 0.00012, 0.0007) * in_foot.clip(0.15, 1), (nt, 1)),
         np.full((nt, N_POOL), 0.00008) * (np.arange(N_POOL) < n_flat)]
    )

    hole_w = np.hstack([np.tile(holeish, (nt, 1)), np.zeros((nt, N_POOL))])
    V, F, at = lathe(R, Y, theta, {"dens": dens, "zone": zone, "disp": disp_amp, "hole": hole_w})
    outward = V * np.array([1.0, 0.0, 1.0])
    probe = (V[:, 1] > PLATE_TOP + 0.02) & (V[:, 1] < PLATE_TOP + 0.06) & (np.hypot(V[:, 0], V[:, 2]) > 0.09)
    F = orient_outward(V, F, probe, outward)

    # The hole sits a touch off-centre.
    ha = rng.uniform(0, 2 * np.pi)
    V[:, [0, 2]] += 0.0018 * np.array([np.cos(ha), np.sin(ha)]) * at["hole"][:, None]

    # Glaze is never perfectly smooth: low bumps along the normal.
    N = vertex_normals(V, F)
    V = V + N * (at["disp"] * wave_noise(V, rng))[:, None]
    return V, F, at


# --------------------------------------------------------------------------- #
# Sprinkles (chocolate granulado)
# --------------------------------------------------------------------------- #
SPRINKLES_PER_MESH = 10_000      # 6 verts each -> 60k verts, keeps indices 16-bit


def sprinkle_template():
    """Triangular rod: ring at -x (0-2), ring at +x (3-5); 3 side quads + 2 end caps."""
    F = []
    for k in range(3):
        k1 = (k + 1) % 3
        F += [[k, 3 + k, 3 + k1], [k, 3 + k1, k1]]
    F += [[0, 1, 2], [3, 5, 4]]
    return np.array(F)


def build_sprinkles(V, F, at, rng):
    Nv = vertex_normals(V, F)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    fd = at["dens"][F].mean(axis=1) * 1e4                      # per m^2
    w = area * fd
    count = int(w.sum())
    fi = rng.choice(len(F), size=count, p=w / w.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    bary = np.column_stack([1 - u - v, u, v])
    P = (tri[fi] * bary[:, :, None]).sum(axis=1)
    n = (Nv[F[fi]] * bary[:, :, None]).sum(axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    zone = np.round(at["zone"][F[fi]].mean(axis=1)).astype(int)  # 0 crest, 1 sides, 2 pools

    # Size: granulado is ~1.5 mm thick, 4.5-7.5 mm long; some broken short ones.
    # (rad is the triangle's circumradius, so the rod reads ~1.5 mm thick.)
    rad = rng.uniform(0.00075, 0.00092, count)
    length = np.where(rng.random(count) < 0.15, rng.uniform(0.0020, 0.0040, count),
                      rng.uniform(0.0045, 0.0075, count))

    # Lie in the tangent plane at a random heading, with a little tilt; a few
    # stick up at a steeper angle. On the crest they pile up in layers.
    a = np.where(np.abs(n[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    head = rng.uniform(0, 2 * np.pi, count)
    d = np.cos(head)[:, None] * t1 + np.sin(head)[:, None] * t2
    tilt = np.radians(rng.normal(0, np.where(zone == 0, 13, 8)))
    steep = (rng.random(count) < 0.07) & (zone == 0)
    tilt[steep] = np.radians(rng.uniform(18, 40, steep.sum()))
    X = np.cos(tilt)[:, None] * d + np.sin(tilt)[:, None] * n
    lift = np.select([zone == 0, zone == 1], [rng.uniform(0.3, 2.4, count), rng.uniform(0.25, 0.9, count)],
                     default=rng.uniform(0.15, 0.55, count))
    lift = lift * rad + np.maximum(np.sin(tilt), 0) * length * 0.42
    C = P + n * lift[:, None]
    Yb = np.cross(n, X)
    Yb /= np.linalg.norm(Yb, axis=1, keepdims=True)
    Zb = np.cross(X, Yb)

    # Build every rod in its local frame (x = length), smooth normals so the
    # 3-sided prism shades round.
    ang = 2 * np.pi * np.arange(3) / 3 + rng.uniform(0, 2 * np.pi, count)[:, None]       # (n, 3)
    half = (length / 2)[:, None]
    ca, sa = np.cos(ang), np.sin(ang)
    loc = np.stack([np.hstack([-half.repeat(3, 1), half.repeat(3, 1)]),
                    rad[:, None] * np.hstack([ca, ca]),
                    rad[:, None] * np.hstack([sa, sa])], axis=-1)                         # (n, 6, 3)
    nrm = np.stack([np.hstack([np.full((count, 3), -0.3), np.full((count, 3), 0.3)]),
                    np.hstack([ca, ca]), np.hstack([sa, sa])], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    basis = np.stack([X, Yb, Zb], axis=1)                                                  # (n, 3, 3)
    SV = C[:, None, :] + np.einsum("nvk,nkj->nvj", loc, basis)
    SN = np.einsum("nvk,nkj->nvj", nrm, basis)

    tf = sprinkle_template()
    # Make template winding agree with the outward normals.
    tv = SV[0][tf]
    fnorm = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    if (fnorm * SN[0][tf].mean(axis=1)).sum(axis=1).mean() < 0:
        tf = tf[:, ::-1].copy()

    # Three slightly different shades; each shade split into <=10k-rod meshes.
    shade = rng.choice(3, size=count, p=[0.35, 0.40, 0.25])
    groups = []
    for gi, name in enumerate(["SprinkleDark", "SprinkleMid", "SprinkleLight"]):
        idx = np.nonzero(shade == gi)[0]
        for ci, sel in enumerate(np.array_split(idx, -(-len(idx) // SPRINKLES_PER_MESH))):
            gF = (tf[None, :, :] + (np.arange(len(sel)) * 6)[:, None, None]).reshape(-1, 3)
            groups.append((f"{name}_{ci}", name, SV[sel].reshape(-1, 3), SN[sel].reshape(-1, 3), gF))
    zone_counts = {z: int((zone == k).sum()) for k, z in enumerate(["crest", "sides", "pools"])}
    return groups, count, zone_counts


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_glb(path: Path, meshes: list[dict]):
    mat_names = list(MATERIALS)
    binbuf = bytearray()
    views, accessors, gl_meshes, nodes = [], [], [], []

    def add_view(data: bytes, target: int) -> int:
        while len(binbuf) % 4:
            binbuf.append(0)
        views.append({"buffer": 0, "byteOffset": len(binbuf), "byteLength": len(data), "target": target})
        binbuf.extend(data)
        return len(views) - 1

    shared = {}  # meshes passing the same vertex array share its accessors
    for m in meshes:
        pos = m["V"].astype(np.float32)
        nrm = m["N"].astype(np.float32)
        idx = m["F"].astype(np.uint16 if len(pos) < 65536 else np.uint32).ravel()
        if id(m["V"]) not in shared:
            shared[id(m["V"])] = len(accessors)
            accessors.append({"bufferView": add_view(pos.tobytes(), 34962), "componentType": 5126,
                              "count": len(pos), "type": "VEC3",
                              "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()})
            accessors.append({"bufferView": add_view(nrm.tobytes(), 34962), "componentType": 5126,
                              "count": len(nrm), "type": "VEC3"})
        acc_p = shared[id(m["V"])]
        accessors.append({"bufferView": add_view(idx.tobytes(), 34963),
                          "componentType": 5123 if idx.dtype == np.uint16 else 5125,
                          "count": len(idx), "type": "SCALAR"})
        gl_meshes.append({"name": m["name"], "primitives": [{
            "attributes": {"POSITION": acc_p, "NORMAL": acc_p + 1}, "indices": len(accessors) - 1,
            "material": mat_names.index(m["material"]), "mode": 4}]})
        nodes.append({"name": m["name"], "mesh": len(gl_meshes) - 1})

    materials = []
    for name, p in MATERIALS.items():
        mat = {"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": p["color"] + [1.0], "metallicFactor": 0.0, "roughnessFactor": p["roughness"]}}
        if p["clearcoat"] > 0:
            mat["extensions"] = {"KHR_materials_clearcoat": {
                "clearcoatFactor": p["clearcoat"], "clearcoatRoughnessFactor": p["clearcoat_roughness"]}}
        materials.append(mat)

    binbuf.extend(b"\0" * (-len(binbuf) % 4))
    root = {"name": "ChocolateRingCake", "children": list(range(1, len(nodes) + 1))}
    gltf = {
        "asset": {"version": "2.0", "generator": "ar-menu-poc/build_cake.py"},
        "extensionsUsed": ["KHR_materials_clearcoat"],
        "scene": 0, "scenes": [{"name": "Scene", "nodes": [0]}],
        "nodes": [root] + nodes, "meshes": gl_meshes, "materials": materials,
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binbuf)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binbuf)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binbuf), 0x004E4942) + bytes(binbuf))


def write_usdz(path: Path, meshes: list[dict]):
    import tempfile

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # usdc may stay mapped on Windows
    usdc = Path(tmp.name) / "cake.usdc"
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
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p["color"]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(p["roughness"])
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(p["clearcoat"])
        sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(p["clearcoat_roughness"])
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        mats[name] = mat

    for m in meshes:
        mesh = UsdGeom.Mesh.Define(stage, f"/ChocolateRingCake/{m['name']}")
        pts = m["V"].astype(np.float32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m["F"]), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m["F"].astype(np.int32).ravel()))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(m["N"].astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*pts.min(axis=0).tolist()),
                                             Gf.Vec3f(*pts.max(axis=0).tolist())]))
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
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    pV, pF = build_plate()
    gV, gF, at = build_ganache(rng)
    groups, n_spr, zones = build_sprinkles(gV, gF, at, rng)
    gN = vertex_normals(gV, gF)
    covered = at["dens"][gF].mean(axis=1) >= 10.0

    meshes = [
        {"name": "Plate", "V": pV, "N": vertex_normals(pV, pF), "F": pF, "material": "Plate"},
    ] + [
        # Same vertices, faces split by how densely sprinkled they are.
        {"name": name, "V": gV, "N": gN, "F": gF[sel], "material": name}
        for name, sel in (("Ganache", ~covered), ("GanacheCovered", covered))
    ] + [{"name": name, "V": v, "N": n, "F": f, "material": mat} for name, mat, v, n, f in groups]

    allV = np.vstack([m["V"] for m in meshes])
    lo, hi = allV.min(axis=0), allV.max(axis=0)
    cake_hi = gV[:, 1].max() - PLATE_TOP
    tris = sum(len(m["F"]) for m in meshes)
    verts = sum(len(m["V"]) for m in meshes)

    write_glb(OUT_DIR / "cake.glb", meshes)
    write_usdz(OUT_DIR / "cake.usdz", meshes)

    body = np.hypot(gV[:, 0], gV[:, 2])[(gV[:, 1] > PLATE_TOP + 0.012)]
    print(f"sprinkles      : {n_spr:,}  {zones}")
    print(f"triangles      : {tris:,}   vertices: {verts:,}")
    print(f"overall size   : {(hi - lo)[0] * 100:.1f} x {(hi - lo)[2] * 100:.1f} cm, {(hi - lo)[1] * 100:.1f} cm tall")
    print(f"cake           : {body.max() * 200:.1f} cm across at the base, {cake_hi * 100:.1f} cm tall")
    for f in ("cake.glb", "cake.usdz"):
        print(f"{f:<15}: {(OUT_DIR / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
