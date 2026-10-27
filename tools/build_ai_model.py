"""Turn an AI-generated 3D model (Higgsfield / Tripo GLB) into a real-size AR menu model.

The raw files are made for the generator's own viewer, not for a phone in AR: the wrong
size (a 23 cm bowl arrives 47 cm across), ~2 M triangles and 60 MB, and a glassy surface
that reads as plastic. This script:
  1. Size: centres the dish on its vertical axis, stands it on the table and scales it so
     its footprint is the real width.
  2. Mesh: welds the vertices the generator split along its texture seams (otherwise
     decimation opens hairline cracks there), decimates to a web-sized triangle count
     and lays out a fresh UV atlas.
  3. Texture: every texel of the new atlas casts a short ray along its normal onto the
     original full-detail mesh and copies the generator's colour and roughness from the
     hit, so no detail is lost to the decimation.
  4. Material: roughness is lifted out of the glassy range (food is matte-satin); with
     --glaze, bright low-saturation texels (a glazed bowl) stay glossy.
  5. Export: models/<name>.glb (web / Android) and models/<name>.usdz (iPhone).

World frame: Y up, metres, origin at the centre of the dish's underside.
The generator's files are large and not in the repo: put them in tools/ai/ (git-ignored).
Run:  .venv/Scripts/python tools/build_ai_model.py tools/ai/gnocchi.glb gnocchi --width 0.23 --glaze
      .venv/Scripts/python tools/build_ai_model.py tools/ai/tiramisu.glb tiramisu --width 0.16
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh
import xatlas
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from model_io import write_glb, write_usdz  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("src", type=Path, help="generator's GLB")
ap.add_argument("name", help="output name: models/<name>.glb and .usdz")
ap.add_argument("--width", type=float, required=True, help="real footprint across, metres")
ap.add_argument("--faces", type=int, default=130_000, help="triangles after decimation")
ap.add_argument("--tex", type=int, default=4096, help="colour texture size")
ap.add_argument("--glaze", action="store_true", help="keep bright, unsaturated texels glossy (glazed bowl)")
ap.add_argument("--turn", type=float, default=0.0, help="turn about the vertical axis, degrees")
ap.add_argument("--out", type=Path, default=ROOT / "models")
args = ap.parse_args()
T = args.tex

# --------------------------------------------------------------------------- #
# 1. Source mesh at real size
# --------------------------------------------------------------------------- #
src = trimesh.load(args.src).to_geometry()
hiV = np.asarray(src.vertices, float)
hiF = np.asarray(src.faces)
hiUV = np.asarray(src.visual.uv, float)
mat = src.visual.material
base_src = np.asarray(mat.baseColorTexture.convert("RGB"), np.float32)
mr_src = (np.asarray(mat.metallicRoughnessTexture.convert("RGB"), np.float32)
          if mat.metallicRoughnessTexture is not None else None)

c = (hiV.min(0) + hiV.max(0)) / 2
hiV -= [c[0], hiV[:, 1].min(), c[2]]
foot = hiV[:, [0, 2]].max(0) - hiV[:, [0, 2]].min(0)
hiV *= args.width / foot.max()
if args.turn:
    a = np.radians(args.turn)
    hiV = hiV @ np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]).T
hi = trimesh.Trimesh(hiV, hiF, process=False)

# --------------------------------------------------------------------------- #
# 2. Weld, decimate, new atlas
# --------------------------------------------------------------------------- #
_, first, inv = np.unique(np.round(hiV / 1e-6).astype(np.int64), axis=0, return_index=True, return_inverse=True)
Vw, Fw = hiV[first], inv.reshape(-1)[hiF]
Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
Vd, Fd = fast_simplification.simplify(Vw.astype(np.float32), Fw.astype(np.int32),
                                      target_reduction=max(0.0, 1 - args.faces / len(Fw)))
po = xatlas.PackOptions()
po.padding, po.resolution, po.bilinear = 2, T, True
atlas = xatlas.Atlas()
atlas.add_mesh(Vd.astype(np.float32), Fd.astype(np.uint32))
atlas.generate(pack_options=po)
print(f"atlas: {atlas.chart_count:,} charts")
assert atlas.atlas_count == 1, "charts spilled into a second atlas: lower --faces or raise --tex"
vmap, F, UV = atlas[0]
V = Vd[vmap]
# normals from the welded mesh, so both copies of a seam vertex shade the same
N = np.asarray(trimesh.Trimesh(Vd, Fd, process=False).vertex_normals, np.float32)[vmap]
print(f"mesh: {len(hiF):,} -> {len(F):,} triangles, {len(V):,} vertices")

# --------------------------------------------------------------------------- #
# 3. Bake: texel -> ray onto the source mesh -> source texture
# --------------------------------------------------------------------------- #
tri_uv = UV[F] * T          # glTF UVs run top-down, so v maps straight to rows
tid = np.full((T, T), -1, np.int32)
bary = np.zeros((T, T, 3), np.float32)
for i, (a, b, cc) in enumerate(tri_uv):
    x0, y0 = np.maximum(np.floor(np.minimum(np.minimum(a, b), cc)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(np.maximum(np.maximum(a, b), cc)).astype(int), T - 1)
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    den = (b[1] - cc[1]) * (a[0] - cc[0]) + (cc[0] - b[0]) * (a[1] - cc[1])
    if abs(den) < 1e-12:
        continue
    l0 = ((b[1] - cc[1]) * (xs - cc[0]) + (cc[0] - b[0]) * (ys - cc[1])) / den
    l1 = ((cc[1] - a[1]) * (xs - cc[0]) + (a[0] - cc[0]) * (ys - cc[1])) / den
    l2 = 1 - l0 - l1
    inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
    yy, xx = (ys[inside] - 0.5).astype(int), (xs[inside] - 0.5).astype(int)
    tid[yy, xx] = i
    bary[yy, xx] = np.stack([l0[inside], l1[inside], l2[inside]], -1)
mask = tid >= 0
ti, bw = tid[mask], bary[mask]
P = (V[F[ti]] * bw[..., None]).sum(1)
Nn = (N[F[ti]] * bw[..., None]).sum(1)
Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-9

EPS = 0.004
ray = trimesh.ray.ray_pyembree.RayMeshIntersector(hi)
origins = P + Nn * EPS
loc, ridx, hit = ray.intersects_location(origins, -Nn, multiple_hits=False)
good = np.linalg.norm(loc - origins[ridx], axis=1) < 2 * EPS
ridx, hit, loc = ridx[good], hit[good], loc[good]
uvs = np.full((len(P), 2), np.nan)
bh = trimesh.triangles.points_to_barycentric(hi.triangles[hit], loc)
uvs[ridx] = (hiUV[hiF[hit]] * bh[..., None]).sum(1)
miss = np.isnan(uvs[:, 0])
if miss.any():   # nothing within reach along the normal: nearest source vertex
    uvs[miss] = hiUV[cKDTree(hiV).query(P[miss])[1]]
print(f"bake: {mask.mean() * 100:.0f}% of the atlas used, {100 * (1 - miss.mean()):.1f}% of texels hit by rays")


def sample(img, uv):
    h, w = img.shape[:2]
    x, y = uv[:, 0] * (w - 1), (1 - uv[:, 1]) * (h - 1)   # trimesh flips v on load
    return np.stack([map_coordinates(img[..., k], [y, x], order=1) for k in range(img.shape[2])], -1)


def to_atlas(vals):
    img = np.zeros((T, T, vals.shape[1]), np.float32)
    img[mask] = vals
    idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)   # bleed over seams
    return img[idx[0], idx[1]]


col = sample(base_src, uvs)
r_src = sample(mr_src, uvs)[:, 1] / 255 if mr_src is not None else np.full(len(uvs), 0.6)

# --------------------------------------------------------------------------- #
# 4. Roughness: out of the glassy range
# --------------------------------------------------------------------------- #
rough = np.clip(0.45 + 0.4 * r_src, 0.5, 0.85)
if args.glaze:
    rgb = col / 255
    glaze = (rgb.mean(1) > 0.62) & (rgb.max(1) - rgb.min(1) < 0.18)
    rough = np.where(glaze, 0.18, rough)
rough_img = np.zeros((T, T, 3), np.float32)
rough_img[..., 0] = 255
rough_img[..., 1] = to_atlas(rough[:, None] * 255)[..., 0]

# --------------------------------------------------------------------------- #
# 5. Export
# --------------------------------------------------------------------------- #
tex = {}
for fname, arr, size, q in ((f"{args.name}_basecolor.jpg", to_atlas(col), T, 88),
                            (f"{args.name}_roughness.jpg", rough_img, T // 4, 90)):
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if size != T:
        im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    tex[fname] = buf.getvalue()
materials = {"Dish": dict(color=[1.0, 1.0, 1.0], texture=f"{args.name}_basecolor.jpg",
                          rough_tex=f"{args.name}_roughness.jpg", roughness=1.0, clearcoat=0.0,
                          clearcoat_roughness=0.0)}
mesh = {"V": V.astype(np.float32), "N": N.astype(np.float32), "F": F.astype(np.int64), "UV": UV.astype(np.float32),
        "name": "Dish", "material": "Dish"}
root = args.name.capitalize()
args.out.mkdir(exist_ok=True)
write_glb(args.out / f"{args.name}.glb", [mesh], tex, materials, root, generator="ar-menu-poc/build_ai_model.py")
write_usdz(args.out / f"{args.name}.usdz", [mesh], tex, materials, root)
lo, hi_ = V.min(0), V.max(0)
print(f"size: {(hi_ - lo)[0] * 100:.1f} x {(hi_ - lo)[2] * 100:.1f} cm, {(hi_ - lo)[1] * 100:.1f} cm tall")
for ext in ("glb", "usdz"):
    print(f"{args.name}.{ext}: {(args.out / f'{args.name}.{ext}').stat().st_size / 1e6:.2f} MB")
