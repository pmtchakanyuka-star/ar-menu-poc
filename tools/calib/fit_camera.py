"""Fit a pinhole camera and the cake's profile to the reference photo.

Inputs:  tools/calib/edges.json (from detect_edges.py) + a few hand-measured points.
Outputs: tools/calib/calibration.json  camera + profile, rescaled so the cake's
                                        base is CAKE_BASE_DIAMETER across
         tools/calib/fit_overlay.png   the fitted model drawn over the photo

World frame: Y up, metres, origin on the cake axis at the plate's top surface;
azimuth theta = 0 faces the camera.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
E = json.loads((HERE / "edges.json").read_text())
W, H = E["image_size"]
CX, CY = W / 2, H / 2
RP = 0.16                    # plate rim radius: provisional scale, rescaled at the end
CAKE_BASE_DIAMETER = 0.26    # what the user asked for

# Hand-measured on zoomed crops of the photo (image px).
HOLE_RIM = np.array([(512, 700), (842, 700), (668, 572)])    # left, right, far (near edge is an occluder)
HOLE_FLOOR = np.array([(548, 735), (815, 735), (680, 655)])  # left, right, far (near edge hidden by rim)
BASE_FRONT = np.array([(680, 1290)])                          # highlight line where the cake meets the pool, front

O1 = np.array([660.0, 1050.0])   # ray origin used for the plate / pool edges
O2 = np.array([677.0, 680.0])    # ray origin used for the top outline

NAMES = "f cam_h cam_d yaw pitch roll Rf Hf Rh Hh Rc Hc Ro Ho Rb Rq rim_h".split()
X0 = dict(f=1200, cam_h=0.40, cam_d=0.30, yaw=0.0, pitch=np.radians(50), roll=0.0,
          Rf=0.045, Hf=0.050, Rh=0.058, Hh=0.085, Rc=0.085, Hc=0.100, Ro=0.112, Ho=0.085,
          Rb=0.130, Rq=0.152, rim_h=0.007)
LO = dict(f=700, cam_h=0.10, cam_d=0.05, yaw=-0.5, pitch=0.3, roll=-0.2, Rf=0.02, Hf=0.02, Rh=0.03,
          Hh=0.04, Rc=0.05, Hc=0.05, Ro=0.07, Ho=0.03, Rb=0.08, Rq=0.10, rim_h=0.0)
HI = dict(f=4000, cam_h=1.5, cam_d=1.5, yaw=0.5, pitch=1.4, roll=0.2, Rf=0.07, Hf=0.09, Rh=0.09,
          Hh=0.13, Rc=0.12, Hc=0.15, Ro=0.15, Ho=0.14, Rb=0.16, Rq=0.17, rim_h=0.02)


def unpack(x):
    return dict(zip(NAMES, x))


def camera(p):
    yaw, pitch, roll = p["yaw"], p["pitch"], p["roll"]
    d = np.array([-np.sin(yaw) * np.cos(pitch), -np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    right = np.cross(d, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(d, right)
    cr, sr = np.cos(roll), np.sin(roll)
    Rm = np.stack([cr * right + sr * down, -sr * right + cr * down, d])
    C = np.array([0.0, p["cam_h"], p["cam_d"]])
    return p["f"], C, Rm


def project(P, cam):
    f, C, Rm = cam
    Q = (P - C) @ Rm.T
    return np.stack([CX + f * Q[:, 0] / Q[:, 2], CY + f * Q[:, 1] / Q[:, 2]], axis=-1)


def circle(r, y, n=720):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.sin(t), np.full(n, y), r * np.cos(t)], axis=-1)


def smooth_resample(points, n, sub=24):
    pts = np.asarray(points, dtype=float)
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    t = np.linspace(0.0, 1.0, sub, endpoint=False)[:, None]
    dense = [0.5 * (2 * P[i] + (-P[i - 1] + P[i + 1]) * t
                    + (2 * P[i - 1] - 5 * P[i] + 4 * P[i + 1] - P[i + 2]) * t**2
                    + (-P[i - 1] + 3 * P[i] - 3 * P[i + 1] + P[i + 2]) * t**3) for i in range(1, len(P) - 2)]
    dense = np.vstack(dense + [pts[-1:]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    tt = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(tt, s, dense[:, 0]), np.interp(tt, s, dense[:, 1])])


def control_points(p):
    """Cake profile control points (r, y), hole centre -> rim -> crest -> side -> base."""
    Rf, Hf, Rh, Hh, Rc, Hc, Ro, Ho, Rb = (p[k] for k in "Rf Hf Rh Hh Rc Hc Ro Ho Rb".split())
    return [
        (0.0, Hf), (0.6 * Rf, Hf), (Rf, Hf + 0.0012),                         # sauce in the hole
        (Rf + 0.35 * (Rh - Rf), Hf + 0.35 * (Hh - Hf)),                       # hole wall
        (Rf + 0.75 * (Rh - Rf), Hf + 0.80 * (Hh - Hf)),
        (Rh, Hh),                                                              # rim / inner shoulder
        (Rh + 0.35 * (Rc - Rh), Hh + 0.80 * (Hc - Hh)),
        (Rc, Hc),                                                              # crest
        (Rc + 0.55 * (Ro - Rc), Hc - 0.25 * (Hc - Ho)),
        (Ro, Ho),                                                              # outer shoulder
        (Ro + 0.35 * (Rb - Ro), 0.55 * Ho),                                    # side
        (Rb - 0.002, 0.012),
        (Rb, 0.005),                                                           # base
        (Rb + 0.006, 0.0038),                                                  # glaze foot
    ]


RIM_INDEX = 5  # control point index of the hole rim


def outer_surface(p, n_theta=540, n_prof=90):
    """Dense points on the cake from the rim over the crest down to the base."""
    cp = control_points(p)
    prof = smooth_resample(cp[RIM_INDEX:13], n_prof)
    t = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    r, y = prof[:, 0], prof[:, 1]
    return np.stack([np.outer(np.sin(t), r), np.tile(y, (n_theta, 1)), np.outer(np.cos(t), r)], -1).reshape(-1, 3)


def polar(xy, origin):
    v = xy - origin
    return np.arctan2(v[:, 1], v[:, 0]), np.hypot(v[:, 0], v[:, 1])


def ray_rho(curve_xy, origin, angles):
    a, r = polar(curve_xy, origin)
    o = np.argsort(a)
    a, r = a[o], r[o]
    a = np.concatenate([a - 2 * np.pi, a, a + 2 * np.pi])
    return np.interp(angles, a, np.tile(r, 3))


def silhouette_rho(pts_xy, origin, angles, half_width=np.radians(0.75)):
    a, r = polar(pts_xy, origin)
    bins = 1440
    idx = ((a + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    mx = np.zeros(bins)
    np.maximum.at(mx, idx, r)
    k = int(np.ceil(half_width / (2 * np.pi) * bins))
    out = []
    for ang in angles:
        b = int((ang + np.pi) / (2 * np.pi) * bins) % bins
        out.append(mx[[(b + j) % bins for j in range(-k, k + 1)]].max())
    return np.array(out)


def nearest(points, curve_xy):
    d = np.linalg.norm(points[:, None, :] - curve_xy[None, :, :], axis=-1)
    return d.min(axis=1)


obs_rim = np.array([pt[:2] for pt in E["plate_rim"]])
obs_pool = np.array([pt[:2] for pt in E["pool_edge"]])
obs_top = np.array([pt[:2] for pt in E["top_outline"]])
a_rim, r_rim = polar(obs_rim, O1)
a_pool, r_pool = polar(obs_pool, O1)
a_top, r_top = polar(obs_top, O2)


def residuals(x):
    p = unpack(x)
    cam = camera(p)
    res = [
        r_rim - ray_rho(project(circle(RP, p["rim_h"]), cam), O1, a_rim),
        0.6 * (r_pool - ray_rho(project(circle(p["Rq"], 0.0015), cam), O1, a_pool)),  # pool is irregular
        r_top - silhouette_rho(project(outer_surface(p), cam), O2, a_top),
        3.0 * nearest(HOLE_RIM, project(circle(p["Rh"], p["Hh"]), cam)),
        3.0 * nearest(HOLE_FLOOR, project(circle(p["Rf"], p["Hf"]), cam)),
        3.0 * nearest(BASE_FRONT, project(circle(p["Rb"], 0.005), cam)),
        [3.0 * (p["rim_h"] - 0.007) / 0.003],                                   # weak prior
    ]
    return np.concatenate([np.atleast_1d(r) for r in res])


def overlay(p, path):
    cam = camera(p)
    im = Image.open(HERE / "reference.jpg").convert("RGB")
    d = ImageDraw.Draw(im)

    def poly(P, col, w=2):
        xy = project(P, cam)
        d.line([tuple(v) for v in np.vstack([xy, xy[:1]])], fill=col, width=w)

    poly(circle(RP, p["rim_h"]), (0, 160, 255))
    poly(circle(p["Rq"], 0.0015), (255, 0, 200))
    poly(circle(p["Rb"], 0.005), (255, 140, 0))
    poly(circle(p["Rh"], p["Hh"]), (0, 255, 120))
    poly(circle(p["Rf"], p["Hf"]), (0, 255, 255))
    poly(circle(p["Ro"], p["Ho"]), (255, 255, 255), 1)
    poly(circle(p["Rc"], p["Hc"]), (200, 200, 200), 1)
    # profile drawn in the plane facing the camera's left/right (theta = +-90 deg)
    prof = smooth_resample(control_points(p), 200)
    for s in (1, -1):
        P = np.stack([s * prof[:, 0], prof[:, 1], np.zeros(len(prof))], -1)
        xy = project(P, cam)
        d.line([tuple(v) for v in xy], fill=(255, 60, 60), width=2)
    for pts, col in ((obs_rim, (0, 160, 255)), (obs_pool, (255, 0, 200)), (obs_top, (255, 230, 0)),
                     (HOLE_RIM, (0, 255, 120)), (HOLE_FLOOR, (0, 255, 255)), (BASE_FRONT, (255, 140, 0))):
        for x, y in pts:
            d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=col, width=2)
    im.resize((W // 2, H // 2)).save(path)


def main():
    x0 = np.array([X0[k] for k in NAMES], float)
    lo = np.array([LO[k] for k in NAMES], float)
    hi = np.array([HI[k] for k in NAMES], float)
    sol = least_squares(residuals, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0,
                        x_scale="jac", diff_step=1e-4, max_nfev=400)
    p = unpack(sol.x)
    res = residuals(sol.x)
    n = [len(r_rim), len(r_pool), len(r_top), 3, 3, 1]
    parts = np.split(np.abs(res[:-1]), np.cumsum(n)[:-1])
    print("status", sol.status, sol.message)
    for name, r in zip(["plate rim", "pool edge", "top outline", "hole rim x3", "hole floor x3", "base x3"], parts):
        print(f"  {name:14s} median {np.median(r):6.2f}px  max {r.max():6.2f}px")
    s = (CAKE_BASE_DIAMETER / 2) / p["Rb"]
    print(f"  focal {p['f']:.0f}px  pitch {np.degrees(p['pitch']):.1f}deg  yaw {np.degrees(p['yaw']):.1f}deg  "
          f"roll {np.degrees(p['roll']):.2f}deg")
    print(f"  camera {p['cam_d'] * s * 100:.1f} cm back, {p['cam_h'] * s * 100:.1f} cm above the plate (scaled)")
    lengths = {k: p[k] * s for k in "Rf Hf Rh Hh Rc Hc Ro Ho Rb Rq rim_h".split()}
    print("  scaled profile (cm):", {k: round(v * 100, 1) for k, v in lengths.items()},
          f" plate diameter {2 * RP * s * 100:.1f} cm")
    cal = {
        "image_size": [W, H], "focal_px": p["f"], "principal_point": [CX, CY],
        "yaw": p["yaw"], "pitch": p["pitch"], "roll": p["roll"],
        "camera_position": [0.0, p["cam_h"] * s, p["cam_d"] * s],
        "plate_rim_radius": RP * s, **lengths,
    }
    (HERE / "calibration.json").write_text(json.dumps(cal, indent=1))
    overlay(p, HERE / "fit_overlay.png")


if __name__ == "__main__":
    main()
