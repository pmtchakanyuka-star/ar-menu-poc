# AR Menu — proof of concept

Scan the QR code with a phone and tap **View on your table**. A chocolate ring
cake appears on your table at its real size: 26 cm across, 11 cm tall, on a
32 cm plate. Walk around it and lean in.

**Live:** https://pmtchakanyuka-star.github.io/ar-menu-poc/

<img src="qr.png" width="180" alt="QR code for the AR menu page">

## How it works

| Piece | What it is |
| --- | --- |
| `index.html` | The menu-item page. It uses Google's [`<model-viewer>`](https://modelviewer.dev) for the 3D preview and the AR hand-off. `ar-scale="fixed"` locks the cake to its real size. |
| `models/cake.glb` | glTF 2.0 model for the web preview and Android: WebXR in Chrome, with Scene Viewer as the fallback. |
| `models/cake.usdz` | USDZ model for iPhone and iPad (AR Quick Look). |
| `tools/calib/detect_edges.py` | Finds the plate rim, the edge of the glaze pool and the cake's top outline in the reference photo. |
| `tools/calib/fit_camera.py` | Fits a camera and the cake's profile to those edges, plus a few hand-measured points on the hole and base, then scales the result so the base is 26 cm. |
| `tools/build_cake.py` | Builds the shape and projects the photo onto it as texture. Writes both model files. |
| `tools/make_qr.py` | Writes `qr.png` / `qr.svg`. |

### The model is made from one photo

1. **Shape.** A pinhole camera and the cake's profile are fitted to the photo. The profile runs from the pool of glaze in the hole, up the hole wall, over the rounded crest and down the flared sides to the base. The fitted edges land within a few pixels of the photo. The only real-world number supplied is the base width (26 cm, a standard ring pan); everything else comes from the photo. The glaze pool's wavy outline on the plate is back-projected from the photo too.
2. **Texture.** The photo is projected onto that shape through the fitted camera, so everything the camera saw head-on keeps its real pixels. One photo can't see everything (the back, the near wall of the hole), so those parts are filled with randomly chosen patches of the seen surface at the same height, with crossfades and no mirroring. First the photo's lighting is evened out around the cake, so patches match and the AR viewer's lights do the shading. A bump map and a roughness map come from the same pixels: glossy glaze pools, satin sprinkles.
3. **Export.** Both files are in metres and Y-up, with the origin under the plate, so AR puts the cake straight onto the table.

| | |
| --- | --- |
| Cake | 26 cm across the base, 11 cm tall (31.6 cm plate) |
| Triangles | ~101k |
| Textures | colour 2048×1024, bump 2048×1024, roughness 1024×512 (JPEG) |
| Files | `cake.glb` 4.4 MB · `cake.usdz` 3.3 MB |

Both files are validated. The Khronos glTF Validator reports 0 errors and 0 warnings, and all 28 OpenUSD validators pass for the USDZ, including the usdz packaging rules.

## Rebuild

The reference photo isn't in the repo, because it shows more than the cake. Put it at
`tools/calib/reference.jpg`, then:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
.venv/Scripts/python tools/calib/detect_edges.py
.venv/Scripts/python tools/calib/fit_camera.py    # writes fit_overlay.png - check the fit by eye
.venv/Scripts/python tools/build_cake.py
.venv/Scripts/python tools/make_qr.py
```

To use a different real size, change `CAKE_BASE_DIAMETER` in `tools/calib/fit_camera.py`.

## From PoC to a real AR menu

- **Scan dishes instead of photographing them.** A 1–2 minute walk-around scan with Reality Composer or Object Capture (iPhone), Polycam, or Luma sees every side and measures true size. That gives a fully photo-real model with no filled-in back. The same page and QR flow work as they are.
- **Compress and serve.** Use meshopt or Draco for the GLB, KTX2 textures and a CDN.
- **One QR per table.** It opens the menu, and each dish opens in AR from there.
