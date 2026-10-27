# Gnocchi: photo-real scan plan

The current `models/gnocchi.glb` is textured from one overhead photo. The top is pixel-real,
but the sides of the gnocchi facing away from that camera are stretched. This plan replaces it
with a full photogrammetry scan, so every surface carries real pixels.

## Decisions

| | |
| --- | --- |
| Capture | Full orbit shoot, Canon EOS M50 II |
| Hidden surfaces | Filled only with real pixels from elsewhere on the plate, never generated |
| Highlights | De-lit texture plus real gloss, so the shine moves with the room's light in AR |
| Size | Bowl 23 cm across the rim |
| File budget | Up to ~10 MB per file: 4K colour texture, detailed mesh |
| Devices | iPhone (USDZ, Quick Look) and Android/web (GLB), both checked |
| Software | Windows laptop, no NVIDIA GPU: COLMAP (CPU build) + OpenMVS, then Blender |
| Where | A Claude session on the laptop (Claude Desktop, or `claude remote-control` in this repo), so it can drive Blender |
| Menu text | "Pumpkin Gnocchi", description unchanged |

## 1. The shoot (about 10 minutes, once the dish is plated)

Sauce skins over and dries within minutes and steam ruins alignment. Plate it fresh, let the
steam stop (about 1–2 minutes), then shoot straight away. Don't touch or move anything until
you're done.

**Set-up**
- Soft, even light: a window with no direct sun, or ceiling light plus a white sheet as a
  diffuser. No flash. Turn off any light that makes a hard shadow.
- Put the bowl on a **patterned, matte** surface such as a newspaper, a printed tablecloth or
  a cutting board with visible grain. A plain white table gives the software nothing to lock
  on to.
- Lay a ruler or an A4 sheet flat beside the bowl to confirm the scale.
- Walk round the plate. Don't use a turntable, because the background must stay still.

**Camera (M50 II)**
- Mode M (manual). ISO 100–400, f/8, shutter 1/100 s or faster handheld. Use a tripod if the
  light is low.
- Fixed white balance (Daylight or a custom setting, **not** Auto). Fixed exposure for every shot.
- RAW + JPEG fine. Lens at 24–35 mm; don't zoom between shots.
- Autofocus on the food for each shot is fine. Turn off any picture style that sharpens or
  boosts colour (use Neutral).

**Coverage: about 100 photos**
- Ring 1, low (~15° above the table): 30 shots, one every ~12°, whole bowl in frame.
- Ring 2, mid (~45°): 30 shots, every ~12°.
- Ring 3, high (~70°): 24 shots, every ~15°.
- Overhead: 6 shots straight down, turning slightly between them.
- Close-ups: 10–15 shots filling the frame with gnocchi, from all sides, each overlapping
  its neighbours by about two-thirds.
- Every photo sharp: zoom in on the camera's screen to check a few as you go.

**Optional, for a flawless bowl:** after eating, wash and dry the bowl and shoot one quick
ring of the empty bowl in the same spot.

**Hand-off:** copy the full-resolution JPEGs (and RAWs, if any) into `tools/gnocchi/orbit/`
in this repo (git-ignored).

## 2. Processing (on the laptop)

Install first:
- [COLMAP](https://github.com/colmap/colmap/releases), the Windows "nocuda" build. Its own
  dense step needs CUDA, so it only does camera alignment here.
- [OpenMVS](https://github.com/cdcseacave/openMVS/releases), Windows build. It does the
  dense cloud, mesh, refinement and texturing on the CPU.
- Blender 4.x with its MCP add-on connected to Claude (already set up).

Steps Claude will run and check:
1. **Align (COLMAP):** feature extraction, exhaustive matching, sparse reconstruction,
   undistortion. Every photo should register; the plate and the ruler come out sharp in the
   sparse cloud.
2. **Reconstruct (OpenMVS):** `InterfaceCOLMAP` → `DensifyPointCloud` → `ReconstructMesh` →
   `RefineMesh` → `TextureMesh`, at full resolution. This takes about 1–2 hours on CPU.
3. **Clean up (Blender):**
   - Scale to the 23 cm rim (checked against the ruler).
   - Level it on the table and cut away the table and newspaper.
   - Rebuild the bowl as a clean surface. Keep its painted pattern from the scan, or from the
     empty-bowl ring if you shot one.
   - Fill holes under the walnuts with real neighbouring texture.
4. **Bake (Blender):**
   - Decimate the food to about 150k triangles. Bake the full-detail mesh into a normal map.
   - Colour: 4096², de-lit by dividing out the scan's own shading, so it holds pure colour.
   - Roughness: glossy sauce, satin gnocchi, matte walnuts, glazed bowl. Ambient occlusion
     goes into the creases.
5. **Export:** GLB and USDZ through `tools/model_io.py`, the same writers as the other dishes.
   - glTF Validator 0 errors / 0 warnings; the 28 OpenUSD validators pass.
   - Under ~10 MB each.
   - Renders compared side by side with the photos from the same angles.
6. **Ship:** replace `models/gnocchi.glb` / `.usdz` and the menu thumbnail; record the steps
   in the README.
