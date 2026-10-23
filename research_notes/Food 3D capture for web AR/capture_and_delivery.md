# Food 3D Capture for Web AR — Capture, Blender Post-Processing, and Delivery (as of Oct 2026)

Note: developers.google.com and modelviewer.dev could not be fetched directly in this session (DNS failure). Their content is cited from search-result snippets and mirrors, so check it against the live pages.

## (a) Capturing food for photogrammetry with a dedicated camera

### Takeaway
Use a turntable with a fixed camera, a dark matte non-reflective surface, and cross-polarized or very diffuse even lighting. Use fixed manual exposure and focus, high overlap (roughly 60-85%, about 10-15 degrees per shot, several camera heights), and a clean, sharp dataset. Gloss and translucency are the main enemies. I found no photogrammetry guide written specifically for food, so food guidance below is adapted from object and cultural-heritage scanning.

### Cited Findings
- **Gloss: cross-polarization.** Polarizer film goes over the lights and a polarizing filter goes on the lens. Rotate the lens filter until it is perpendicular to the film to remove specular highlights. — [Pix-Pro](https://www.pix-pro.com/blog/cross-polarization); [80.lv miska jug (RealityCapture)](https://80.lv/articles/using-photogrammetry-to-create-a-hungarian-miska-jug-in-realitycapture/)
- Turntables give the best results with cross-polarized lighting. Without it, a turntable setup needs perfectly diffused, uniform lighting from all angles. Cross-polarization costs a lot of light, so plan for strong continuous light or strobes. — [Pix-Pro](https://www.pix-pro.com/blog/cross-polarization)
- **Example cross-polarized turntable capture:** 3 cross-polarized lights at 6000K, f/20, 1/60 s, ISO 400. — [Fab listing](https://www.fab.com/listings/1a1c0e81-3cd2-487c-8bc1-a743350392b8)
- **Food glare:** glare can be avoided on flat surfaces by choosing the viewing angle, but this is not possible for curved food products. That supports polarization over angle tricks. — [Springer chapter (agrifood imaging)](https://link.springer.com/chapter/10.1007/978-3-031-35593-6_7)
- **Clean dataset:** you need enough uniform light, enough overlap, and a focused subject. Use a black velvet or other non-reflective turntable and backdrop. — [Pix-Pro](https://www.pix-pro.com/blog/cross-polarization)
- **Lighting:** diffuse, flat, and as even as possible. — [Swedish National Heritage Board (RAA)](https://www.raa.se/in-english/outreach-and-exhibitions/guide-for-publishing-3d-models/equipment-and-procedures-for-photogrammetry/)
- **Overlap:** sources range from 50-60% ([UNSW Making](https://www.making.unsw.edu.au/learn/photogrammetry/)) to at least 70% ([Oxford Brookes](https://telsupport.brookes.ac.uk/articles/how-do-i-create-a-3d-model-using-photogrammetry/)) to more than 85% across three heights at 15-degree steps ([UCLA](https://dal.ucla.edu/seams/node/477)). A common turntable step is about 10 degrees per photo. — [Dartmouth](https://dreamstudio.dartmouth.edu/?p=630)
- **Shiny, transparent, and smooth surfaces capture poorly.** Heritage guides suggest coating with talc or plaster, which is not acceptable for food you plan to serve. — [UNSW](https://www.making.unsw.edu.au/learn/photogrammetry/); [Northwestern, shiny object](https://mti.it.northwestern.edu/2022/04/photogrammetry-of-a-shiny-object/)
- **Turntable masking:** you can mask once per camera position using a shot without the subject. With "exclude stationary tie points" enabled in alignment (Metashape), masking is often unnecessary. — [Dartmouth](https://dreamstudio.dartmouth.edu/?p=630)
- **Real dish example:** a sloppy joe captured with about 50 photos on a Sony a7III with a 24mm lens at f/6.3. — [Sketchfab](https://sketchfab.com/3d-models/philly-cheesesteak-sloppy-joes-427c2651c1ad4776a5dfa263ecb7cccf)
- **AR menu precedent:** TU Delft thesis on an AR menu using photogrammetry with 1:1-scale dishes. Kabaq/Bareburger and QReal/Denny's deployments used QR codes or Snapcodes on printed menus. — [TU Delft](https://repository.tudelft.nl/file/File_b0fd7a17-302a-4e21-9514-eec09dedc82f); [AR Insider](https://arinsider.co/2023/04/05/how-can-restaurants-use-ar/)
- **Background masking tools:**
  - BiRefNet (the basis of BRIA RMBG v2.0) is a high-resolution dichotomous segmentation model that is strong on fine edges. It can fail where the subject blends into the background. — [ComfyUI docs](https://docs.comfy.org/tutorials/utility/remove-background-birefnet); [BRIA](https://bria.ai/introducing-the-rmbg-v2-0-model-the-next-generation-in-background-removal-from-images)
  - rembg wraps U2-Net, BiRefNet, and SAM models and outputs greyscale confidence masks. — [tessl rembg](https://tessl.io/registry/tessl/pypi-rembg/2.0.0/evals)
  - SAM 2 is promptable (point or box) rather than automatic. — [image_vision docs](https://hexdocs.pm/image_vision/background.html)

### Inferences
- **Practical food recipe:**
  - Mount the camera on a tripod and the plate on a turntable over black velvet.
  - Use 2-3 polarized LED panels with a circular polarizer on the lens.
  - Set manual exposure, white balance, and focus, shoot RAW, and include a gray card or color checker in the first frame.
  - Work at f/8-f/11 on APS-C or full frame, which balances depth of field against diffraction. Focus stacking is rarely practical with food that changes over time.
  - Shoot 3 heights × 24-36 steps, about 72-108 frames. Add a top-down ring for sauces and garnish.
- **Speed matters.** Shoot quickly, within minutes, before sauces dull, steam dissipates, or garnish wilts. Spritz glossy sauces only if the plan is to bake gloss back in as roughness.
- Steam cannot be reconstructed. Shoot after it subsides and add it later as a VFX or particle effect if wanted.
- Plate rims are glossy and featureless. Cross-polarization helps. Optionally replace the plate with a modeled plate in Blender.
- Thin garnish (herbs, microgreens) will be lost or turn to noise. Consider adding it back as alpha-card geometry.
- **Scale:** put coded targets or a printed ruler on the turntable (outside the plate), or measure the plate diameter and scale in post.
- Mask with BiRefNet or rembg in batch, then use SAM 2 to fix failures.
- **Free reconstruction software:** Meshroom/AliceVision, COLMAP + OpenMVS, and RealityCapture (free for users under a revenue threshold). Check current licensing.

### Gaps
- No food-specific photogrammetry guide found. There were no published photo counts for sauces or garnish beyond the single Sketchfab example.
- No sources found on focus stacking for photogrammetry or on SAM 2 used in photogrammetry pipelines specifically.

## (b) Post-processing in Blender and Blender MCP

### Takeaway
Blender MCP (ahujasid/blender-mcp, PyPI now `mcp-for-blender`) is an LLM-to-live-Blender bridge. Its core is `execute_blender_code`, which runs arbitrary Python, plus viewport inspection and asset or AI-generation imports. It is good for scripted cleanup, decimation, baking, and export. It does not do photogrammetry reconstruction, which has to happen in Meshroom, COLMAP, RealityCapture, or similar. For repeatable batches, a headless `blender -b -P script.py` run with bpy is more robust.

### Cited Findings
- **Tools:**
  - `execute_blender_code` runs arbitrary Python in the session.
  - `look` shows the viewport or a camera view, multi-angle sheets, or render modes.
  - `get_scene_info` gives a scene summary.
  - `generate_3d` uses Tripo, Hunyuan3D, Hyper3D Rodin, or Meshy.
  - `search_assets` and `import_asset` cover Poly Haven, Sketchfab, and Poly Pizza.
  - `get_addon_status` and telemetry controls.
  — [GitHub ahujasid/blender-mcp](https://github.com/ahujasid/blender-mcp)
- **Package and version:** the package was renamed to `mcp-for-blender` and is upgraded with `uvx mcp-for-blender@latest update`. It works with Claude Desktop/Code, Cursor, VS Code, Codex, and others, and can run in Docker. — [GitHub](https://github.com/ahujasid/blender-mcp)
- **Limits and warnings:**
  - Complex tasks should be split into steps, and long requests can time out.
  - The socket server has no authentication or encryption, so keep it on localhost.
  - Run only one MCP client instance at a time.
  - Poly Haven downloads can freeze the UI.
  - A minimal anonymous usage record is collected by default. Opt out with `DISABLE_TELEMETRY=true`.
  - Save your work before running code.
  — [GitHub](https://github.com/ahujasid/blender-mcp)
- **Blender USD/USDZ export lacks some shader options**, for example occlusion is lost. — [AOUSD forum](https://forum.aousd.org/t/obj-gltf-to-usdz-usdzconvert-discontinued/1648)

### Inferences
- **Good uses of MCP for this project:**
  - Import the photogrammetry OBJ or PLY.
  - Delete floaters and turntable remnants.
  - Decimate or run Remesh/QuadriFlow to a target triangle count.
  - Smart UV project or xatlas-style unwrap.
  - Bake high-poly color and normal maps to low-poly with Cycles.
  - Set the origin to the plate bottom-center, apply scale in meters, and export GLB.
  - `look` makes visual QA checks fast.
- **Not suitable:** reconstruction itself, or AI `generate_3d` output as the hero asset. Generated output is not a faithful scan of the real dish, which raises accuracy and truth-in-menu concerns.
- **Production path:** turn the MCP-proven script into a headless bpy script so it is reproducible and needs no LLM or GUI.

### Gaps
- No other Blender MCP implementations were researched in detail.

## (c) Optimizing and delivering to browser AR

### Takeaway
Ship one GLB via `<model-viewer>` with `ar ar-modes="webxr scene-viewer quick-look"`, and let it auto-generate USDZ for iOS or supply `ios-src`. Practical budgets are about 30-100k triangles, 2048 px textures at most, and a GLB under about 5-10 MB. Optimize with gltf-transform. The QR code links to an HTTPS product page.

### Cited Findings
- **model-viewer `ar-modes`** accepts `webxr`, `scene-viewer`, and `quick-look`. If `ios-src` is omitted, model-viewer has auto-generated USDZ on tap since v1.7, including scene-graph edits. Supply `ios-src` (.usdz or .reality) when the auto output falls short, for example for animations. WebXR needs HTTPS and the iframe `xr-spatial-tracking` permission. — [Khronos news on model-viewer 1.7](https://www.khronos.org/news/permalink/model-viewer-1.7-released-with-auto-generation-of-usdz-on-the-fly); [tessl model-viewer AR docs](https://tessl.io/registry/tessl/npm-google--model-viewer/files/docs/ar.md); [Google ARCore model-viewer page](https://developers.google.com/ar/develop/webxr/model-viewer?hl=en)
- **Scene Viewer:**
  - Intent base URL is `https://arvr.google.com/scene-viewer/1.0?file=...`.
  - `resizable=false` locks scale in AR, which keeps a dish at true 1:1 size.
  - `mode=ar_only` is used with the `com.google.ar.core` package.
  - Scene Viewer falls back to 3D for non-AR devices.
  — [Google Scene Viewer docs](https://developers.google.com/ar/develop/java/scene-viewer)
- **Scene Viewer-style budgets (third-party, "adapted from Google"):** 2048×2048 maximum texture size, 1 UV set per mesh, sRGB. Sources disagree on the rest:
  - Vyking: 30-50k triangles and 5 MB.
  - Power.xyz: 30k triangles and 10 MB.
  — [Vyking](https://docs.vyking.io/docs/model-requirements); [Power.xyz](https://power.xyz/resources/support/doc/3d-asset-specs/)
- **iOS Quick Look rule of thumb (WWDC 2018):** about 100k polygons, one set of 2048² PBR textures, and animation of 10 s or less. Apple gives no current official file-size limit. Memory is what triggers "too large": a 1.6 MB USDZ with 4096² textures failed. — [Apple Dev Forums 693867](https://developer.apple.com/forums/thread/693867); [Apple Dev Forums 118710](https://developer.apple.com/forums/thread/118710); [Netguru](https://netguru.com/blog/ar-quick-look-and-usdz)
- **gltf-transform:**
  - `gltf-transform optimize in.glb out.glb --texture-compress webp` runs the whole pipeline.
  - Individual commands are `meshopt`, `draco`, `simplify`, `resize`, `webp`, `uastc`, and `etc1s` (the last two produce KTX2/Basis).
  - Meshopt also compresses morph targets and animation and decodes faster. Draco gives smaller static geometry but decodes more slowly.
  — [npm @gltf-transform/core](https://classic.yarnpkg.com/en/package/@gltf-transform/core); [Cinevva guide](https://app.cinevva.com/guides/optimize-glb-for-web)
- **GLB to USDZ conversion:**
  - Apple retired its Python usdzconvert.
  - Reality Converter (macOS) works for GLB but sometimes fails with an opaque "conversion failed" error.
  - Blender USD export loses occlusion.
  - GLB is a safer route than OBJ, which came in at 1/100 scale in one report.
  - Free in-browser converters exist (Innerscene).
  — [AOUSD forum](https://forum.aousd.org/t/obj-gltf-to-usdz-usdzconvert-discontinued/1648); [Apple Dev Forums 685448](https://developer.apple.com/forums/thread/685448); [Apple Dev Forums 651921](https://developer.apple.com/forums/thread/651921); [Innerscene](https://www.innerscene.com/tools/glb-to-usdz)

### Inferences
- **Suggested dish target:**
  - 20-60k triangles.
  - One material with baseColor + normal (+ ORM) at 2048 or 1024.
  - Meshopt geometry.
  - Textures: WebP for the smallest download, or KTX2 (UASTC for normal maps, ETC1S for color) to save GPU memory.
  - Final GLB of about 2-5 MB.
- **Example pipeline:** `gltf-transform simplify` (or decimate in Blender) → `resize --width 2048 --height 2048` → `meshopt` → `webp`/`etc1s`. KTX2 and Meshopt need decoders in model-viewer. Verify that iOS auto-USDZ handles compressed inputs.
- **Real-world scale:** model in meters with origin at the plate base. Use `resizable=false` for Scene Viewer, and model-viewer's `ar-scale="fixed"`, which I could not verify directly on modelviewer.dev.
- **QR flow:** printed QR → HTTPS menu-item page with `<model-viewer>` → AR button routes to WebXR, Scene Viewer, or Quick Look by device.

### Gaps
- modelviewer.dev and developers.google.com pages could not be fetched. `ar-scale` and `ar-placement` details and the official Scene Viewer glTF requirements (file size, polygon counts) are unverified.
- No current (2025-2026) official Apple numeric limits found.
- Unverified whether model-viewer's USDZ export preserves KTX2 or Meshopt-compressed assets. Many pipelines keep an uncompressed or WebP variant for iOS.
