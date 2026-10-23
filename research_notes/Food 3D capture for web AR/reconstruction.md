# Reconstruction tools for plated food (60–120 photos), as of Oct 2026

## Photogrammetry: COLMAP+OpenMVS, Meshroom, RealityScan, Apple Object Capture

### Takeaway
RealityScan (formerly RealityCapture) is free for anyone under $1M revenue, has a documented CLI, and a Linux CLI build that runs under Wine. That makes it the strongest free option for scripted, textured meshes. COLMAP (BSD, global mapper since 4.x) plus OpenMVS is the fully open-source scriptable stack. Meshroom (MPL-2.0) is open and node-based. Apple Object Capture accepts DSLR JPGs but needs a Mac.

### Cited Findings
- RealityScan is free for individuals and companies under $1M USD revenue in the past 12 months, and for students and educators. Seats above that threshold cost $1,250 per seat per year. — [RealityScan license](https://www.realityscan.com/license); [Download page](https://www.realityscan.com/en-US/download)
- RealityScan 2.0 (June 2025) is the rebranded RealityCapture. — [RealityScan news](https://www.realityscan.com/news/realityscan-20-new-release-brings-powerful-new-features-to-a-rebranded-realitycapture)
- RealityScan 2.1 (Nov 2025) added CLI support with full documentation. Epic also ships a Linux command-line build in a bundled Wine environment, and advises against using the desktop UI on Linux. — [CG Channel](https://www.cgchannel.com/2025/11/epic-games-releases-realityscan-2-1/); [Radiance Fields](https://radiancefields.com/realityscan-version-2-1-released)
- COLMAP is BSD-licensed, but its dependencies are licensed separately and can affect the license of a build. — [COLMAP license](https://colmap.readthedocs.io/en/latest/license.html)
- GLOMAP (global SfM, BSD-3) was integrated into COLMAP 4.0. COLMAP 4.2.0 adds multi-component global mapping and LoMa learned matching. — [Radiance Fields](https://radiancefields.com/colmap-4.2.0-adds-multi-component-global-mapping-and-loma-learned-matching); [GLOMAP listing](https://awesome.ecosyste.ms/projects/github.com%2Fcolmap%2Fglomap)
- Meshroom 2025.1.0 (Aug 18, 2025) is MPL-2.0. It ships AliceVision 3.3.0 and adds:
  - a plugin system and pipeline templates
  - object/turntable/360° reconstruction pipelines
  - color calibration and RAW-to-EXR
  - multi-view photometric stereo

  A 2026 nightly build exists. — [GitHub release](https://github.com/alicevision/Meshroom/releases/tag/v2025.1.0); [MPC](https://www.mpcvfx.com/en/news/mpc-celebrates-the-release-of-meshroom-2025-1-0/); [Digital Production](https://digitalproduction.com/2025/08/20/meshroom-2025-1-templates-plugins-and-photometric-stereo/)
- Apple Object Capture (RealityKit `PhotogrammetrySession`, macOS 12+) accepts images from a DSLR or drone. Details:
  - Input formats: HEIC, JPG and PNG. Output formats: USDZ, USDA and OBJ.
  - Apple ships a CLI sample, HelloPhotogrammetry.
  - Apple recommends 20–200 images and advises avoiding transparent or highly reflective areas.
  - DSLR photos lack the depth data needed for real-world scale.

  — [WWDC21 notes](https://wwdcnotes.com/notes/wwdc21/10076); [Apple docs](https://developer.apple.com/documentation/realitykit/photogrammetrysession.md)
- Forum users report internal errors 4011/4012 with some image sets on macOS. — [Apple forums](https://developer.apple.com/forums/thread/794169)

### Inferences
- For a script-driven proof of concept on rented Linux GPUs, the two options are COLMAP/GLOMAP → OpenMVS (fully open source) and the RealityScan CLI under Wine (free, but the EULA should be checked for cloud/server use).
- Object Capture produces USDZ directly, which suits iOS Quick Look, but needs Mac hardware (a rented Mac or a local machine).
- Glossy sauces and specular highlights break multi-view stereo. A matte, diffuse capture setup (cross-polarization, even lighting) will likely matter more than which tool you pick.

### Gaps
- No current OpenMVS release or license info was retrieved. From memory it is AGPL-3.0, which is unverified.
- No food-specific benchmark of these tools was found.
- No reliable per-dish timings were found. The RealityScan EULA terms for cloud/headless use were not verified.

## Gaussian splatting / neural → mesh (3DGS, 2DGS, SuGaR, PGSR, GOF, gsplat, splatfacto, Postshot)

### Takeaway
Splats render glossy and thin food more realistically than photogrammetry. Extracted meshes (2DGS, PGSR or GOF, usually via TSDF fusion) are improving, but they are noisier and less clean than photogrammetry meshes, especially for thin structures. Licensing also matters: the Inria 3DGS code and its derivatives are non-commercial only.

### Cited Findings
- Reported mean DTU Chamfer distances (lower is better), from one table: 2DGS ≈0.80, GOF ≈0.74, PGSR ≈0.52. — [GS-2M paper](https://arxiv.org/pdf/2509.22276)
  - Another paper reports a different ordering (2DGS 0.823, GOF 0.804), so protocols differ between papers. — [2DGH](https://arxiv.org/pdf/2408.16982)
- 2DGS, GOF and PGSR meshes are typically extracted after training by TSDF fusion. Depth-fusion pipelines are sensitive to view-dependent depth noise and struggle with thin structures. — [arXiv 2604.07337](https://arxiv.org/pdf/2604.07337); [arXiv 2506.18575](https://arxiv.org/pdf/2506.18575)
- DTU ground truth lacks the backs of objects, which biases the metric toward open or incomplete meshes. — [arXiv 2604.07337](https://arxiv.org/pdf/2604.07337)
- Mesh-refinement add-ons improve 3DGS, GOF, 2DGS and PGSR meshes alike. — [arXiv 2511.03950](https://arxiv.org/pdf/2511.03950)
- Gaussian splatting reduced reflection artifacts compared with photogrammetry while preserving geometric accuracy in non-reflective areas. — [ISPRS 2024](https://isprs-archives.copernicus.org/articles/XLVIII-2-W8-2024/371/2024/)
- Photogrammetry struggles with reflective, transparent or fine geometry. — [Pix-Pro (vendor)](https://www.pix-pro.com/blog/gaussian-vs-photogrammetry)
- The Inria/MPII 3DGS license is non-commercial (research and evaluation) only, and derivative submodules such as diff-surfel-rasterization (2DGS) and simple-knn carry the same license. — [ScanCode LicenseDB](https://scancode-licensedb.aboutcode.org/gaussian-splatting-2024.html); [diff-surfel-rasterization LICENSE](https://huggingface.co/spaces/yslan/GaussianAnything-AIGC3D/raw/a0896dd52a14e7ee5bdd71ff4c5aa0b6682ec79e/submodules/diff-surfel-rasterization/LICENSE.md)
- Compared with the original 3DGS implementation on an A100 (MipNeRF360), gsplat uses up to 10% less training time and 4× less memory. Nerfstudio's splatfacto uses gsplat (`ns-train splatfacto --data <colmap>`). — [gsplat paper](https://arxiv.org/html/2409.06765v1); [Spheron (vendor blog)](https://www.spheron.network/blog/deploy-3d-gaussian-splatting-gpu-cloud/)
- Postshot v1.0 pricing (Aug 2025):
  - Free: unlimited training, but saved files are usable only in the free tier.
  - Indie (€17/month): unlocks PLY splat export.
  - Studio: €39/month.

  Postshot imports COLMAP and RealityScan poses. — [Radiance Fields](https://radiancefields.com/postshot-launches-v1-0-out-of-beta)

### Inferences
- A good hybrid pipeline for menu AR: run one SfM pass (COLMAP/GLOMAP or RealityScan), then:
  - build a photogrammetry mesh with baked texture for the glTF/USDZ AR deliverable;
  - train a splat (gsplat/splatfacto, Apache-2.0 per its repo, unverified here) for the in-browser hero view.
- PGSR or 2DGS meshes are worth trying on glossy dishes, but they are non-commercial code. For a commercial menu product, use them for evaluation only or reimplement on gsplat (nerfstudio has 2DGS-style options).
- SuGaR is older and also built on the Inria code; no current benchmarks were retrieved for it.

### Gaps
- Postshot mesh export was not confirmed.
- No license text was retrieved for PGSR, GOF or SuGaR.
- The gsplat Apache-2.0 license was not verified from a primary source.
- No study tests texture quality of splat-derived meshes on food.

## Browser AR for splats without an app

### Takeaway
Mesh AR without an app works today: model-viewer uses glTF for Android Scene Viewer/WebXR and USDZ for iOS Quick Look. Splat AR without an app is partial: WebXR splat viewers (SuperSplat/PlayCanvas, Spark) work on Android Chrome and Quest, but iOS Quick Look does not render splats. KHR_gaussian_splatting is still a Release Candidate.

### Cited Findings
- Khronos announced the KHR_gaussian_splatting release candidate on Feb 3, 2026. Status details:
  - It was still listed as a Release Candidate in July 2026.
  - SPZ compression is a child extension, KHR_gaussian_splatting_compression_spz_2.
  - Early adopters include CesiumJS and Esri.

  — [Khronos press release](https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release); [Iconem lineage doc](https://docs.3dtiled.iconem.com/docs/references/3dtiles-gltf/3dgs-extension-lineage); [AEC Magazine](https://aecmag.com/news/khronos-announces-gltf-gaussian-splatting-extension/)
- iOS can save splats in USDZ but cannot display them in AR Quick Look. An Apple DTS engineer said Quick Look "may not support this technique" and suggested filing an enhancement request. — [Apple forums](https://developer.apple.com/forums/thread/804604)
- Needle Engine works around this by falling back to a mesh for iOS AR. — [Needle forum](https://forum.needle.tools/t/gaussian-splat-ar-on-iphones/2515)
- WWDC26 covered OpenUSD gaining Gaussian splat support. This is format support, not confirmed Quick Look display. — [WWDC26 session 285](https://developer-rno.apple.com/videos/play/wwdc2026/285)
- The SuperSplat/PlayCanvas viewer supports AR, tested on Quest 2/3 and Android phones. It reads PLY, compressed PLY and SOGS. — [PlayCanvas blog](https://blog.playcanvas.com/publish-your-gaussian-splats-with-supersplat/); [PlayCanvas docs](https://developer.playcanvas.com/user-manual/gaussian-splatting/viewing)
- Spark (World Labs) is an open-source Three.js/WebGL2 splat renderer for desktop, mobile and WebXR. — [Spark GitHub](https://github.com/sparkjsdev/spark)

### Inferences
- iPhone users are a large share of restaurant customers, so the AR deliverable should be a mesh (GLB + USDZ). Splats can be an optional 3D view in the page and AR on Android/WebXR.

### Gaps
- No evidence was found of model-viewer splat support. The model-viewer repo was not checked directly.
- No Spark AR test on Android was found.

## Cloud GPU feasibility and cost per dish

### Takeaway
Everything except Object Capture can run on rented Linux GPUs. COLMAP, OpenMVS, gsplat and nerfstudio are native; RealityScan runs as a CLI under Wine. No sourced per-dish timings or costs were found.

### Cited Findings
- The RealityScan Linux CLI build runs in Wine. — [CG Channel](https://www.cgchannel.com/2025/11/epic-games-releases-realityscan-2-1/)
- gsplat benchmarks were run on an A100 and it uses 4× less memory than the original 3DGS, which helps on smaller GPUs such as Colab T4s. — [gsplat paper](https://arxiv.org/html/2409.06765v1)
- A vendor blog describes deploying 3DGS on cloud GPUs with splatfacto. — [Spheron](https://www.spheron.network/blog/deploy-3d-gaussian-splatting-gpu-cloud/)

### Inferences
- These are rough estimates (not from a source) for 60–120 photos on one modern GPU:
  - SfM: minutes.
  - Splat training: about 10–30 minutes.
  - Splat cost: under $1 on a ~$1–2/hour rented GPU.

  Benchmark on real dishes before relying on these numbers.
- Free Colab and HF Spaces sessions have time limits and preemption, so they are fine for demos but unreliable for batch processing.

### Gaps
- No sourced wall-clock or cost figures per dish.
- Colab and HF Spaces limits and pricing were not retrieved.
