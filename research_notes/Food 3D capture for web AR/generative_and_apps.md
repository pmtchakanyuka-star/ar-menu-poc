# Generative AI image-to-3D and consumer scanning apps for food 3D models (as of Oct 2026)

## Open models (Hunyuan3D, TRELLIS, TripoSR, SF3D/SPAR3D, InstantMesh): licenses, GPU, fidelity, multi-view

### Takeaway
The truly permissive models are TRELLIS.2 (MIT) and TripoSR (MIT). Hunyuan3D 2.x uses a Tencent non-commercial or territorial license, and SF3D/SPAR3D are free only for organizations under $1M revenue. Hunyuan3D 3.x appears to be hosted only. All of these models infer the unseen sides of an object, so they produce a believable dish rather than a faithful copy of a real one. Multi-view variants (Hunyuan3D-2mv, Hunyuan 3.x Pro MultiView with 4 to 8 images) narrow that gap but do not replace photogrammetry.

### Cited Findings
- Hunyuan3D-2.1 code headers state it is licensed under the "TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT" — [HF Hunyuan3D-2.1 multiview_utils.py](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/38e83be15b2054c29a8ae2e2ef69daf68b7dd923/hy3dpaint/utils/multiview_utils.py); a third-party wiki says Hunyuan3D 2.0 is Apache-2.0, which conflicts — [ComfyUI Wiki](https://comfyui-wiki.com/en/models/hunyuan3d)
- Hunyuan3D 2.0 added MV (multi-view) and MV Mini variants — [PixelSham](https://www.pixelsham.com/2025/06/14/tencent-hunyuan3d-2-0-adds-mv-multi-view-and-mv-mini)
- Hunyuan 3D 3.0/3.1 Pro MultiView is offered as a hosted model that takes up to 4 (3.0) or up to 8 (3.1) reference images. No open weights were found — [Scenario 3.0 MV](https://www.scenario.com/models/hunyuan-3d-30-pro-multiview); [Scenario 3.1 MV](https://www.scenario.com/models/hunyuan-3d-31-pro-multiview); [Scenario help](https://help.scenario.com/articles/5967392966-hunyuan-3d-models-the-essentials)
- The Hunyuan3D pro tier via API offers configurable polygon counts and up to 4K PBR textures. The rapid tier takes about 2 to 3 minutes — [Atlas Cloud](https://www.atlascloud.ai/ja/models/tencent/hunyuan3d-rapid/image-to-3d)
- TRELLIS.2: Microsoft, 4B parameters, released Dec 2025 — [ComfyUI Wiki news](https://comfyui-wiki.com/news/2025-12-18-microsoft-trellis2-3d-generation). It is MIT licensed and commercial use is allowed — [Webkul](https://webkul.com/blog/trellis-2/); [Cinevva](https://app.cinevva.com/news/2026-03-20-trellis-2-microsoft-open-source-3d)
- TRELLIS.2 VRAM: about 24GB is commonly cited — [Webkul](https://webkul.com/blog/trellis-2/). Another source gives 16GB for 512³, 40GB for 1024³ and 80GB for 1536³ (less established) — [Codesota](https://www.codesota.com/news/trellis-2-3d-generation). Reported H100 times are about 3s at 512³, 17s at 1024³ and 60s at 1536³ — same sources
- TripoSR: weights and code are MIT and allow commercial use. It was trained on a CC-BY Objaverse subset and runs at about 0.5s on an A100 — [Stability AI](https://stability.ai/news/triposr-3d-generation). It needs about 6GB VRAM per image — [Paperspace](https://blog.paperspace.com/triposr-rapid-3d-object-synthesis-from-single-images)
- SF3D: organizations with US$1M or more in annual revenue need an enterprise license — [HF stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d). Under $1M, the Community License allows commercial use — [Stability membership](https://stability.ai/membership). SPAR3D is described as being under the Community License — [iMagazine](https://imagazine.pl/2025/01/28/stability-ai-prezentuje-spar3d-generatywna-ai-tworzaca-strukturalne-obiekty-3d-z-dwuwymiarowych-obrazow)
- Generative models for food: diffusion builds coarse shape first and fine texture last. Models struggle with thin, continuous structures (noodles) and repeating textures (seeds) — [AI Weekly](https://aiweekly.co/alerts/why-ai-generated-food-images-look-so-wrong-per-researchers). A 2D food study found a 15% hallucination rate, with omissions such as missing lemon wedges — [arXiv 2507.06571](https://arxiv.org/pdf/2507.06571)

### Inferences
- From a single photo, these models invent the back and underside of a dish, and how they render garnish, sauce sheen and plate rims varies from run to run. That is acceptable for a "representative" dish but risky where an "as served" claim matters.
- Multi-view input (Hunyuan3D-2mv, or the hosted 3.x MultiView) reduces how much the model invents about the back of the dish, but it is still conditioning, not measured reconstruction. Scale and portion size are not guaranteed.
- For a free, commercially safe self-hosted pipeline: TRELLIS.2 (24GB+ GPU) or TripoSR (6GB, lower quality).

### Gaps
- No published benchmark of image-to-3D fidelity on plated food was found.
- InstantMesh license and VRAM could not be verified (believed to be TencentARC, Apache-2.0; unconfirmed).
- The exact territorial carve-outs in the Hunyuan3D license (e.g. EU/UK/KR exclusions) were not checked against the official LICENSE file.

## Higgsfield ("Hixfield") 3D feature

### Takeaway
Higgsfield is mainly a video and image generation platform. It does offer image-to-GLB generation (a `generate_3d` tool in its MCP connector) and a Blender add-on that places generated geometry using cloud compute. No public source names the underlying 3D model or gives 3D-specific pricing.

### Cited Findings
- Higgsfield's Blender add-on (Aug 2026) places real, editable geometry in the scene, runs on Higgsfield's servers rather than the user's GPU, supports Blender 4.2 to 5.1, and requires a logged-in account — [WikiDocs blog (KR)](https://wikidocs.net/blog/@openwiki/29644/)
- The Higgsfield MCP server's own instructions (observed in this research session's tool configuration) say: "Use generate_3d to turn an image into a 3D GLB mesh" and recommend `models_explore` to pick models. This is a first-party tool description, not a web page.
- Earlier coverage frames Higgsfield around video camera "Motion Controls" — [Meshed3D newsletter](https://maily.so/meshed3d/posts/2qzplgder4x)

### Inferences
- It is likely a credit-based wrapper around a third-party image-to-3D model (e.g. Hunyuan/Tripo-class), so fidelity for food would match those single-image generators (plausible, not faithful). This is unconfirmed.

### Gaps
- The higgsfield.ai site could not be reached from this environment (DNS failure). The underlying model, credit cost per 3D generation, and any multi-view input were not verified.

## Scanning apps and AR-menu SaaS used by creators

### Takeaway
For free capture with GLB and USDZ export, KIRI Engine is the strongest option: since version 4.0 the free tier has unlimited exports and up to 150 photos per scan. Polycam's free tier is limited to glTF export, with USDZ on paid tiers. Scaniverse keeps on-device scanning free, and RealityScan is free under $1M revenue. Turnkey AR-menu vendors (ARmenu UK, Plattr, Kabaq/QReal) scan or model dishes for you, priced per dish or by subscription.

### Cited Findings
- KIRI Engine 4.0 (Sept 2025) removed the 3-exports-per-week cap. All users get unlimited export to OBJ, FBX, STL, GLB, glTF, USDZ and PLY. Basic users get 150 photos per scan and Pro users 500 — [All3DP](https://all3dp.com/4/kiri-engines-free-4-0-update-gives-you-unlimited-photogrammetry-exports/); [KIRI Basic vs Pro](https://www.kiriengine.app/blog/kiri-engine-basic-vs-pro)
- KIRI's web version accepts DSLR or drone photos — [3DSourced](https://www.3dsourced.com/3d-scanners/kiri-engine-review/). KIRI's comparison table lists web photogrammetry upload as a Pro feature (possible conflict) — [KIRI Basic vs Pro](https://www.kiriengine.app/blog/kiri-engine-basic-vs-pro)
- Polycam free plan: "only export GLTF files". USDZ and the other formats are on paid plans — [Polycam Help](https://learn.poly.cam/hc/en-us/articles/27756102599572). Paid pricing is about $26.99/month or $199.99/year — [SkyeBrowse](https://www.skyebrowse.com/news/posts/polycam-review); a $30/month Basic plan is also reported (conflict) — [CostBench](https://costbench.com/software/ai-3d-generation/polycam-ai/hidden-costs/)
- Scaniverse: on-device scanning is free. New Plus ($20/month) and Pro ($50/month) cloud tiers have been added — [SkyeBrowse](https://www.skyebrowse.com/news/posts/best-photogrammetry-app)
- RealityScan mobile: free, with AR guidance. Processing happens in the cloud — [App Store](https://apps.apple.com/us/app/realityscan-3d-scanning-app/id1584832280?uo=4); [HKU wiki](https://bookstack.hku.nl/books/3d-scan/page/realityscan). It is free for organizations under $1M revenue — [SkyeBrowse](https://www.skyebrowse.com/news/posts/best-photogrammetry-app)
- ARmenu (UK): from £49 per dish, £149 for 5 dishes, £299 for 15 dishes, including 3D modelling, hosting and QR codes. It claims a 2.4× order uplift (vendor claim). Kabaq is now QReal and claims a 20 to 26% AOV increase (vendor-reported) — [Petpooja](https://blog.petpooja.com/glossary/ar-menu/)
- Indian vendor quotes for 3D modelling run Rs 1,500 to 4,000 per dish (early 2026) — [Petpooja](https://blog.petpooja.com/glossary/ar-menu/)
- Plattr sends a team with professional scanning equipment. Diners scan a QR code to view dishes in 3D without an app — [Plattr](https://plattr.it.com/)
- Kabaq uses scanning to create food models for web, social and AR — [Hospitality Ireland](https://www.hospitalityireland.com/technology/video-check-future-restaurant-menus-48707)

### Inferences
- The professional AR-menu vendors rely on photogrammetry or scanning rather than generative AI, which suggests that fidelity to the real dish is what sells.
- A typical DIY creator workflow is phone capture (KIRI/Polycam) → Blender cleanup and decimation → GLB plus USDZ → `<model-viewer>` via QR code.

### Gaps
- No pages were found for Owow or ARitize pricing. Luma's current export options were not verified (Luma has shifted toward video generation, unconfirmed). Qlone was not researched.
- No specific Instagram or YouTube food creators documenting their toolchain were found in the searches.

## AI as a hybrid (photogrammetry + AI, AI for unscannable items)

### Takeaway
The defensible hybrid is to use photogrammetry for the geometry and texture of the dish, and AI only for supporting work: background removal and delighting, texture in-painting of holes, and generating hard-to-scan elements (glassware, clear drinks, steam) that are labelled as representative.

### Cited Findings
- One industry article recommends keeping AI out of the final image and gating any AI step with QA and human approval — [Zembula](https://www.zembula.com/blog/ai-generated-food-photos/)
- Hunyuan3D 2.1 ships a separate paint/texture stage (hy3dpaint) that can be run on an existing mesh — [HF Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/38e83be15b2054c29a8ae2e2ef69daf68b7dd923/hy3dpaint/utils/multiview_utils.py)

### Inferences
- Transparent or reflective items (glasses, drinks, glazes) defeat photogrammetry. For these, generating a model from a product photo, or modelling a simple glass by hand with a transmission material, is the practical route.
- Retexturing a scanned mesh with AI risks the same hallucination problems. It should be limited to filling holes.

### Gaps
- No documented end-to-end case study of a "photogrammetry mesh + AI texture" food pipeline was found.
