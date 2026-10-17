"""glTF (.glb) and USDZ writers shared by the model builders.

A mesh is a dict: V (n,3) positions, N (n,3) normals, F (m,3) triangles, optional
UV (n,2) and T (n,4) tangents, plus "name" and "material" (a key of `materials`).
A material is a dict: color (linear RGB), roughness, clearcoat, clearcoat_roughness,
and optional texture / normal / rough_tex file names (keys of `textures`, JPEG bytes).
Both files are metres and Y-up.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def write_glb(path: Path, meshes: list[dict], textures: dict[str, bytes], materials: dict, root: str,
              generator: str = "ar-menu-poc"):
    """Binary glTF 2.0: one root node named `root`, one child node per mesh."""
    mat_names = list(materials)
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

    gl_materials = []
    for name, p in materials.items():
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
        gl_materials.append(mat)

    binbuf.extend(b"\0" * (-len(binbuf) % 4))
    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "extensionsUsed": ["KHR_materials_clearcoat"],
        "scene": 0, "scenes": [{"name": "Scene", "nodes": [0]}],
        "nodes": [{"name": root, "children": list(range(1, len(nodes) + 1))}] + nodes,
        "meshes": gl_meshes, "materials": gl_materials, "accessors": accessors, "bufferViews": views,
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


def write_usdz(path: Path, meshes: list[dict], textures: dict[str, bytes], materials: dict, root: str):
    """ARKit USDZ package with UsdPreviewSurface materials; same layout as write_glb."""
    import tempfile

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    work = Path(tmp.name)
    (work / "textures").mkdir()
    for name, data in textures.items():
        (work / "textures" / name).write_bytes(data)
    usdc = work / f"{path.stem}.usdc"
    stage = Usd.Stage.CreateNew(str(usdc))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    xform = UsdGeom.Xform.Define(stage, f"/{root}")
    stage.SetDefaultPrim(xform.GetPrim())
    UsdGeom.Scope.Define(stage, f"/{root}/Materials")

    mats = {}
    for name, p in materials.items():
        mp = f"/{root}/Materials/{name}"
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
        mesh = UsdGeom.Mesh.Define(stage, f"/{root}/{m['name']}")
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
