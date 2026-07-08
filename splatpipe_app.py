"""
SplatPipe — standalone video ➜ 3D Gaussian Splat web app.

Runs as a serverless app on Modal (https://modal.com):
  - a lightweight web frontend (upload page + job list + built-in 3D viewer)
  - a GPU worker (T4) that runs ffmpeg → COLMAP → splat training → export
    in the background, then shuts down (you pay only for active minutes).

Usage:
  pip install modal
  modal setup                      # one-time login
  modal serve splatpipe_app.py     # dev mode (temporary URL, hot reload)
  modal deploy splatpipe_app.py    # permanent URL

NOTE: the deployed URL is public to anyone who knows it. Keep it private,
or add Modal proxy-auth if you want real protection.
"""

import time

import modal

app = modal.App("splatpipe")

# Persistent storage for uploaded videos and finished models
volume = modal.Volume.from_name("splatpipe-data", create_if_missing=True)
# Small shared dict for job status (survives across containers)
jobs = modal.Dict.from_name("splatpipe-jobs", create_if_missing=True)

DATA = "/data"

# ---------------------------------------------------------------------------
# GPU worker image: CUDA toolkit (gsplat JIT-compiles kernels), COLMAP,
# ffmpeg, nerfstudio.
# ---------------------------------------------------------------------------
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("colmap", "ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.1.2", "torchvision==0.16.2")
    # Modal's builder sets CC/CXX to clang, which this image doesn't have;
    # point them at gcc so nerfstudio's source deps (fpsample, pyliblzfse) build
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install("nerfstudio")
    # gsplat compiles its CUDA kernels on first use; make sure it targets the T4.
    # QT_QPA_PLATFORM keeps COLMAP's Qt happy on a headless container.
    .env({"TORCH_CUDA_ARCH_LIST": "7.5", "QT_QPA_PLATFORM": "offscreen"})
)

# CPU-only OpenDroneMap image for the metric "blueprint" layer (orthophoto,
# elevation model, textured mesh). ODM runs fine without a GPU; ffmpeg is
# added for frame extraction.
odm_image = (
    modal.Image.from_registry("opendronemap/odm:3.5.6", add_python="3.11")
    # the ODM image's ENTRYPOINT (python3 /code/run.py) would run at container
    # boot under Modal's injected python and crash-loop; clear it
    .entrypoint([])
    # gdal-bin: ODM's own gdal utilities live off-PATH in SuperBuild/install/bin
    .apt_install("ffmpeg", "gdal-bin")
)

# CPU-only image for turning a splat .ply into walkable meshes (open3d Poisson)
mesh_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libgomp1", "libx11-6", "libegl1")
    # trimesh does the GLB export: open3d's writer emits corrupt buffer views
    # for uint16-indexed (decimated) meshes
    .pip_install("open3d==0.19.0", "plyfile", "numpy", "trimesh")
)

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.*"
)


def _set(job_id: str, **fields):
    """Merge fields into the job's status record."""
    rec = jobs.get(job_id, {})
    rec.update(fields)
    rec["updated"] = time.time()
    jobs[job_id] = rec


# ---------------------------------------------------------------------------
# Background GPU pipeline
# ---------------------------------------------------------------------------
@app.function(
    image=gpu_image,
    gpu="T4",
    timeout=3 * 60 * 60,
    volumes={DATA: volume},
)
def process_video(job_id: str):
    import glob
    import shutil
    import subprocess
    from pathlib import Path

    volume.reload()  # see the video the web container just uploaded
    job_dir = Path(DATA) / "jobs" / job_id
    video = next(job_dir.glob("input.*"))

    def run(stage: str, cmd: list[str], cwd: Path):
        _set(job_id, status="running", stage=stage, log="")
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        last_push = 0.0
        tail: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail = (tail + [line])[-4:]
            if time.time() - last_push > 3:
                _set(job_id, log="\n".join(tail))
                last_push = time.time()
        proc.wait()
        _set(job_id, log="\n".join(tail))
        if proc.returncode != 0:
            raise RuntimeError(f"{stage} failed (exit {proc.returncode}): {tail[-1] if tail else ''}")

    try:
        _set(job_id, status="running", stage="starting", started=time.time())

        # 1. frames + camera poses (ffmpeg + COLMAP)
        run("extracting frames & camera poses", [
            "ns-process-data", "video",
            "--data", str(video),
            "--output-dir", str(job_dir / "proc"),
            "--num-frames-target", "150",
            # apt COLMAP is CPU-only; GPU SIFT would try to open an OpenGL
            # context on a headless box and abort
            "--no-gpu",
        ], job_dir)
        volume.commit()

        # 2. train the gaussian splat
        run("training gaussian splat", [
            "ns-train", "splatfacto",
            "--data", str(job_dir / "proc"),
            "--output-dir", str(job_dir / "outputs"),
            "--viewer.quit-on-train-completion", "True",
        ], job_dir)
        volume.commit()

        # 3. export to .ply
        cfgs = sorted(glob.glob(str(job_dir / "outputs" / "**" / "config.yml"), recursive=True))
        if not cfgs:
            raise RuntimeError("training produced no config.yml")
        run("exporting model", [
            "ns-export", "gaussian-splat",
            "--load-config", cfgs[-1],
            "--output-dir", str(job_dir / "export"),
        ], job_dir)

        plys = sorted((job_dir / "export").glob("*.ply"))
        if not plys:
            raise RuntimeError("export produced no .ply")
        shutil.copy(plys[-1], job_dir / "model.ply")

        # free space: keep only the final model + a few debug bits.
        # The input video is kept so the ODM map layer can reuse it.
        for sub in ("proc", "outputs", "export"):
            shutil.rmtree(job_dir / sub, ignore_errors=True)
        volume.commit()

        _set(job_id, status="done", stage="done",
             size_mb=round((job_dir / "model.ply").stat().st_size / 1e6, 1))
    except Exception as e:
        volume.commit()
        _set(job_id, status="error", error=str(e))
        raise


# Runs under the ODM image's system python (has GDAL+numpy): writes meta.json
# with the orthophoto geotransform and a downsampled DSM elevation grid so the
# frontend can turn a click on the blueprint into a 3D map-frame point.
ODM_META_SCRIPT = """
import json, sys
from osgeo import gdal
ortho = gdal.Open(sys.argv[1]); dsm = gdal.Open(sys.argv[2])
band = dsm.GetRasterBand(1)
w, h = dsm.RasterXSize, dsm.RasterYSize
gw = min(300, w); gh = max(1, round(h * gw / w))
arr = band.ReadAsArray(buf_xsize=gw, buf_ysize=gh)
nd = band.GetNoDataValue()
grid = []
for row in arr:
    out = []
    for v in row:
        fv = float(v)
        bad = fv != fv or (nd is not None and abs(fv - nd) < 1e-6)
        out.append(None if bad else round(fv, 2))
    grid.append(out)
meta = {
  "ortho": {"gt": list(ortho.GetGeoTransform()), "size": [ortho.RasterXSize, ortho.RasterYSize]},
  "dsm": {"gt": list(dsm.GetGeoTransform()), "size": [w, h], "grid_size": [gw, gh], "grid": grid},
}
json.dump(meta, open(sys.argv[3], "w"))
print("meta.json written", gw, "x", gh)
"""


# ---------------------------------------------------------------------------
# Background ODM pipeline (CPU): orthophoto + DSM + textured mesh from the
# same video. This is the metric/blueprint layer of the digital twin.
# ---------------------------------------------------------------------------
@app.function(
    image=odm_image,
    cpu=8,
    memory=16384,
    timeout=4 * 60 * 60,
    volumes={DATA: volume},
)
def process_odm(job_id: str):
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path

    # ODM's helper scripts call bare `python3`; Modal's injected python
    # shadows the image's system python (which has all ODM/OpenSfM native
    # deps), so put /usr/bin first for the whole ODM process tree.
    odm_env = dict(os.environ, PATH="/usr/bin:" + os.environ.get("PATH", ""))

    volume.reload()
    job_dir = Path(DATA) / "jobs" / job_id
    videos = sorted(job_dir.glob("input.*"))
    if not videos:
        _set(job_id, odm_status="error", odm_error="input video no longer stored — re-upload it")
        return
    video = videos[0]

    def run(stage: str, cmd: list[str]):
        _set(job_id, odm_status="running", odm_stage=stage, odm_log="")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=odm_env,
        )
        last_push = 0.0
        tail: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)  # full log lands in `modal app logs`
                tail = (tail + [line])[-4:]
            if time.time() - last_push > 3:
                _set(job_id, odm_log="\n".join(tail))
                last_push = time.time()
        proc.wait()
        _set(job_id, odm_log="\n".join(tail))
        if proc.returncode != 0:
            raise RuntimeError(f"{stage} failed (exit {proc.returncode}): {tail[-1] if tail else ''}")

    work = Path("/tmp/odm")  # scratch on container-local disk, not the volume
    proj = work / "proj"
    images = proj / "images"
    try:
        _set(job_id, odm_status="running", odm_stage="starting", odm_started=time.time())
        images.mkdir(parents=True, exist_ok=True)

        # ~120 frames spread evenly across the video
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(video)], capture_output=True, text=True)
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        fps = min(120 / max(duration, 1.0), 4.0)
        run("extracting frames", [
            "ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps:.4f}",
            "-q:v", "2", str(images / "frame_%04d.jpg"),
        ])

        # ODM's deps live in the image's system python; Modal's injected
        # python3 shadows it on PATH, so the interpreter must be explicit
        run("photogrammetry (ODM)", [
            "/usr/bin/python3", "/code/run.py", "--project-path", str(work), "proj",
            "--dsm", "--orthophoto-png", "--skip-report",
            "--max-concurrency", "8",
            # video frames carry no EXIF/GPS; make reconstruction tolerant:
            "--min-num-features", "12000",     # hazy aerial imagery
            "--matcher-type", "bruteforce",    # exhaustive matching, ~120 images is small
            "--ignore-gsd",                    # GSD heuristics assume GPS/altitude
        ])

        out = job_dir / "odm"
        out.mkdir(exist_ok=True)
        ortho_dir = proj / "odm_orthophoto"
        print("odm_orthophoto dir:", sorted(p.name for p in ortho_dir.iterdir()) if ortho_dir.exists() else "MISSING", flush=True)
        ortho_png = ortho_dir / "odm_orthophoto.png"
        ortho_tif = ortho_dir / "odm_orthophoto.tif"
        dsm = proj / "odm_dem" / "dsm.tif"
        if ortho_tif.exists():
            shutil.copy(ortho_tif, out / "orthophoto.tif")
        if ortho_png.exists():
            shutil.copy(ortho_png, out / "orthophoto.png")
        elif ortho_tif.exists():
            # don't depend on ODM's --orthophoto-png; render the PNG ourselves
            subprocess.run(["gdal_translate", "-of", "PNG", "-ot", "Byte", "-scale",
                            str(ortho_tif), str(out / "orthophoto.png")], check=False, env=odm_env)
        if dsm.exists():
            shutil.copy(dsm, out / "dsm.tif")
            hill = work / "hill.tif"
            subprocess.run(["gdaldem", "hillshade", str(dsm), str(hill), "-z", "1.5"], check=False)
            if hill.exists():
                subprocess.run(["gdal_translate", "-of", "PNG", str(hill),
                                str(out / "hillshade.png")], check=False)
        tex = proj / "odm_texturing"
        if tex.exists():
            shutil.make_archive(str(out / "mesh"), "zip", tex)
        if (out / "orthophoto.tif").exists() and (out / "dsm.tif").exists():
            subprocess.run(["/usr/bin/python3", "-c", ODM_META_SCRIPT,
                            str(out / "orthophoto.tif"), str(out / "dsm.tif"),
                            str(out / "meta.json")], check=False, env=odm_env)
        if not (out / "orthophoto.png").exists():
            raise RuntimeError("ODM finished but produced no orthophoto")
        volume.commit()
        _set(job_id, odm_status="done", odm_stage="done",
             odm_files=sorted(p.name for p in out.iterdir()))
    except Exception as e:
        volume.commit()
        _set(job_id, odm_status="error", odm_error=str(e))
        raise


# ---------------------------------------------------------------------------
# Walk-mesh builder (CPU): gaussian centers -> Poisson surface -> mesh.glb
# (vertex colors, for Blender/Unity) + collision.glb (decimated, for physics
# and the in-browser Walk mode). Tuning constants:
# ---------------------------------------------------------------------------
MESH_OPACITY_MIN = 0.3        # drop near-transparent gaussians (floaters/haze)
MESH_SCALE_PCTL = 98          # drop gaussians bigger than this scale percentile
MESH_POISSON_DEPTH = 10       # Poisson octree depth; 9 = coarser/faster
MESH_DENSITY_QUANTILE = 0.06  # trim the sparsest Poisson vertices (halo)
COLLISION_TARGET_TRIS = 50_000  # decimation target for the physics mesh


@app.function(image=mesh_image, cpu=8, memory=8192, timeout=1800, volumes={DATA: volume})
def build_walk_mesh(job_id: str):
    import numpy as np
    import open3d as o3d
    from pathlib import Path
    from plyfile import PlyData

    volume.reload()
    job_dir = Path(DATA) / "jobs" / job_id
    try:
        _set(job_id, mesh_status="running", mesh_stage="loading splat")
        v = PlyData.read(str(job_dir / "model.ply"))["vertex"]
        xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
        opa = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
        scales = np.exp(np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]])).max(axis=1)
        SH0 = 0.28209479177387814
        rgb = np.clip(0.5 + SH0 * np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]), 0, 1)
        keep = (opa > MESH_OPACITY_MIN) & (scales < np.percentile(scales, MESH_SCALE_PCTL))
        print(f"splats: {len(xyz)}, kept after floater filter: {int(keep.sum())}", flush=True)
        xyz, rgb = xyz[keep], rgb[keep]

        _set(job_id, mesh_stage=f"poisson meshing {int(keep.sum()):,} points")
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz)
        pc.colors = o3d.utility.Vector3dVector(rgb)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        pc.orient_normals_consistent_tangent_plane(15)
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pc, depth=MESH_POISSON_DEPTH)
        dens = np.asarray(dens)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, MESH_DENSITY_QUANTILE))
        clusters, counts, _ = mesh.cluster_connected_triangles()
        mesh.remove_triangles_by_mask(np.asarray(clusters) != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        full_tris = len(mesh.triangles)
        print(f"poisson mesh: {full_tris} triangles", flush=True)

        _set(job_id, mesh_stage="exporting mesh.glb + collision.glb")
        import trimesh

        def export_glb(m, path, colors):
            kw = {}
            if colors and len(m.vertex_colors):
                kw["vertex_colors"] = (np.asarray(m.vertex_colors) * 255).astype(np.uint8)
            trimesh.Trimesh(vertices=np.asarray(m.vertices),
                            faces=np.asarray(m.triangles), process=False, **kw).export(path)

        export_glb(mesh, str(job_dir / "mesh.glb"), colors=True)
        col = mesh.simplify_quadric_decimation(COLLISION_TARGET_TRIS) \
            if full_tris > COLLISION_TARGET_TRIS else mesh
        col.remove_unreferenced_vertices()
        export_glb(col, str(job_dir / "collision.glb"), colors=False)  # physics mesh: no colors
        volume.commit()
        _set(job_id, mesh_status="done", mesh_stage="done",
             mesh_tris=full_tris, collision_tris=len(col.triangles),
             mesh_mb=round((job_dir / "mesh.glb").stat().st_size / 1e6, 1))
    except Exception as e:
        volume.commit()
        _set(job_id, mesh_status="error", mesh_error=str(e))
        raise


# Backfill map metadata for ODM runs made before meta.json existed
@app.function(image=odm_image, cpu=2, memory=4096, timeout=600, volumes={DATA: volume})
def odm_meta(job_id: str):
    import os
    import subprocess
    from pathlib import Path

    env = dict(os.environ, PATH="/usr/bin:" + os.environ.get("PATH", ""))
    volume.reload()
    out = Path(DATA) / "jobs" / job_id / "odm"
    r = subprocess.run(["/usr/bin/python3", "-c", ODM_META_SCRIPT,
                        str(out / "orthophoto.tif"), str(out / "dsm.tif"),
                        str(out / "meta.json")], capture_output=True, text=True, env=env)
    print(r.stdout, r.stderr, flush=True)
    if (out / "meta.json").exists():
        volume.commit()
        _set(job_id, odm_files=sorted(p.name for p in out.iterdir()))


# ---------------------------------------------------------------------------
# Web frontend
# ---------------------------------------------------------------------------
@app.function(image=web_image, volumes={DATA: volume})
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import uuid
    from pathlib import Path

    from fastapi import FastAPI, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    api = FastAPI()

    @api.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @api.post("/api/jobs")
    async def create_job(video: UploadFile):
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        ext = (video.filename or "video.mp4").rsplit(".", 1)[-1].lower()
        job_dir = Path(DATA) / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        dest = job_dir / f"input.{ext}"
        with dest.open("wb") as f:
            while chunk := await video.read(8 * 1024 * 1024):
                f.write(chunk)
        volume.commit()
        jobs[job_id] = {"status": "queued", "stage": "waiting for GPU",
                        "name": video.filename, "created": time.time()}
        process_video.spawn(job_id)
        return {"job_id": job_id}

    @api.get("/api/version")
    def version():
        return {"marker": "v-walk-1", "autoframe_in_page": "autoFrame" in PAGE}

    @api.get("/api/jobs")
    def list_jobs():
        items = [{"job_id": k, **v} for k, v in jobs.items()]
        items.sort(key=lambda r: r.get("created", 0), reverse=True)
        return items[:50]

    @api.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        rec = jobs.get(job_id)
        if rec is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return {"job_id": job_id, **rec}

    @api.get("/api/jobs/{job_id}/model.ply")
    def job_model(job_id: str):
        volume.reload()
        path = Path(DATA) / "jobs" / job_id / "model.ply"
        if not path.exists():
            return JSONResponse({"error": "model not ready"}, status_code=404)
        return FileResponse(path, media_type="application/octet-stream",
                            filename=f"{job_id}.ply")

    @api.post("/api/jobs/{job_id}/odm")
    async def start_odm(job_id: str, video: UploadFile | None = None):
        rec = jobs.get(job_id)
        if rec is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        if rec.get("odm_status") in ("queued", "running"):
            return {"job_id": job_id, "odm_status": rec["odm_status"]}
        job_dir = Path(DATA) / "jobs" / job_id
        volume.reload()
        if not any(job_dir.glob("input.*")):
            if video is None:
                # older jobs deleted their video after training; ask for it back
                return JSONResponse({"error": "video no longer stored — re-upload it"},
                                    status_code=409)
            ext = (video.filename or "video.mp4").rsplit(".", 1)[-1].lower()
            job_dir.mkdir(parents=True, exist_ok=True)
            dest = job_dir / f"input.{ext}"
            with dest.open("wb") as f:
                while chunk := await video.read(8 * 1024 * 1024):
                    f.write(chunk)
            volume.commit()
        _set(job_id, odm_status="queued", odm_stage="waiting for worker",
             odm_error="", odm_log="")
        process_odm.spawn(job_id)
        return {"job_id": job_id, "odm_status": "queued"}

    @api.get("/api/jobs/{job_id}/odm/{fname}")
    def odm_file(job_id: str, fname: str):
        allowed = {"orthophoto.png": "image/png", "orthophoto.tif": "image/tiff",
                   "hillshade.png": "image/png", "meta.json": "application/json",
                   "dsm.tif": "image/tiff", "mesh.zip": "application/zip"}
        if fname not in allowed:
            return JSONResponse({"error": "unknown file"}, status_code=404)
        volume.reload()
        path = Path(DATA) / "jobs" / job_id / "odm" / fname
        if not path.exists():
            return JSONResponse({"error": "not ready"}, status_code=404)
        return FileResponse(path, media_type=allowed[fname], filename=f"{job_id}-{fname}")

    @api.post("/api/jobs/{job_id}/mesh")
    def start_mesh(job_id: str):
        rec = jobs.get(job_id)
        if rec is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        if rec.get("mesh_status") in ("queued", "running"):
            return {"job_id": job_id, "mesh_status": rec["mesh_status"]}
        volume.reload()
        if not (Path(DATA) / "jobs" / job_id / "model.ply").exists():
            return JSONResponse({"error": "model not ready"}, status_code=409)
        _set(job_id, mesh_status="queued", mesh_stage="waiting for worker", mesh_error="")
        build_walk_mesh.spawn(job_id)
        return {"job_id": job_id, "mesh_status": "queued"}

    @api.get("/api/jobs/{job_id}/mesh/{fname}")
    def mesh_file(job_id: str, fname: str):
        allowed = {"mesh.glb": "model/gltf-binary", "collision.glb": "model/gltf-binary"}
        if fname not in allowed:
            return JSONResponse({"error": "unknown file"}, status_code=404)
        volume.reload()
        path = Path(DATA) / "jobs" / job_id / fname
        if not path.exists():
            return JSONResponse({"error": "not ready"}, status_code=404)
        return FileResponse(path, media_type=allowed[fname], filename=f"{job_id}-{fname}")

    @api.post("/api/jobs/{job_id}/odm/meta")
    def gen_odm_meta(job_id: str):
        rec = jobs.get(job_id)
        if rec is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        if rec.get("odm_status") != "done":
            return JSONResponse({"error": "map layer not built yet"}, status_code=409)
        odm_meta.spawn(job_id)
        return {"ok": True}

    @api.post("/api/jobs/{job_id}/registration")
    async def save_registration(job_id: str, payload: dict):
        import math
        if jobs.get(job_id) is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        try:
            scale = float(payload["scale"])
            q = [float(x) for x in payload["q"]]
            t = [float(x) for x in payload["t"]]
            rmse = float(payload.get("rmse", 0.0))
            n = int(payload.get("n", 0))
            ok = (scale > 0 and len(q) == 4 and len(t) == 3
                  and all(map(math.isfinite, q + t + [scale, rmse])))
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            return JSONResponse({"error": "invalid registration"}, status_code=400)
        _set(job_id, registration={"scale": scale, "q": q, "t": t,
                                   "rmse": rmse, "n": n, "created": time.time()})
        return {"ok": True}

    return api


# ---------------------------------------------------------------------------
# Frontend page (upload + job list + built-in splat viewer)
# ---------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SplatPipe</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{--ink:#0b0e14;--ink-2:#11151f;--line:#232a3a;--text:#e8ebf2;--dim:#8b93a7;
      --iris:#9d8cff;--iris-soft:rgba(157,140,255,.16);--ok:#6fd08c;--err:#e8909f}
*{margin:0;padding:0;box-sizing:border-box}
body{background:radial-gradient(1200px 700px at 70% -10%,rgba(157,140,255,.06),transparent 60%),var(--ink);
     color:var(--text);font-family:'Space Grotesk',system-ui,sans-serif;min-height:100vh}
.wrap{max-width:880px;margin:0 auto;padding:40px 20px 80px}
header{display:flex;align-items:baseline;gap:14px;margin-bottom:34px}
h1{font-size:24px;font-weight:700;letter-spacing:-.01em}
header .tag{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--dim)}
.card{border:1px solid var(--line);border-radius:16px;background:var(--ink-2);padding:28px;margin-bottom:22px}
#drop{text-align:center;padding:40px 28px;cursor:pointer;transition:border-color .15s,background .15s}
#drop.dragging{border-color:var(--iris);background:#131828}
#drop:focus-visible{outline:2px solid var(--iris);outline-offset:3px}
#drop .big{font-size:18px;font-weight:500;margin-top:16px}
#drop .hint{color:var(--dim);font-size:14px;margin-top:6px}
.gauss{width:120px;height:76px}
button.primary{font:500 15px 'Space Grotesk',sans-serif;color:var(--ink);background:var(--iris);
  border:none;border-radius:10px;padding:11px 24px;cursor:pointer;margin-top:20px}
button.primary:hover{filter:brightness(1.08)}
button.primary:disabled{opacity:.45;cursor:default}
h2{font-size:15px;font-weight:500;color:var(--dim);text-transform:uppercase;
   letter-spacing:.08em;margin:0 0 14px}
.job{display:flex;align-items:center;gap:14px;padding:14px 0;border-top:1px solid var(--line);flex-wrap:wrap}
.job:first-of-type{border-top:none}
.job .dot{width:9px;height:9px;border-radius:50%;flex:none}
.dot.queued,.dot.running{background:var(--iris);animation:pulse 1.4s ease-in-out infinite}
.dot.done{background:var(--ok)} .dot.error{background:var(--err)}
@keyframes pulse{50%{opacity:.35}}
@media (prefers-reduced-motion: reduce){.dot{animation:none}}
.job .meta{flex:1;min-width:220px}
.job .name{font-size:15px}
.job .stage{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--dim);margin-top:3px;
  white-space:pre-wrap;word-break:break-word}
.job .stage.error{color:var(--err)}
.job a,.job button.view,.job button.act{font:500 13px 'Space Grotesk',sans-serif;color:var(--text);background:transparent;
  border:1px solid var(--line);border-radius:8px;padding:7px 14px;cursor:pointer;text-decoration:none}
.job a:hover,.job button.view:hover,.job button.act:hover{border-color:var(--iris);color:var(--iris)}
/* blueprint overlay */
#bpwrap{display:none;position:fixed;inset:0;z-index:30;background:var(--ink);overflow:hidden;text-align:center}
#bpcanvas{position:relative;display:inline-block;margin-top:64px;transform-origin:0 0}
#bpwrap img{display:block;max-width:calc(100vw - 28px);height:auto}
.regdot{position:absolute;width:18px;height:18px;border-radius:50%;transform:translate(-50%,-50%);
  font:600 11px 'IBM Plex Mono',monospace;color:#0b0e14;display:flex;align-items:center;
  justify-content:center;pointer-events:none;border:1px solid rgba(0,0,0,.45)}
#bpbar{position:fixed;top:14px;left:14px;z-index:2;display:flex;gap:8px;align-items:center;
  background:rgba(13,17,26,.82);border:1px solid var(--line);border-radius:12px;padding:9px 12px;
  backdrop-filter:blur(6px);font-family:'IBM Plex Mono',monospace;font-size:12px;max-width:calc(100vw - 28px);flex-wrap:wrap}
#bpbar .vname{color:var(--iris);word-break:break-all}
#bpbar button,#bpbar a{font:500 12.5px 'Space Grotesk',sans-serif;color:var(--text);background:transparent;
  border:1px solid var(--line);border-radius:8px;padding:5px 11px;cursor:pointer;text-decoration:none}
#bpbar button:hover,#bpbar a:hover{border-color:var(--iris);color:var(--iris)}
#empty{color:var(--dim);font-size:14px;padding:6px 0}
/* viewer modal */
#viewerwrap{display:none;position:fixed;inset:0;z-index:20;background:var(--ink)}
#viewer{position:absolute;inset:0}
#viewerbar{position:absolute;top:14px;left:14px;z-index:2;display:flex;gap:8px;align-items:center;
  background:rgba(13,17,26,.82);border:1px solid var(--line);border-radius:12px;padding:9px 12px;
  backdrop-filter:blur(6px);font-family:'IBM Plex Mono',monospace;font-size:12px;max-width:calc(100vw - 28px);flex-wrap:wrap}
#viewerbar .vname{color:var(--iris);word-break:break-all}
#viewerbar button{font:500 12.5px 'Space Grotesk',sans-serif;color:var(--text);background:transparent;
  border:1px solid var(--line);border-radius:8px;padding:5px 11px;cursor:pointer}
#viewerbar button:hover{border-color:var(--iris);color:var(--iris)}
#viewerbar button.on{border-color:var(--iris);color:var(--iris)}
#viewerbar input{font:500 12.5px 'IBM Plex Mono',monospace;color:var(--text);background:var(--ink);
  border:1px solid var(--line);border-radius:8px;padding:5px 8px;width:110px}
#vread{color:var(--ok)}
#vprogress{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);z-index:1;
  font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--dim)}
</style>
</head>
<body>
<div class="wrap">
  <header><h1>SplatPipe</h1><span class="tag">video &#10142; 3D gaussian splat</span></header>

  <div class="card" id="drop" tabindex="0" role="button" aria-label="Upload a video">
    <svg class="gauss" viewBox="0 0 150 96" aria-hidden="true">
      <defs><radialGradient id="g" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#9d8cff" stop-opacity=".9"/>
        <stop offset="100%" stop-color="#9d8cff" stop-opacity="0"/></radialGradient></defs>
      <circle cx="75" cy="46" r="17" fill="url(#g)"/><circle cx="49" cy="34" r="11" fill="url(#g)" opacity=".85"/>
      <circle cx="103" cy="36" r="12" fill="url(#g)" opacity=".8"/><circle cx="60" cy="64" r="12" fill="url(#g)" opacity=".75"/>
      <circle cx="94" cy="66" r="10" fill="url(#g)" opacity=".7"/><circle cx="30" cy="52" r="8" fill="url(#g)" opacity=".55"/>
      <circle cx="121" cy="54" r="8" fill="url(#g)" opacity=".55"/><circle cx="75" cy="18" r="7" fill="url(#g)" opacity=".5"/>
    </svg>
    <div class="big">Drop a video to build a 3D model</div>
    <div class="hint">30&ndash;90&thinsp;s orbit or walkthrough &middot; mp4/mov &middot; takes ~30&ndash;50 min on a cloud GPU</div>
    <button class="primary" id="pick">Choose video</button>
    <input type="file" id="file" accept="video/*" hidden/>
    <div class="hint" id="upstate" style="margin-top:14px"></div>
  </div>

  <div class="card">
    <h2>Jobs</h2>
    <div id="jobs"><div id="empty">Nothing yet. Your finished models appear here &mdash; they stay saved between visits.</div></div>
  </div>
</div>

<div id="bpwrap">
  <div id="bpcanvas"><img id="bpimg" alt="orthophoto"/></div>
  <div id="bpbar">
    <span class="vname" id="bpname"></span>
    <button id="bpmode" title="Toggle orthophoto / elevation relief">Relief</button>
    <button id="bpreg" title="Mark the map spots matching your numbered 3D points (in order)">Register</button>
    <button id="bpalign" title="Compute the splat-to-map alignment from the point pairs">Align</button>
    <button id="bpclearpts" title="Clear all registration points">Clear pts</button>
    <button id="bpfit" title="Reset zoom and pan">Reset view</button>
    <span id="bpread"></span>
    <a id="bpdsm" href="#" title="Elevation model GeoTIFF">DSM .tif</a>
    <button id="bpclose">Close</button>
  </div>
</div>

<div id="viewerwrap">
  <div id="viewer"></div>
  <div id="vprogress">loading&hellip;</div>
  <div id="viewerbar">
    <span class="vname" id="vname"></span>
    <button id="vflip" title="Model upside down? Flip the camera's up axis">Flip up-axis</button>
    <button id="vfly" title="Free-fly camera: WASD + mouse look, E/Q up/down, Shift fast, scroll = speed, Esc exits">Fly</button>
    <button id="vwalk" title="Walk on the model: WASD + mouse look, Shift run, Space jump, scroll = eye height, Esc exits. Builds a walk mesh on first use (~2 min)">Walk</button>
    <button id="vmeasure" title="Click two points on the model to measure the distance">Measure</button>
    <button id="vreg" title="Registration: click 3-5 distinctive spots, then mark the same spots on the Blueprint">Register</button>
    <span id="vread"></span>
    <span id="vcal" style="display:none">
      <input id="vcalin" type="number" min="0" step="any" placeholder="real length (m)"/>
      <button id="vcalok" title="Enter the real-world length of the last measurement to calibrate this model to meters">Set scale</button>
    </span>
    <button id="vclose">Close</button>
  </div>
</div>

<script type="importmap">
{"imports":{
  "three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/",
  "three-mesh-bvh":"https://cdn.jsdelivr.net/npm/three-mesh-bvh@0.7.8/build/index.module.js",
  "@mkkellogg/gaussian-splats-3d":"https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"
}}
</script>
<script type="module">
import * as THREE from 'three';
import * as GS from '@mkkellogg/gaussian-splats-3d';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

const $ = (id) => document.getElementById(id);
const drop = $('drop'), file = $('file'), upstate = $('upstate');

// ---------- upload ----------
async function upload(f){
  if(!f) return;
  upstate.textContent = `Uploading ${f.name}\u2026`;
  $('pick').disabled = true;
  const fd = new FormData(); fd.append('video', f);
  try{
    const r = await fetch('/api/jobs', {method:'POST', body:fd});
    if(!r.ok) throw new Error(await r.text());
    upstate.textContent = 'Uploaded. Pipeline started in the background \u2014 you can close this tab and come back.';
    refresh();
  }catch(e){ upstate.textContent = 'Upload failed: ' + e.message; }
  $('pick').disabled = false; file.value = '';
}
$('pick').addEventListener('click', e => { e.stopPropagation(); file.click(); });
drop.addEventListener('click', () => file.click());
drop.addEventListener('keydown', e => { if(e.key==='Enter'||e.key===' '){e.preventDefault();file.click();} });
file.addEventListener('change', () => upload(file.files[0]));
['dragover','dragenter'].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault();drop.classList.add('dragging');}));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault();drop.classList.remove('dragging');}));
drop.addEventListener('drop', e => upload(e.dataTransfer.files[0]));

// ---------- job list ----------
const recs = {};   // latest job records by id (refreshed every poll)
function esc(s){ const d=document.createElement('div'); d.textContent=s||''; return d.innerHTML; }
function fmtElapsed(rec){
  if(!rec.started) return '';
  const end = (rec.status==='done'||rec.status==='error') ? rec.updated : Date.now()/1000;
  const m = Math.max(0, Math.round((end - rec.started)/60));
  return ` \u00b7 ${m} min`;
}
async function refresh(){
  try{
    const list = await (await fetch('/api/jobs')).json();
    list.forEach(r => { recs[r.job_id] = r; });
    const box = $('jobs');
    if(!list.length){ box.innerHTML = '<div id="empty">Nothing yet. Your finished models appear here \u2014 they stay saved between visits.</div>'; return; }
    box.innerHTML = list.map(r => {
      const st = r.status || 'queued';
      let stage = st==='error' ? (r.error||'failed') : (r.stage||st) + fmtElapsed(r);
      const log = st==='running' && r.log ? `\n${r.log}` : '';
      const os = r.odm_status;
      if(os==='queued'||os==='running') stage += `\nmap layer: ${r.odm_stage||os}${r.odm_log?'\n'+r.odm_log:''}`;
      else if(os==='error') stage += `\nmap layer failed: ${r.odm_error||''}`;
      const ms = r.mesh_status;
      if(ms==='queued'||ms==='running') stage += `\nwalk mesh: ${r.mesh_stage||ms}`;
      else if(ms==='error') stage += `\nwalk mesh failed: ${r.mesh_error||''}`;
      let act = '';
      if(st==='done'){
        act = `<button class="view" data-id="${r.job_id}" data-name="${esc(r.name)}">View in 3D</button>
           <a href="/api/jobs/${r.job_id}/model.ply">Download .ply${r.size_mb?` (${r.size_mb} MB)`:''}</a>`;
        if(os==='done')
          act += `<button class="act bp" data-id="${r.job_id}" data-name="${esc(r.name)}">Blueprint</button>
           <a href="/api/jobs/${r.job_id}/odm/mesh.zip">Mesh</a>`;
        else if(os!=='queued' && os!=='running')
          act += `<button class="act odmgo" data-id="${r.job_id}" title="Build orthophoto + elevation + mesh on a CPU worker (~1–2 h)">${os==='error'?'Retry map':'Map layer'}</button>`;
      }
      return `<div class="job"><span class="dot ${st}"></span>
        <div class="meta"><div class="name">${esc(r.name)||r.job_id}</div>
        <div class="stage ${st==='error'?'error':''}">${esc(stage)}${esc(log)}</div></div>${act}</div>`;
    }).join('');
    box.querySelectorAll('button.view').forEach(b =>
      b.addEventListener('click', () => openViewer(b.dataset.id, b.dataset.name)));
    box.querySelectorAll('button.odmgo').forEach(b =>
      b.addEventListener('click', () => startOdm(b.dataset.id)));
    box.querySelectorAll('button.bp').forEach(b =>
      b.addEventListener('click', () => openBlueprint(b.dataset.id, b.dataset.name)));
  }catch(e){ /* transient network errors: just retry next tick */ }
}
refresh(); setInterval(refresh, 4000);

// ---------- ODM map layer ----------
async function startOdm(id, file){
  const opts = {method:'POST'};
  if(file){ const fd = new FormData(); fd.append('video', file); opts.body = fd; }
  try{
    const r = await fetch(`/api/jobs/${id}/odm`, opts);
    if(r.status === 409){
      // this job's video was cleaned up after training: ask for it again
      const inp = document.createElement('input');
      inp.type = 'file'; inp.accept = 'video/*';
      inp.addEventListener('change', () => { if(inp.files[0]) startOdm(id, inp.files[0]); });
      inp.click();
      return;
    }
  }catch(e){ /* transient; job list will show state */ }
  refresh();
}

let bpMode = 'orthophoto', bpMeta = null, bpMetaJob = null, regOnBp = false, regBpJob = null;

function bpRead(t){ $('bpread').textContent = t; }
function bpDotsVisible(v){ document.querySelectorAll('#bpcanvas .regdot').forEach(d => d.style.display = v ? 'flex' : 'none'); }
function bpAddDot(u, v, i){
  const d = document.createElement('div');
  d.className = 'regdot'; d.textContent = i + 1;
  d.style.background = REGC[i % REGC.length];
  d.style.left = (u * 100) + '%'; d.style.top = (v * 100) + '%';
  $('bpcanvas').appendChild(d);
}

async function fetchBpMeta(id){
  bpMeta = null; bpMetaJob = id;
  for(let i = 0; i < 20; i++){
    if(bpMetaJob !== id) return;           // closed or switched job
    const r = await fetch(`/api/jobs/${id}/odm/meta.json`);
    if(r.ok){ bpMeta = await r.json(); bpRead(''); return; }
    if(i === 0){
      bpRead('preparing map metadata…');
      await fetch(`/api/jobs/${id}/odm/meta`, {method:'POST'});
    }
    await new Promise(res => setTimeout(res, 3000));
  }
  bpRead('map metadata unavailable');
}

function openBlueprint(id, name){
  if(regJob && regJob !== id){ clearRegState(); }
  document.querySelectorAll('#bpcanvas .regdot').forEach(d => d.remove());
  if(regBpJob !== id){ regOdm = []; regOdmUV = []; regBpJob = id; }
  else regOdmUV.forEach((uv, i) => bpAddDot(uv[0], uv[1], i));
  bpMode = 'orthophoto';
  bpResetView();
  $('bpwrap').dataset.id = id;
  $('bpname').textContent = (name || id) + ' — orthophoto';
  $('bpimg').src = `/api/jobs/${id}/odm/orthophoto.png`;
  $('bpdsm').href = `/api/jobs/${id}/odm/dsm.tif`;
  $('bpmode').textContent = 'Relief';
  $('bpwrap').style.display = 'block';
  fetchBpMeta(id);
}
$('bpmode').addEventListener('click', () => {
  const id = $('bpwrap').dataset.id;
  bpMode = bpMode === 'orthophoto' ? 'hillshade' : 'orthophoto';
  $('bpimg').src = `/api/jobs/${id}/odm/${bpMode}.png`;
  $('bpname').textContent = $('bpname').textContent.replace(/— .*$/, '— ' + (bpMode === 'orthophoto' ? 'orthophoto' : 'elevation relief'));
  $('bpmode').textContent = bpMode === 'orthophoto' ? 'Relief' : 'Ortho';
  bpDotsVisible(bpMode === 'orthophoto');   // dots are ortho-frame only
});
$('bpclose').addEventListener('click', () => { $('bpwrap').style.display = 'none'; $('bpimg').removeAttribute('src'); bpMetaJob = null; });

// -- blueprint zoom (wheel, toward cursor) + drag pan
let bpZoom = 1, bpTx = 0, bpTy = 0, bpDrag = null, bpDragDist = 0;
function bpApply(){ $('bpcanvas').style.transform = `translate(${bpTx}px,${bpTy}px) scale(${bpZoom})`; }
function bpResetView(){ bpZoom = 1; bpTx = 0; bpTy = 0; bpApply(); }
$('bpfit').addEventListener('click', bpResetView);
$('bpwrap').addEventListener('wheel', e => {
  e.preventDefault();
  const rect = $('bpcanvas').getBoundingClientRect();
  const u = (e.clientX - rect.left) / rect.width, v = (e.clientY - rect.top) / rect.height;
  const baseL = rect.left - bpTx, baseT = rect.top - bpTy;
  const baseW = rect.width / bpZoom, baseH = rect.height / bpZoom;
  bpZoom = Math.min(14, Math.max(1, bpZoom * (e.deltaY < 0 ? 1.25 : 0.8)));
  if(bpZoom === 1){ bpTx = 0; bpTy = 0; }
  else{
    bpTx = e.clientX - u * baseW * bpZoom - baseL;
    bpTy = e.clientY - v * baseH * bpZoom - baseT;
  }
  bpApply();
}, {passive: false});
$('bpwrap').addEventListener('mousedown', e => { bpDrag = [e.clientX, e.clientY, bpTx, bpTy]; bpDragDist = 0; });
document.addEventListener('mousemove', e => {
  if(!bpDrag) return;
  bpDragDist = Math.max(bpDragDist, Math.hypot(e.clientX - bpDrag[0], e.clientY - bpDrag[1]));
  if(bpDragDist > 5){
    bpTx = bpDrag[2] + (e.clientX - bpDrag[0]);
    bpTy = bpDrag[3] + (e.clientY - bpDrag[1]);
    bpApply();
  }
});
document.addEventListener('mouseup', () => { bpDrag = null; });

// -- registration: map-side point picking
function setBpReg(on){
  regOnBp = on;
  $('bpreg').classList.toggle('on', on);
  if(on){
    if(bpMode !== 'orthophoto') $('bpmode').click();
    bpRead(`mark your numbered 3D points on the map, in order (3D: ${regSplat.length} / map: ${regOdm.length})`);
  } else bpRead('');
}
$('bpreg').addEventListener('click', () => setBpReg(!regOnBp));

function dsmElev(X, Y){
  const d = bpMeta.dsm, gw = d.grid_size[0], gh = d.grid_size[1], gt = d.gt;
  const u = ((X - gt[0]) / gt[1]) * gw / d.size[0], v = ((Y - gt[3]) / gt[5]) * gh / d.size[1];
  const iu = Math.floor(u), iv = Math.floor(v), cand = [];
  for(let dv = 0; dv <= 1; dv++) for(let du = 0; du <= 1; du++){
    const gx = iu + du, gy = iv + dv;
    if(gx >= 0 && gx < gw && gy >= 0 && gy < gh && d.grid[gy][gx] != null)
      cand.push({z: d.grid[gy][gx], w: Math.max((1 - Math.abs(u - gx)) * (1 - Math.abs(v - gy)), 1e-6)});
  }
  if(!cand.length) return null;
  const tw = cand.reduce((a, c) => a + c.w, 0);
  return cand.reduce((a, c) => a + c.z * c.w, 0) / tw;
}

$('bpimg').addEventListener('click', e => {
  if(bpDragDist > 5) return;                       // that was a pan, not a pick
  if(!regOnBp || bpMode !== 'orthophoto') return;
  if(!bpMeta){ bpRead('map metadata still loading…'); return; }
  if(regOdm.length >= 5){ bpRead('5 points max — Align or Clear pts'); return; }
  const rect = $('bpimg').getBoundingClientRect();
  const u = (e.clientX - rect.left) / rect.width, v = (e.clientY - rect.top) / rect.height;
  const gt = bpMeta.ortho.gt, ow = bpMeta.ortho.size[0], oh = bpMeta.ortho.size[1];
  const X = gt[0] + u * ow * gt[1] + v * oh * gt[2];
  const Y = gt[3] + u * ow * gt[4] + v * oh * gt[5];
  const Z = dsmElev(X, Y);
  if(Z == null){ bpRead('no elevation data at that spot — click nearer the model'); return; }
  regOdm.push([X, Y, Z]); regOdmUV.push([u, v]); regBpJob = $('bpwrap').dataset.id;
  bpAddDot(u, v, regOdm.length - 1);
  bpRead(`map point ${regOdm.length} set (3D: ${regSplat.length} / map: ${regOdm.length})`);
});

$('bpalign').addEventListener('click', async () => {
  const id = $('bpwrap').dataset.id;
  if(regJob && regJob !== id){ bpRead('3D points belong to a different job — Clear pts'); return; }
  const n = Math.min(regSplat.length, regOdm.length);
  if(n < 3){ bpRead(`need ≥3 pairs (3D: ${regSplat.length} / map: ${regOdm.length})`); return; }
  const r = horn(regSplat.slice(0, n), regOdm.slice(0, n));
  if(!r){ bpRead('degenerate points — pick spread-out, non-collinear spots'); return; }
  const resp = await fetch(`/api/jobs/${id}/registration`, {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({scale: r.scale, q: r.q, t: r.t, rmse: r.rmse, n})});
  if(!resp.ok){ bpRead('failed to save alignment'); return; }
  if(recs[id]) recs[id].registration = r;
  bpRead(`aligned ✓ scale ${r.scale.toFixed(3)} map-u/u · RMSE ${r.rmse.toFixed(2)} map-u · ${n} pairs`);
  setBpReg(false);
});
$('bpclearpts').addEventListener('click', () => { clearRegState(); bpRead(''); setRead(''); });

function clearRegState(){
  regSplat = []; regOdm = []; regOdmUV = []; regJob = null; regBpJob = null;
  document.querySelectorAll('#bpcanvas .regdot').forEach(d => d.remove());
  if(regGroup) regGroup.clear();
}

// -- Horn's closed-form absolute orientation (quaternion, with scale)
function qRot(q, p){
  const w = q[0], x = q[1], y = q[2], z = q[3], px = p[0], py = p[1], pz = p[2];
  const uvx = y*pz - z*py, uvy = z*px - x*pz, uvz = x*py - y*px;
  const uuvx = y*uvz - z*uvy, uuvy = z*uvx - x*uvz, uuvz = x*uvy - y*uvx;
  return [px + 2*(w*uvx + uuvx), py + 2*(w*uvy + uuvy), pz + 2*(w*uvz + uuvz)];
}
function horn(src, dst){
  const n = src.length;
  const cs = [0,0,0], cd = [0,0,0];
  for(const p of src){ cs[0] += p[0]/n; cs[1] += p[1]/n; cs[2] += p[2]/n; }
  for(const p of dst){ cd[0] += p[0]/n; cd[1] += p[1]/n; cd[2] += p[2]/n; }
  const S = [[0,0,0],[0,0,0],[0,0,0]];
  let varS = 0;
  for(let i = 0; i < n; i++){
    const a = [src[i][0]-cs[0], src[i][1]-cs[1], src[i][2]-cs[2]];
    const b = [dst[i][0]-cd[0], dst[i][1]-cd[1], dst[i][2]-cd[2]];
    varS += a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
    for(let j = 0; j < 3; j++) for(let k = 0; k < 3; k++) S[j][k] += a[j]*b[k];
  }
  if(varS < 1e-12) return null;
  const Sxx=S[0][0], Sxy=S[0][1], Sxz=S[0][2], Syx=S[1][0], Syy=S[1][1], Syz=S[1][2], Szx=S[2][0], Szy=S[2][1], Szz=S[2][2];
  const N = [
    [Sxx+Syy+Szz, Syz-Szy,      Szx-Sxz,      Sxy-Syx],
    [Syz-Szy,     Sxx-Syy-Szz,  Sxy+Syx,      Szx+Sxz],
    [Szx-Sxz,     Sxy+Syx,     -Sxx+Syy-Szz,  Syz+Szy],
    [Sxy-Syx,     Szx+Sxz,      Syz+Szy,     -Sxx-Syy+Szz]];
  let c = 0;
  for(const row of N) c = Math.max(c, row.reduce((a, v) => a + Math.abs(v), 0));
  let q = [1, 0.001, 0.002, 0.003];  // asymmetric start so power iteration can't stall
  for(let it = 0; it < 200; it++){
    const r = [0,0,0,0];
    for(let j = 0; j < 4; j++){ r[j] = c*q[j]; for(let k = 0; k < 4; k++) r[j] += N[j][k]*q[k]; }
    const m = Math.hypot(r[0], r[1], r[2], r[3]);
    if(m < 1e-15) return null;
    q = [r[0]/m, r[1]/m, r[2]/m, r[3]/m];
  }
  let num = 0;
  for(let i = 0; i < n; i++){
    const a = qRot(q, [src[i][0]-cs[0], src[i][1]-cs[1], src[i][2]-cs[2]]);
    num += a[0]*(dst[i][0]-cd[0]) + a[1]*(dst[i][1]-cd[1]) + a[2]*(dst[i][2]-cd[2]);
  }
  const s = num / varS;
  if(!(s > 0) || !isFinite(s)) return null;
  const rc = qRot(q, cs);
  const t = [cd[0]-s*rc[0], cd[1]-s*rc[1], cd[2]-s*rc[2]];
  let se = 0;
  for(let i = 0; i < n; i++){
    const p = qRot(q, src[i]);
    const dx = s*p[0]+t[0]-dst[i][0], dy = s*p[1]+t[1]-dst[i][1], dz = s*p[2]+t[2]-dst[i][2];
    se += dx*dx + dy*dy + dz*dz;
  }
  return {scale: s, q, t, rmse: Math.sqrt(se/n)};
}

// ---------- viewer ----------
let viewer = null, vJob = null, vName = null, flipped = false;

// COLMAP puts each scene in an arbitrary coordinate frame, so a fixed camera
// can open onto empty space. Aim at the robust center of the splat mass.
function autoFrame(){
  const mesh = viewer.splatMesh;
  const n = mesh.getSplatCount();
  if(!n) return;
  const step = Math.max(1, Math.floor(n/20000));
  const pt = new THREE.Vector3();
  const xs=[], ys=[], zs=[];
  for(let i=0;i<n;i+=step){ mesh.getSplatCenter(i, pt); xs.push(pt.x); ys.push(pt.y); zs.push(pt.z); }
  const med = a => { a.sort((p,q)=>p-q); return a[a.length>>1]; };
  const c = new THREE.Vector3(med(xs), med(ys), med(zs));
  const ds = xs.map((x,i)=>Math.hypot(x-c.x, ys[i]-c.y, zs[i]-c.z)).sort((p,q)=>p-q);
  const r = Math.max(ds[Math.floor(ds.length*0.7)], 0.5);  // core radius, ignore floater shell
  coreRadius = r; flySpeed = r * 0.5;
  const dir = new THREE.Vector3(0,0,-1)
    .addScaledVector(new THREE.Vector3().copy(viewer.camera.up), 0.4).normalize();
  viewer.camera.position.copy(c).addScaledVector(dir, r*2.2);
  viewer.camera.lookAt(c);
  if(viewer.controls){ viewer.controls.target.copy(c); viewer.controls.update(); }
}
async function openViewer(jobId, name, keepFlip=false){
  if(!keepFlip) flipped = false;
  vJob = jobId; vName = name;
  $('viewerwrap').style.display = 'block';
  $('vprogress').style.display = 'block';
  $('vname').textContent = name || jobId;
  if(flyOn) setFly(false);
  if(walkOn) setWalk(false);
  if(walkMeshJob !== jobId){ walkMesh = null; walkMeshJob = null; }
  setMeasure(false); setReg3d(false);
  markGroup = null; regGroup = null; mPts = []; lastDist = null; setRead('');
  if(regJob && regJob !== jobId){ regSplat = []; regOdm = []; regOdmUV = []; regJob = null; }
  if(viewer){ try{ await viewer.dispose(); }catch(e){} viewer = null; }
  $('viewer').replaceChildren();
  viewer = new GS.Viewer({
    rootElement: $('viewer'),
    cameraUp: flipped ? [0,1,0] : [0,-1,0],
    initialCameraPosition: [0,0,-4], initialCameraLookAt: [0,0,0],
    sharedMemoryForWorkers: false, gpuAcceleratedSort: false, selfDrivenMode: true,
  });
  try{
    await viewer.addSplatScene(`/api/jobs/${jobId}/model.ply`, {
      format: GS.SceneFormat.Ply, showLoadingUI: false, progressiveLoad: false,
      splatAlphaRemovalThreshold: 5,
      onProgress: (p,l) => { $('vprogress').textContent = `${l||'loading'} ${Math.round(p)}%`; },
    });
    try{ autoFrame(); }catch(e){ console.log('auto-frame failed:', e); }
    if(viewer.threeScene){
      markGroup = new THREE.Group(); regGroup = new THREE.Group();
      viewer.threeScene.add(markGroup); viewer.threeScene.add(regGroup);
      if(regJob === jobId) regSplat.forEach((pt, i) => addRegMark(new THREE.Vector3(pt[0], pt[1], pt[2]), i));
    }
    viewer.start();
    $('vprogress').style.display = 'none';
  }catch(e){ $('vprogress').textContent = 'Failed to load model: ' + (e.message||e); }
}
$('vclose').addEventListener('click', async () => {
  $('viewerwrap').style.display = 'none';
  if(viewer){ try{ await viewer.dispose(); }catch(e){} viewer = null; $('viewer').replaceChildren(); }
});
$('vflip').addEventListener('click', () => { if(vJob){ flipped = !flipped; openViewer(vJob, vName, true); } });

// ---------- fly camera + measure tool ----------
let flyOn = false, measureOn = false, savedControls = null;
let coreRadius = 4, flySpeed = 2, flyLastT = 0;
let markGroup = null, mPts = [], lastDist = null;
let regGroup = null, regSplat = [], regOdm = [], regOdmUV = [], regJob = null, regOn3d = false;
const REGC = ['#9d8cff', '#6fd08c', '#e8909f', '#e8ebf2', '#f0b06a'];
const keys = {};

function setRead(t){ $('vread').textContent = t; }
function getScale(){
  const v = parseFloat(localStorage.getItem('splatscale-' + vJob));
  return (isFinite(v) && v > 0) ? v : null;
}
function fmtDist(d){
  const reg = (recs[vJob] || {}).registration;
  let out = d.toFixed(2) + ' u';
  if(reg && reg.scale > 0) out += ` = ${(d*reg.scale).toFixed(2)} map-u`;
  const s = getScale();
  if(s) out += ` = ${(d*s).toFixed(2)} m`;
  if(!reg && !s) out += ' (uncalibrated)';
  return out;
}

// -- fly: detach OrbitControls (its update() overwrites external camera moves)
function setFly(on){
  if(!viewer && on) return;
  flyOn = on;
  $('vfly').classList.toggle('on', on);
  if(on){
    setMeasure(false);
    setReg3d(false);
    savedControls = viewer.controls;
    viewer.controls = null;
    const el = viewer.renderer && viewer.renderer.domElement;
    if(el && el.requestPointerLock) el.requestPointerLock();
    flyLastT = performance.now();
    requestAnimationFrame(flyTick);
    setRead('WASD move · E/Q up/down · Shift fast · scroll speed · Esc exit');
  }else{
    if(document.exitPointerLock && document.pointerLockElement) document.exitPointerLock();
    if(viewer && savedControls){
      viewer.controls = savedControls;
      const fwd = new THREE.Vector3(0,0,-1).applyQuaternion(viewer.camera.quaternion);
      viewer.controls.target.copy(viewer.camera.position).addScaledVector(fwd, Math.max(coreRadius*0.6, 0.1));
      viewer.controls.update();
    }
    savedControls = null;
    setRead('');
  }
}
function flyTick(t){
  if(!flyOn || !viewer) return;
  const dt = Math.min((t - flyLastT)/1000, 0.1); flyLastT = t;
  const cam = viewer.camera;
  const fwd = new THREE.Vector3(0,0,-1).applyQuaternion(cam.quaternion);
  const right = new THREE.Vector3(1,0,0).applyQuaternion(cam.quaternion);
  const v = new THREE.Vector3();
  if(keys['KeyW']) v.add(fwd);   if(keys['KeyS']) v.sub(fwd);
  if(keys['KeyD']) v.add(right); if(keys['KeyA']) v.sub(right);
  if(keys['KeyE']||keys['Space']) v.add(cam.up); if(keys['KeyQ']) v.sub(cam.up);
  if(v.lengthSq() > 0){
    v.normalize().multiplyScalar(flySpeed * ((keys['ShiftLeft']||keys['ShiftRight']) ? 4 : 1) * dt);
    cam.position.add(v);
  }
  requestAnimationFrame(flyTick);
}
$('vfly').addEventListener('click', () => setFly(!flyOn));

// -- walk: gravity + collision against a Poisson mesh built from the splat
let walkOn = false, walkMesh = null, walkMeshJob = null, eyeH = 1, walkVel = 0, walkGrounded = false;
const walkRay = new THREE.Raycaster();
walkRay.firstHitOnly = true;   // three-mesh-bvh fast path

async function ensureWalkMesh(id){
  for(let i = 0; i < 90; i++){
    if(vJob !== id) return false;                    // viewer switched jobs
    const r = await fetch(`/api/jobs/${id}/mesh/collision.glb`);
    if(r.ok){
      const buf = await r.arrayBuffer();
      const gltf = await new GLTFLoader().parseAsync(buf, '');
      let found = null;
      gltf.scene.updateMatrixWorld(true);
      gltf.scene.traverse(o => { if(o.isMesh && !found) found = o; });
      if(!found){ setRead('walk mesh file is empty'); return false; }
      found.geometry.computeBoundsTree();
      walkMesh = found; walkMeshJob = id;
      return true;
    }
    if(i === 0){
      const p = await fetch(`/api/jobs/${id}/mesh`, {method: 'POST'});
      if(!p.ok){ setRead('cannot build walk mesh: ' + p.status); return false; }
      setRead('building walk mesh on a CPU worker (~2 min)…');
    }
    const rec = recs[id] || {};
    if(rec.mesh_status === 'error'){ setRead('walk mesh failed: ' + (rec.mesh_error || '')); return false; }
    if(rec.mesh_stage && rec.mesh_status === 'running') setRead('walk mesh: ' + rec.mesh_stage);
    await new Promise(res => setTimeout(res, 4000));
  }
  setRead('walk mesh timed out — try again');
  return false;
}

function groundUnder(pos, up){
  walkRay.set(pos.clone().addScaledVector(up, coreRadius), up.clone().negate());
  walkRay.far = coreRadius * 5;
  const hits = walkRay.intersectObject(walkMesh, false);
  return hits.length ? hits[0].point : null;
}

async function startWalk(){
  if(!viewer) return;
  if(walkMeshJob !== vJob) walkMesh = null;
  if(!walkMesh){
    $('vwalk').disabled = true;
    const ok = await ensureWalkMesh(vJob);
    $('vwalk').disabled = false;
    if(!ok) return;
  }
  setWalk(true);
}
function setWalk(on){
  if(walkOn === on) return;
  walkOn = on;
  $('vwalk').classList.toggle('on', on);
  if(on){
    setMeasure(false); setReg3d(false); if(flyOn) setFly(false);
    savedControls = viewer.controls;
    viewer.controls = null;
    const up = viewer.camera.up.clone().normalize();
    eyeH = Math.max(coreRadius * 0.02, 1e-4);
    walkVel = 0; walkGrounded = false;
    const g = groundUnder(viewer.camera.position, up);
    if(g) viewer.camera.position.copy(g).addScaledVector(up, eyeH);
    const el = viewer.renderer && viewer.renderer.domElement;
    if(el && el.requestPointerLock) el.requestPointerLock();
    flyLastT = performance.now();
    requestAnimationFrame(walkTick);
    setRead('WASD walk · Shift run · Space jump · scroll eye height · Esc exit');
  }else{
    if(document.exitPointerLock && document.pointerLockElement) document.exitPointerLock();
    if(viewer && savedControls){
      viewer.controls = savedControls;
      const fwd = new THREE.Vector3(0,0,-1).applyQuaternion(viewer.camera.quaternion);
      viewer.controls.target.copy(viewer.camera.position).addScaledVector(fwd, Math.max(coreRadius*0.6, 0.1));
      viewer.controls.update();
    }
    savedControls = null;
    setRead('');
  }
}
function walkTick(t){
  if(!walkOn || !viewer || !walkMesh) return;
  const dt = Math.min((t - flyLastT)/1000, 0.05); flyLastT = t;
  const cam = viewer.camera;
  const up = cam.up.clone().normalize();
  let feet = cam.position.clone().addScaledVector(up, -eyeH);

  // horizontal movement in the ground plane
  const fwd = new THREE.Vector3(0,0,-1).applyQuaternion(cam.quaternion);
  fwd.addScaledVector(up, -fwd.dot(up));
  const right = new THREE.Vector3(1,0,0).applyQuaternion(cam.quaternion);
  right.addScaledVector(up, -right.dot(up));
  const v = new THREE.Vector3();
  if(fwd.lengthSq() > 1e-8){ fwd.normalize(); if(keys['KeyW']) v.add(fwd); if(keys['KeyS']) v.sub(fwd); }
  if(right.lengthSq() > 1e-8){ right.normalize(); if(keys['KeyD']) v.add(right); if(keys['KeyA']) v.sub(right); }
  if(v.lengthSq() > 0){
    const speed = eyeH * 2.2 * ((keys['ShiftLeft']||keys['ShiftRight']) ? 3 : 1);  // ~walking pace
    v.normalize().multiplyScalar(speed * dt);
    // wall check at chest height; slide along the surface if blocked
    const chest = feet.clone().addScaledVector(up, eyeH * 0.55);
    walkRay.set(chest, v.clone().normalize());
    walkRay.far = v.length() + eyeH * 0.35;
    const hit = walkRay.intersectObject(walkMesh, false)[0];
    if(!hit) feet.add(v);
    else if(hit.face){
      const nrm = hit.face.normal.clone().transformDirection(walkMesh.matrixWorld);
      const slide = v.clone().addScaledVector(nrm, -v.dot(nrm));
      walkRay.set(chest, slide.clone().normalize());
      walkRay.far = slide.length() + eyeH * 0.35;
      if(slide.lengthSq() > 1e-12 && !walkRay.intersectObject(walkMesh, false).length) feet.add(slide);
    }
  }

  // gravity + ground clamp (step up/down follows terrain when grounded)
  if(keys['Space'] && walkGrounded){ walkVel = eyeH * 2.6; walkGrounded = false; }
  const g = groundUnder(feet.clone().addScaledVector(up, eyeH * 0.5), up);
  if(g){
    const above = feet.clone().sub(g).dot(up);
    if(walkGrounded && walkVel === 0 && above < eyeH * 0.6){
      feet = g;                                   // follow terrain
    }else{
      walkVel -= eyeH * 7.5 * dt;                 // fall
      feet.addScaledVector(up, walkVel * dt);
      if(feet.clone().sub(g).dot(up) <= 0){ feet = g; walkVel = 0; walkGrounded = true; }
      else walkGrounded = false;
    }
  }else{
    walkVel = 0; walkGrounded = false;            // over the void: hover
  }

  cam.position.copy(feet).addScaledVector(up, eyeH);
  requestAnimationFrame(walkTick);
}
$('vwalk').addEventListener('click', () => { walkOn ? setWalk(false) : startWalk(); });

document.addEventListener('pointerlockchange', () => {
  if(!document.pointerLockElement){
    if(flyOn) setFly(false);
    if(walkOn) setWalk(false);
  }
});
document.addEventListener('mousemove', e => {
  if(!(flyOn || walkOn) || !document.pointerLockElement || !viewer) return;
  const cam = viewer.camera;
  // yaw around the scene's visual up so the horizon stays level, pitch around local right
  const yaw = new THREE.Quaternion().setFromAxisAngle(cam.up, -e.movementX * 0.002);
  cam.quaternion.premultiply(yaw);
  const right = new THREE.Vector3(1,0,0).applyQuaternion(cam.quaternion);
  const pitch = new THREE.Quaternion().setFromAxisAngle(right, -e.movementY * 0.002);
  cam.quaternion.premultiply(pitch);
});
document.addEventListener('keydown', e => {
  if(e.target.tagName === 'INPUT') return;
  keys[e.code] = true;
  if(flyOn && ['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','Space'].includes(e.code)) e.preventDefault();
});
document.addEventListener('keyup', e => { keys[e.code] = false; });
document.addEventListener('wheel', e => {
  if(flyOn) flySpeed = Math.max(0.01, flySpeed * (e.deltaY < 0 ? 1.25 : 0.8));
  if(walkOn) eyeH = Math.max(1e-4, eyeH * (e.deltaY < 0 ? 1.12 : 0.9));
}, {passive: true});

// -- measure: pick splat surface points with the library's raycaster
function setMeasure(on){
  measureOn = on;
  $('vmeasure').classList.toggle('on', on);
  if(on){ if(flyOn) setFly(false); if(regOn3d) setReg3d(false); clearMarks(); setRead('click the first point on the model'); }
  else { clearMarks(); if(!regOn3d) setRead(''); }
}

// -- registration: splat-side point picking
function setReg3d(on){
  if(regOn3d === on) return;
  regOn3d = on;
  $('vreg').classList.toggle('on', on);
  if(on){
    if(flyOn) setFly(false);
    if(measureOn) setMeasure(false);
    setRead(`registration: click 3–5 distinctive spots on the model (have ${regSplat.length}), then mark them on the Blueprint in the same order`);
  } else if(!measureOn) setRead('');
}
$('vreg').addEventListener('click', () => setReg3d(!regOn3d));
function addRegMark(p, i){
  if(!regGroup) return;
  const s = new THREE.Mesh(new THREE.SphereGeometry(coreRadius*0.014, 16, 16),
                           new THREE.MeshBasicMaterial({color: REGC[i % REGC.length]}));
  s.position.copy(p); regGroup.add(s);
}
function clearMarks(){
  mPts = []; lastDist = null;
  if(markGroup) markGroup.clear();
  $('vcal').style.display = 'none';
}
function addMark(p){
  if(!markGroup) return;
  const s = new THREE.Mesh(new THREE.SphereGeometry(coreRadius*0.012, 16, 16),
                           new THREE.MeshBasicMaterial({color: 0x9d8cff}));
  s.position.copy(p); markGroup.add(s);
}
function pickPoint(e){
  const dims = new THREE.Vector2();
  viewer.getRenderDimensions(dims);
  viewer.raycaster.setFromCameraAndScreenPosition(
    viewer.camera, new THREE.Vector2(e.offsetX, e.offsetY), dims);
  const hits = [];
  viewer.raycaster.intersectSplatMesh(viewer.splatMesh, hits);
  return hits.length ? hits[0].origin.clone() : null;
}
let mDown = null;
$('viewer').addEventListener('mousedown', e => { mDown = [e.clientX, e.clientY]; }, true);
$('viewer').addEventListener('mouseup', e => {
  if((!measureOn && !regOn3d) || flyOn || !viewer) return;
  if(mDown && Math.hypot(e.clientX - mDown[0], e.clientY - mDown[1]) > 5) return;  // was a drag
  e.stopPropagation();  // keep the viewer's click-to-refocus out of picking clicks
  const p = pickPoint(e);
  if(!p){ setRead('no surface under cursor — click on the model'); return; }
  if(measureOn){
    if(mPts.length === 2) clearMarks();
    mPts.push(p); addMark(p);
    if(mPts.length === 2){
      lastDist = mPts[0].distanceTo(mPts[1]);
      if(markGroup){
        const g = new THREE.BufferGeometry().setFromPoints(mPts);
        markGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({color: 0x9d8cff})));
      }
      setRead(fmtDist(lastDist));
      $('vcal').style.display = '';
    }else setRead('point 1 set — click the second point');
  }else{
    if(regJob && regJob !== vJob){ setRead('registration points from another job pending — Clear pts in the Blueprint'); return; }
    if(regSplat.length >= 5){ setRead('5 points max — open the Blueprint to mark them and Align'); return; }
    regJob = vJob;
    regSplat.push([p.x, p.y, p.z]);
    addRegMark(p, regSplat.length - 1);
    setRead(`3D point ${regSplat.length} set (map: ${regOdm.length}) — mark the same spot on the Blueprint`);
  }
}, true);
$('vmeasure').addEventListener('click', () => setMeasure(!measureOn));
$('vcalok').addEventListener('click', () => {
  const m = parseFloat($('vcalin').value);
  if(!(m > 0) || !lastDist){ setRead('measure a distance first, then enter its real length'); return; }
  localStorage.setItem('splatscale-' + vJob, String(m / lastDist));
  $('vcalin').value = '';
  setRead(fmtDist(lastDist));
});
</script>
</body>
</html>
"""
