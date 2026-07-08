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
        return {"marker": "v-odm-1", "autoframe_in_page": "autoFrame" in PAGE}

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
                   "hillshade.png": "image/png",
                   "dsm.tif": "image/tiff", "mesh.zip": "application/zip"}
        if fname not in allowed:
            return JSONResponse({"error": "unknown file"}, status_code=404)
        volume.reload()
        path = Path(DATA) / "jobs" / job_id / "odm" / fname
        if not path.exists():
            return JSONResponse({"error": "not ready"}, status_code=404)
        return FileResponse(path, media_type=allowed[fname], filename=f"{job_id}-{fname}")

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
#bpwrap{display:none;position:fixed;inset:0;z-index:30;background:var(--ink);overflow:auto}
#bpwrap img{display:block;margin:0 auto;max-width:calc(100vw - 28px);height:auto;padding:64px 0 20px}
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
  <img id="bpimg" alt="orthophoto"/>
  <div id="bpbar">
    <span class="vname" id="bpname"></span>
    <button id="bpmode" title="Toggle orthophoto / elevation relief">Relief</button>
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
    <button id="vmeasure" title="Click two points on the model to measure the distance">Measure</button>
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
  "@mkkellogg/gaussian-splats-3d":"https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"
}}
</script>
<script type="module">
import * as THREE from 'three';
import * as GS from '@mkkellogg/gaussian-splats-3d';

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
    const box = $('jobs');
    if(!list.length){ box.innerHTML = '<div id="empty">Nothing yet. Your finished models appear here \u2014 they stay saved between visits.</div>'; return; }
    box.innerHTML = list.map(r => {
      const st = r.status || 'queued';
      let stage = st==='error' ? (r.error||'failed') : (r.stage||st) + fmtElapsed(r);
      const log = st==='running' && r.log ? `\n${r.log}` : '';
      const os = r.odm_status;
      if(os==='queued'||os==='running') stage += `\nmap layer: ${r.odm_stage||os}${r.odm_log?'\n'+r.odm_log:''}`;
      else if(os==='error') stage += `\nmap layer failed: ${r.odm_error||''}`;
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

let bpMode = 'orthophoto';
function openBlueprint(id, name){
  bpMode = 'orthophoto';
  $('bpwrap').dataset.id = id;
  $('bpname').textContent = (name || id) + ' — orthophoto';
  $('bpimg').src = `/api/jobs/${id}/odm/orthophoto.png`;
  $('bpdsm').href = `/api/jobs/${id}/odm/dsm.tif`;
  $('bpmode').textContent = 'Relief';
  $('bpwrap').style.display = 'block';
}
$('bpmode').addEventListener('click', () => {
  const id = $('bpwrap').dataset.id;
  bpMode = bpMode === 'orthophoto' ? 'hillshade' : 'orthophoto';
  $('bpimg').src = `/api/jobs/${id}/odm/${bpMode}.png`;
  $('bpname').textContent = $('bpname').textContent.replace(/— .*$/, '— ' + (bpMode === 'orthophoto' ? 'orthophoto' : 'elevation relief'));
  $('bpmode').textContent = bpMode === 'orthophoto' ? 'Relief' : 'Ortho';
});
$('bpclose').addEventListener('click', () => { $('bpwrap').style.display = 'none'; $('bpimg').removeAttribute('src'); });

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
  setMeasure(false); markGroup = null; mPts = []; lastDist = null; setRead('');
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
    if(viewer.threeScene){ markGroup = new THREE.Group(); viewer.threeScene.add(markGroup); }
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
const keys = {};

function setRead(t){ $('vread').textContent = t; }
function getScale(){
  const v = parseFloat(localStorage.getItem('splatscale-' + vJob));
  return (isFinite(v) && v > 0) ? v : null;
}
function fmtDist(d){
  const s = getScale();
  return d.toFixed(2) + ' units' + (s ? ` = ${(d*s).toFixed(2)} m` : ' (uncalibrated)');
}

// -- fly: detach OrbitControls (its update() overwrites external camera moves)
function setFly(on){
  if(!viewer && on) return;
  flyOn = on;
  $('vfly').classList.toggle('on', on);
  if(on){
    setMeasure(false);
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
document.addEventListener('pointerlockchange', () => {
  if(!document.pointerLockElement && flyOn) setFly(false);
});
document.addEventListener('mousemove', e => {
  if(!flyOn || !document.pointerLockElement || !viewer) return;
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
}, {passive: true});

// -- measure: pick splat surface points with the library's raycaster
function setMeasure(on){
  measureOn = on;
  $('vmeasure').classList.toggle('on', on);
  if(on){ if(flyOn) setFly(false); clearMarks(); setRead('click the first point on the model'); }
  else { clearMarks(); setRead(''); }
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
  if(!measureOn || flyOn || !viewer) return;
  if(mDown && Math.hypot(e.clientX - mDown[0], e.clientY - mDown[1]) > 5) return;  // was a drag
  e.stopPropagation();  // keep the viewer's click-to-refocus out of measure clicks
  const p = pickPoint(e);
  if(!p){ setRead('no surface under cursor — click on the model'); return; }
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
