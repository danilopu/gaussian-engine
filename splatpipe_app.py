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

        # free space: keep only the final model + a few debug bits
        for sub in ("proc", "outputs", "export"):
            shutil.rmtree(job_dir / sub, ignore_errors=True)
        video.unlink(missing_ok=True)
        volume.commit()

        _set(job_id, status="done", stage="done",
             size_mb=round((job_dir / "model.ply").stat().st_size / 1e6, 1))
    except Exception as e:
        volume.commit()
        _set(job_id, status="error", error=str(e))
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
.job a,.job button.view{font:500 13px 'Space Grotesk',sans-serif;color:var(--text);background:transparent;
  border:1px solid var(--line);border-radius:8px;padding:7px 14px;cursor:pointer;text-decoration:none}
.job a:hover,.job button.view:hover{border-color:var(--iris);color:var(--iris)}
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

<div id="viewerwrap">
  <div id="viewer"></div>
  <div id="vprogress">loading&hellip;</div>
  <div id="viewerbar">
    <span class="vname" id="vname"></span>
    <button id="vflip" title="Model upside down? Flip the camera's up axis">Flip up-axis</button>
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
      const stage = st==='error' ? (r.error||'failed') : (r.stage||st) + fmtElapsed(r);
      const log = st==='running' && r.log ? `\n${r.log}` : '';
      const act = st==='done'
        ? `<button class="view" data-id="${r.job_id}" data-name="${esc(r.name)}">View in 3D</button>
           <a href="/api/jobs/${r.job_id}/model.ply">Download .ply${r.size_mb?` (${r.size_mb} MB)`:''}</a>`
        : '';
      return `<div class="job"><span class="dot ${st}"></span>
        <div class="meta"><div class="name">${esc(r.name)||r.job_id}</div>
        <div class="stage ${st==='error'?'error':''}">${esc(stage)}${esc(log)}</div></div>${act}</div>`;
    }).join('');
    box.querySelectorAll('button.view').forEach(b =>
      b.addEventListener('click', () => openViewer(b.dataset.id, b.dataset.name)));
  }catch(e){ /* transient network errors: just retry next tick */ }
}
refresh(); setInterval(refresh, 4000);

// ---------- viewer ----------
let viewer = null, vJob = null, vName = null, flipped = false;
async function openViewer(jobId, name, keepFlip=false){
  if(!keepFlip) flipped = false;
  vJob = jobId; vName = name;
  $('viewerwrap').style.display = 'block';
  $('vprogress').style.display = 'block';
  $('vname').textContent = name || jobId;
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
    viewer.start();
    $('vprogress').style.display = 'none';
  }catch(e){ $('vprogress').textContent = 'Failed to load model: ' + (e.message||e); }
}
$('vclose').addEventListener('click', async () => {
  $('viewerwrap').style.display = 'none';
  if(viewer){ try{ await viewer.dispose(); }catch(e){} viewer = null; $('viewer').replaceChildren(); }
});
$('vflip').addEventListener('click', () => { if(vJob){ flipped = !flipped; openViewer(vJob, vName, true); } });
</script>
</body>
</html>
"""
