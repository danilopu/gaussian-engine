# SplatPipe standalone app — deployment guide

One Python file (`splatpipe_app.py`) gives you a self-contained web app:

- **Upload page** — drop a video in the browser (works from your T480, your phone, anywhere)
- **Background pipeline** — a cloud T4 GPU spins up automatically, runs
  ffmpeg → COLMAP → Gaussian Splat training → export, then shuts down
- **Job list** — live status per video; jobs survive closing the tab, models are stored persistently
- **Built-in 3D viewer** — click "View in 3D" on any finished job, or download the `.ply`

It runs on [Modal](https://modal.com) (serverless GPUs). Free Starter tier: **$30/month
in credits, no credit card**. A scene costs ≈ $0.30–0.50 of T4 time (~30–50 min),
so roughly 60–90 free scenes per month. Idle cost: zero.

## Setup (once, ~5 minutes)

```
pip install modal
modal setup            # opens browser, log in / create free account
```

## Run it

Development mode (temporary URL, live-reloads when you edit the file):

```
modal serve splatpipe_app.py
```

Permanent deployment (stable URL you can bookmark, use from your phone, etc.):

```
modal deploy splatpipe_app.py
```

Both print a `https://…modal.run` URL — that's your app. Open it, drop a video, done.
The **first job is slower** (~10 extra min): the GPU image builds and gsplat compiles
its CUDA kernels. After that the image is cached.

## How it's put together (for when you want to hack on it)

| Piece | Where | What it does |
|---|---|---|
| `web()` | CPU container, always cheap | FastAPI: serves the page, accepts uploads, reports status, streams finished models |
| `process_video()` | T4 GPU container, on demand | Runs the three pipeline stages as subprocesses, pushes stage + log tail to the job dict |
| `modal.Volume` `splatpipe-data` | persistent storage | uploaded videos (deleted after success) and finished `model.ply` files |
| `modal.Dict` `splatpipe-jobs` | persistent state | job status records the frontend polls every 4 s |

Knobs you might want to turn, all in `process_video`:

- `--num-frames-target 150` → lower to 100 for faster COLMAP, raise to 250 for large scenes
- add `--max-num-iterations 15000` to the `ns-train` command for 2× faster, rougher models
- `gpu="T4"` → `gpu="L4"` or `gpu="A10G"` for ~2–3× faster training at a higher rate

## Important notes

- **The URL is public** to anyone who has it. For personal use, keeping it secret is
  usually fine; for real protection, Modal supports proxy auth
  (`modal.web_endpoint(requires_proxy_auth=True)`-style protection — see their docs)
  or put it behind Cloudflare Access.
- **Cost guardrail**: set a spend limit in the Modal dashboard (Settings → Usage limits)
  so a runaway job can never bill past your comfort level. The pipeline also has a
  hard 3-hour timeout per job.
- **First-run shakedown**: the GPU image pins torch 2.1.2 + CUDA 12.1 and installs
  nerfstudio from PyPI. ML packaging shifts over time; if the image build or first
  training run errors, paste the log into Claude and it's usually a one-line version
  pin fix.
- **Failure modes are the same as ever**: if a job errors at the "camera poses" stage,
  the video didn't have enough overlap/sharpness — refilm slower and wider.
- The standalone `splat-viewer.html` from before still works with downloaded `.ply`
  files if you ever want to view models offline.

## If you later get a GPU machine

The architecture ports cleanly: replace the Modal decorators with a local job queue
(e.g. FastAPI + a worker thread) and run the same three `ns-*` commands locally —
the frontend and pipeline logic don't change. Or switch the trainer to
[Brush](https://github.com/ArthurBrussee/brush), which trains on AMD/Intel/Apple GPUs.
