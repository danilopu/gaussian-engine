# SplatPipe — project context

## What this is
A personal video → 3D Gaussian Splat web app, deployed serverlessly on Modal.
The owner uploads a video (phone or drone footage) in a browser; a cloud T4 GPU
runs ffmpeg → COLMAP → splat training (nerfstudio/splatfacto) → .ply export in
the background; finished models are viewable in the built-in 3D viewer.

- `splatpipe_app.py` — the entire app: Modal images, GPU worker
  (`process_video`), FastAPI frontend (`web`), embedded HTML page (`PAGE`).
- `DEPLOY.md` — architecture notes, tunable knobs, cost/safety notes.

The owner's machine (ThinkPad T480, Windows, no CUDA GPU) only runs the Modal
CLI and a browser. All compute happens on Modal.

## Key commands
- `modal serve splatpipe_app.py` — dev mode, temporary URL, hot reload
- `modal deploy splatpipe_app.py` — permanent deployment, stable URL
- `modal app logs splatpipe` — stream logs (primary debugging tool)
- `modal app list` / `modal volume ls splatpipe-data` — inspect state

## Environment facts
- Windows host: prefer cross-platform commands; `python` not `python3` locally.
- Modal free tier: $30/month credits; T4 ≈ $0.59/h; a scene ≈ $0.30–0.50.
  A spend limit is set in the Modal dashboard. Never switch to bigger GPUs
  (A100/H100) without asking the owner.
- The deployed URL is unauthenticated. Do not add auth unless asked, but do
  not weaken this further (e.g. never add public listing/indexing).

## Known-fragile areas (check here first when something breaks)
1. **GPU image build** (`gpu_image` in splatpipe_app.py): torch 2.1.2 +
   CUDA 12.1 base + nerfstudio from PyPI. ML packaging drifts — if the build
   or first training run fails, fix by pinning compatible versions
   (torch / torchvision / nerfstudio / gsplat). Prefer minimal pin changes;
   record what was changed and why in a commit message.
2. **gsplat CUDA kernels**: JIT-compiled on first run (`TORCH_CUDA_ARCH_LIST=7.5`
   for T4). If compilation fails, try installing a prebuilt gsplat wheel
   matching the torch/CUDA pair, or pre-compile in an image build step with
   `gpu="T4"`.
3. **COLMAP from apt** is CPU-only — that's expected and fine; the pose stage
   simply takes 5–20 min.
4. **Volume visibility**: web container writes upload → `volume.commit()`;
   GPU container starts with `volume.reload()`. If files "don't exist",
   suspect a missing commit/reload, not the filesystem.
5. **Job status** lives in `modal.Dict` `splatpipe-jobs`; storage in
   `modal.Volume` `splatpipe-data`. Deleting these wipes history/models.

## Verification standard ("deployed" ≠ "done")
A change is verified only when:
1. `modal deploy` succeeds,
2. GET / returns the page,
3. an uploaded test video runs the full pipeline to `status: done`
   (watch `modal app logs splatpipe`; a full run takes 30–50 min — poll
   patiently rather than assuming failure),
4. `/api/jobs/{id}/model.ply` downloads a non-trivial file (tens of MB+).

## Conventions
- Keep everything in the single `splatpipe_app.py` file unless the owner
  asks to split it.
- git commit after each verified working change; small, descriptive commits.
- Cost discipline: don't launch repeated full training runs to debug —
  reproduce failures with the cheapest possible step (image build, or
  `--max-num-iterations 500` smoke runs) before a full verification run.
- Frontend: dark theme, Space Grotesk + IBM Plex Mono, iris (#9d8cff) accent.
  Preserve the existing look when editing the embedded PAGE.

## Roadmap (owner's stated interests — don't build unprompted)
- Phase 2: WebODM alongside this app for measurable/geo-referenced drone maps
  (orthomosaics, elevation models). Splats = visual model; ODM = survey map.
- Possible later: auth on the URL, quality presets in the UI, notifications.
