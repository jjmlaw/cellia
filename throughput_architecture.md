# Throughput Notebook Architecture

A single Jupyter notebook (`notebooks/throughput.ipynb`) run inside the Jetson container to measure **inference throughput** of a YOLOv12 model under different sizes, compression backends, and input sources. Centered on `model.predict()` — the same code path that runs in production. Accuracy is available as an opt-in sanity check, not the headline.

Keep it simple to start.

## Configurable inputs (top cell)

### Model + backend
| Name | Type | Default | Notes |
|------|------|---------|-------|
| `model_variant` | str | `yolov12n` | `yolov12{n,s,m,l,x}` or path to a custom `.pt` |
| `weights_path` | str \| None | None | If None, Ultralytics auto-downloads by variant |
| `backend` | str | `pytorch` | `pytorch`, `pytorch-half`, `onnx`, `tensorrt-fp32`, `tensorrt-fp16`, `tensorrt-int8` |
| `imgsz` | int | 640 | Inference image size |
| `batch_size` | int | 1 | Per-iteration batch |
| `conf` | float | 0.25 | Production-style threshold (not 0.001) |
| `iou` | float | 0.45 | Production-style NMS threshold |
| `device` | str | `cuda:0` | |

### Input source (production-shaped frames)
| Name | Type | Default | Notes |
|------|------|---------|-------|
| `source_kind` | str | `synthetic` | `synthetic`, `image`, `image_dir`, `video` |
| `source_path` | str \| None | None | Required for `image`, `image_dir`, `video` |

- `synthetic` — random uint8 array of shape `(imgsz, imgsz, 3)` cycled through `predict()`. Cleanest baseline across backends; no I/O cost.
- `image` — single real image, looped. Fixed content; representative of a static-scene benchmark.
- `image_dir` — cycles over images in the directory. Real input variability (varied sizes, varied detection counts → varied NMS cost).
- `video` — pulls frames from a video file in order; closest to a camera feed.

### Measurement scope (each toggle on/off)
| Name | Type | Default | Notes |
|------|------|---------|-------|
| `include_io` | bool | True | Count disk read / decode in latency. Always False for `synthetic`. |
| `include_preprocess` | bool | True | Count letterbox + tensor prep |
| `include_postprocess` | bool | True | Count NMS + result building |

Defaults reflect the **full deployed pipeline** — what the production system actually pays. Toggle off to isolate kernel-only inference time when comparing backends.

### Iteration counts
| Name | Type | Default | Notes |
|------|------|---------|-------|
| `warmup_iters` | int | 20 | Discarded — covers CUDA graph capture, kernel autotune, allocator warmup |
| `measure_iters` | int | 200 | Used for FPS + percentile statistics |

### Accuracy sanity check (opt-in)
| Name | Type | Default | Notes |
|------|------|---------|-------|
| `run_accuracy_check` | bool | False | When True, runs `model.val(data=data_yaml, save_json=True)` after the throughput pass |
| `data_yaml` | str | `coco.yaml` | Used by accuracy check + INT8 calibration |
| `int8_calib_images` | int | 300 | TRT INT8 calibration subset size |

Engines are cached at `/storage/engines/<device>/<variant>_<imgsz>_<backend>.engine` and rebuilt only when missing.

### Run identity
| Name | Type | Default | Notes |
|------|------|---------|-------|
| `run_tag` | str | `""` | Free-text label written to manifest |
| `output_dir` | path | `/storage/results/<run_id>` | `run_id` = timestamp + tag |

## Outputs

### Throughput (primary)
- FPS at the configured `batch_size` and shape
- Latency mean / p50 / p95 / p99 (ms)
- Cold-start latency reported separately (first iteration of the warmup window)
- Ultralytics `Profile` breakdown: preprocess / inference / postprocess (ms per frame)
- Raw per-iteration latencies saved to `latencies.npy`

### Accuracy (only if `run_accuracy_check=True`)
- mAP@0.5, mAP@0.5:0.95, precision, recall
- `predictions.json` (COCO format)

### Artifacts per run (`output_dir/`)
- `manifest.json` — full input config + git SHA + versions (JetPack, torch, TRT, ultralytics)
- `metrics.json` — throughput numbers (and accuracy if enabled)
- `latencies.npy` — raw per-iteration timings, for histogram replay / re-analysis
- `predictions.json` — only when accuracy check is on

### In-notebook
- pandas DataFrame appended one row per run for in-session comparison
- Plots: latency histogram, FPS bar across runs in the table

## Notebook sections

1. **Config** — inputs above in one cell
2. **Environment check** — torch CUDA, TRT version, device name; warn if `jetson_clocks` not set
3. **Model load / backend prep** — load `.pt`; for ONNX/TRT, export (or reuse cached engine); INT8 path uses a 300-image COCO val subset for calibration
4. **Source prep** — load `synthetic` / `image` / `image_dir` / `video` into a frame iterator that yields what the configured measurement scope requires (raw path vs preloaded array)
5. **Throughput pass** — `warmup_iters` discarded, then `measure_iters` timed with `torch.cuda.synchronize()` around each `model.predict()` call; latencies, percentiles, and Ultralytics profile recorded
6. **Accuracy sanity check** — gated on `run_accuracy_check`; runs `model.val(..., save_json=True)`
7. **Persist artifacts** — write `manifest.json`, `metrics.json`, `latencies.npy`, optionally `predictions.json`
8. **Append + plot** — DataFrame row + latency histogram + cross-run FPS bar

## Out of scope (for now)
- pycocotools official COCO eval
- `jtop` system telemetry (skip silently if module not importable)
- Batch-size or imgsz sweep in a single run (re-run with new config)
- Multi-stream / concurrent inference
- Engine-build timing as a measured number (informational only)

## Open follow-ups
- INT8 calibration subset: fixed checked-in list vs. re-sampled per run
- Whether to add an "unbatched real-time loop" mode that drives `predict()` from a live camera (`/dev/video0`) once a real source is wired up
