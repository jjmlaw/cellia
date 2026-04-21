# Architecture

Benchmarking environment for YOLOv12 on NVIDIA Jetson (JetPack 6), running inside a Docker container with JupyterLab for interactive throughput/accuracy experiments across model sizes and TensorRT vs non-TensorRT backends.

## Requirements

### Host (Jetson device)
- JetPack 6.1 (L4T r36.4.0) on Jetson AGX Orin, Orin NX, or Orin Nano
- Docker with NVIDIA Container Runtime (`--runtime=nvidia`)
- `nvpmodel` set to MAXN (`sudo nvpmodel -m 0`) and `sudo jetson_clocks` run before benchmarking for stable clocks
- `jetson-stats` installed on host (`sudo pip install jetson-stats`) for `jtop` thermal/power monitoring
- Sufficient free storage for datasets (COCO val2017 ≈ 1 GB) and TRT engine artifacts
- Swap recommended on Orin Nano (8 GB RAM) for larger model exports

### Container image
- Base: `nvcr.io/nvidia/l4t-jetpack:r36.4.0`
- Python 3.10 (system interpreter — do not use a venv, or TensorRT Python bindings from `/usr/lib/python3.10/dist-packages` will be shadowed)
- PyTorch 2.5 + torchvision (JetPack aarch64 wheels)
- onnxruntime-gpu 1.20 (aarch64 wheel)
- Ultralytics (editable install from repo) with `[export]` extras
- JupyterLab, notebook, ipywidgets, ipykernel
- matplotlib, pandas, seaborn for result visualization

### Network
- LAN access to the Jetson for remote JupyterLab (or local keyboard/monitor)
- Outbound HTTPS for pulling datasets / model weights on first run

## Port pass-throughs

| Host port | Container port | Purpose |
|-----------|----------------|---------|
| 8888      | 8888           | JupyterLab server |
| 6006      | 6006           | TensorBoard (optional, for training/eval curves) |

Launch with `-p 8888:8888 -p 6006:6006`. Bind Jupyter to `0.0.0.0` and run `--no-browser` since the container is headless.

## Mounted volumes

| Host path                  | Container path                  | Mode | Purpose |
|----------------------------|---------------------------------|------|---------|
| `./notebooks`              | `/ultralytics/notebooks`        | rw   | Jupyter notebooks — persisted across container restarts |
| `./results`                | `/ultralytics/results`          | rw   | Benchmark CSVs, plots, tegrastats logs |
| `./datasets`               | `/ultralytics/datasets`         | rw   | COCO val2017 and any custom eval sets (downloaded once, reused) |
| `./weights`                | `/ultralytics/weights`          | rw   | `.pt` checkpoints for YOLOv12 n/s/m/l/x |
| `./engines`                | `/ultralytics/engines`          | rw   | TensorRT `.engine` files — **device-specific**, do not share across Jetson variants |
| `/tmp/argus_socket`        | `/tmp/argus_socket`             | rw   | (Optional) CSI camera access via libargus |
| `/etc/localtime`           | `/etc/localtime`                | ro   | Match host timezone for log timestamps |

### Runtime flags

- `--runtime=nvidia` — required for GPU access
- `--ipc=host` — required for PyTorch multi-process dataloaders (shared memory)
- `--network=host` *or* explicit `-p` mappings — pick one; host networking is simpler on a trusted LAN
- `--name cellia-bench` — stable name for `docker exec` into a running session

### Example run command

```bash
t=ultralytics/ultralytics:latest-jetson-jetpack6
sudo docker run -it --rm \
  --runtime=nvidia --ipc=host \
  -p 8888:8888 -p 6006:6006 \
  -v $PWD/notebooks:/ultralytics/notebooks \
  -v $PWD/results:/ultralytics/results \
  -v $PWD/datasets:/ultralytics/datasets \
  -v $PWD/weights:/ultralytics/weights \
  -v $PWD/engines:/ultralytics/engines \
  --name cellia-bench $t
```
