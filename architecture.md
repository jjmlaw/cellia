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
| 9443      | 9443           | JupyterLab server (password protected) |
| 6006      | 6006           | TensorBoard (optional, for training/eval curves) |

Launch with `-p 9443:9443 -p 6006:6006`. Jupyter binds `0.0.0.0` and runs `--no-browser` since the container is headless.

## Mounted volumes

| Host path                  | Container path    | Mode | Purpose |
|----------------------------|-------------------|------|---------|
| `/storage`                 | `/storage`        | rw   | Persistent workspace: datasets, results, TRT engines, downloaded weights |
| `$PWD/notebooks`           | `/notebooks`      | rw   | Jupyter notebooks (tracked in the repo, bind-mounted so edits persist) |
| `/tmp/argus_socket`        | `/tmp/argus_socket` | rw | (Optional) CSI camera access via libargus |
| `/etc/localtime`           | `/etc/localtime`  | ro   | Match host timezone for log timestamps |

TensorRT `.engine` files are device- and TRT-version-specific — build them on the target Jetson and keep them under `/storage/engines/<device>/`. Do not share engines across Jetson variants.

## Secrets

The JupyterLab password is supplied via `docker/.env` (gitignored). Copy `docker/.env.example` to `docker/.env`, set `JUPYTER_PASSWORD`, and pass `--env-file docker/.env` to `docker run`. The entrypoint hashes the password at startup; nothing is persisted inside the image.

### Runtime flags

- `--runtime=nvidia` — required for GPU access
- `--ipc=host` — required for PyTorch multi-process dataloaders (shared memory)
- `--env-file docker/.env` — loads `JUPYTER_PASSWORD` (required)
- `-p 9443:9443` — explicit port publish (alternative: `--network=host` on a trusted LAN)
- `--name cellia-bench` — stable name for `docker exec` into a running session

### Example run command

```bash
t=cellia/jetson-jetpack6:latest
sudo docker run -it --rm \
  --runtime=nvidia --ipc=host \
  -p 9443:9443 -p 6006:6006 \
  --env-file docker/.env \
  -v /storage:/storage \
  -v $PWD/notebooks:/notebooks \
  --name cellia-bench $t
```

Build from the repo root with the Dockerfile in `docker/`:

```bash
sudo docker build --platform linux/arm64 -f docker/Dockerfile-jetson-jetpack6 -t $t .
```
