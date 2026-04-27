# cellia

YOLOv12 benchmarking environment for NVIDIA Jetson (JetPack 6), packaged as a Docker container with JupyterLab for interactive throughput and accuracy experiments.

See `architecture.md` for mount points, port mappings, and full requirements.

## Jetson Docker container

### Prerequisites

- Jetson AGX Orin, Orin NX, or Orin Nano running JetPack 6.1 (L4T r36.4.0)
- Docker with the NVIDIA Container Runtime installed on the host
- Host directories `/storage` and `./notebooks` (relative to the repo root) exist and are writable

Before benchmarking, pin clocks and power mode on the host for stable numbers:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Build the image

The image layers JupyterLab and our entrypoint on top of the prebuilt `ultralytics/ultralytics:latest-jetson-jetpack6` image, which already ships with CUDA-enabled PyTorch, TensorRT bindings, and Ultralytics.

From the repo root (`cellia/`):

```bash
t=cellia/jetson-jetpack6:latest
sudo docker build --platform linux/arm64 -f docker/Dockerfile-jetson-jetpack6 -t $t .
```

> The Dockerfile lives in `docker/`, but the build context is the **repo root** (`.`) so that `COPY docker/run.sh /run.sh` can see the entrypoint script.

First build will pull the ~8 GB base image; subsequent builds are fast (~1–2 min) since only the Jupyter layer changes.

### Set the JupyterLab password

JupyterLab is **password protected**. The password is read from a local `docker/.env` file (gitignored — never committed).

1. Copy the template:

   ```bash
   cp docker/.env.example docker/.env
   ```

2. Edit `docker/.env` and set a strong password:

   ```bash
   JUPYTER_PASSWORD=your-strong-password
   ```

`.env` is listed in `.gitignore`, so it will not be tracked by git. The container entrypoint hashes the password at startup and configures Jupyter to require it at login. The container will refuse to start if `JUPYTER_PASSWORD` is empty or missing.

### Start the container

From the repo root:

```bash
t=cellia/jetson-jetpack6:latest
sudo docker run -it --rm \
  --ipc=host --runtime=nvidia \
  -p 9443:9443 \
  --env-file docker/.env \
  -v /storage:/storage \
  -v $(pwd):/cellia \
  --name cellia-bench $t
```

Flags explained:

- `--runtime=nvidia` — exposes the Jetson GPU to the container
- `--ipc=host` — shared memory for PyTorch dataloaders
- `-p 9443:9443` — JupyterLab port (host:container)
- `--env-file docker/.env` — loads `JUPYTER_PASSWORD` (and any other env) from the gitignored `docker/.env`
- `-v /storage:/storage` — persistent workspace (datasets, results, TRT engines)
- `-v $(pwd):/cellia` — repo root mounted at `/cellia` (Jupyter starts here, so `scripts/` and `notebooks/` are visible side-by-side)

### Verify the CUDA install

Before running benchmarks, confirm the container sees the Jetson GPU and that PyTorch is the CUDA-enabled Jetson build (not a CPU wheel from PyPI):

```bash
sudo docker run --rm --runtime=nvidia --ipc=host $t \
  python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

Expected output (device name varies by Jetson model):

```
2.5.0a0+872d972e41.nv24.08 True Orin
```

What to check:

- Version string ends in `+...nv24.08` — confirms the Jetson CUDA wheel, not a generic PyPI build
- `True` — PyTorch can reach the GPU
- A device name (`Orin`, etc.) — the NVIDIA runtime is wired up correctly

If you get `False` or `Torch not compiled with CUDA enabled`, either `--runtime=nvidia` was missing or the image was rebuilt from `l4t-jetpack` directly (which re-resolves torch from PyPI and clobbers the Jetson wheel — use the prebuilt `ultralytics/ultralytics:latest-jetson-jetpack6` base as this Dockerfile does).

### Access JupyterLab

On container start, the server listens on port 9443 with token auth disabled and password auth enabled. From another machine on the LAN, open:

```
http://<jetson-ip>:9443/lab
```

You will be prompted for the password you set via `JUPYTER_PASSWORD`.

### Changing the password

Edit `docker/.env`, save, then stop and restart the container. The hash is regenerated from `JUPYTER_PASSWORD` on every container start — nothing is persisted inside the image.

### Stop the container

```bash
sudo docker stop cellia-bench
```

(Or `Ctrl+C` in the terminal where it was launched — `--rm` will clean it up.)
