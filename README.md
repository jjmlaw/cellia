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

From the repo root (`cellia/`):

```bash
t=cellia/jetson-jetpack6:latest
sudo docker build --platform linux/arm64 -f docker/Dockerfile-jetson-jetpack6 -t $t .
```

> The Dockerfile lives in `docker/`, but the build context must be the **repo root** (`.`) so `COPY . .` and `COPY docker/run.sh /run.sh` can see the whole tree.

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
  -v $(pwd)/notebooks:/notebooks \
  --name cellia-bench $t
```

Flags explained:

- `--runtime=nvidia` — exposes the Jetson GPU to the container
- `--ipc=host` — shared memory for PyTorch dataloaders
- `-p 9443:9443` — JupyterLab port (host:container)
- `--env-file docker/.env` — loads `JUPYTER_PASSWORD` (and any other env) from the gitignored `docker/.env`
- `-v /storage:/storage` — persistent workspace (datasets, results, TRT engines)
- `-v $(pwd)/notebooks:/notebooks` — your notebooks, editable from the host

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
