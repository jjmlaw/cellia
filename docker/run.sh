#!/bin/bash
set -e

export SHELL=/bin/bash

mkdir -p /storage/data
rm -rf /storage/lost+found

if [ -z "$JUPYTER_PASSWORD" ]; then
    echo "ERROR: JUPYTER_PASSWORD environment variable must be set" >&2
    echo "Pass it at container start, e.g.:  docker run -e JUPYTER_PASSWORD=mysecret ..." >&2
    exit 1
fi

HASHED_PASSWORD=$(python3 -c "from jupyter_server.auth import passwd; import os; print(passwd(os.environ['JUPYTER_PASSWORD']))")

jupyter lab \
    --ip=0.0.0.0 \
    --port=9443 \
    --no-browser \
    --allow-root \
    --notebook-dir=/storage \
    --ServerApp.token='' \
    --ServerApp.password="$HASHED_PASSWORD"
