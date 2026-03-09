#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/hybridInference"

if [ -f "$HOME/.local/bin/env" ]; then
  source "$HOME/.local/bin/env"
fi

COMPOSE="docker compose -f $HOME/hybridInference/infrastructure/docker/docker-compose.staging.yml --env-file $HOME/hybridInference/.env"

echo "==> Installing Python dependencies"
uv sync

echo "==> Pulling infrastructure images"
$COMPOSE pull

echo "==> Starting infrastructure containers"
$COMPOSE up -d

echo "==> Container status"
$COMPOSE ps

echo "==> Restarting FastAPI service"
sudo systemctl restart hybrid_inference.staging
sudo systemctl status hybrid_inference.staging --no-pager
