#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f "$HOME/.local/bin/env" ]; then
  source "$HOME/.local/bin/env"
fi

COMPOSE="docker compose -f infrastructure/docker/docker-compose.staging.yml"

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
