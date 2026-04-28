#!/usr/bin/env bash
set -euo pipefail

# To make containers visible in Docker Desktop, we run using the Docker Desktop
# engine context by default.
DOCKER_CTX="${DOCKER_CTX:-desktop-linux}"

IMAGE_NAME="${IMAGE_NAME:-snapbeat-api:test}"
CONTAINER_NAME="${CONTAINER_NAME:-snapbeat-api}"
HOST_PORT="${HOST_PORT:-8080}"
CONTAINER_PORT="${CONTAINER_PORT:-8080}"
ENV_FILE="${ENV_FILE:-.env}"

cd "$(dirname "$0")"

# Load env vars (including HF_TOKEN for build secret)
set -a
source "$ENV_FILE"
set +a

echo "Using docker context: ${DOCKER_CTX}"
docker context inspect "$DOCKER_CTX" >/dev/null 2>&1 || {
  echo "ERROR: docker context '${DOCKER_CTX}' not found."
  exit 1
}

echo "Stopping old container (Docker Desktop)..."
docker --context "$DOCKER_CTX" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Building image: ${IMAGE_NAME}"
DOCKER_BUILDKIT=1 docker --context "$DOCKER_CTX" build -f Dockerfile.deploy \
  --secret id=hf_token,env=HF_TOKEN \
  -t "${IMAGE_NAME}" .

echo "Running container: ${CONTAINER_NAME} on http://127.0.0.1:${HOST_PORT}"
docker --context "$DOCKER_CTX" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker --context "$DOCKER_CTX" run -d --name "$CONTAINER_NAME" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  --env-file "$ENV_FILE" \
  -e PORT="${CONTAINER_PORT}" \
  "${IMAGE_NAME}" >/dev/null

echo "POST http://127.0.0.1:${HOST_PORT}/api/skeleton-design"

