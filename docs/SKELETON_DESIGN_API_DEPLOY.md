# Skeleton Design inference — deploy runbook (API: Flask SSE + Redis cache)

This document is the handoff artifact for TechOps / backend. It covers everything
needed to build and run the Skeleton Design inference **HTTP API** on a GPU host.

The deliverable is a Docker image that runs a long-lived Flask server exposing:

- `POST /api/skeleton-design` (SSE): accepts `{ "audio_url": "..." }`, streams logs,
  and returns the final SnapBeat JSON inside the SSE `end` event.

The service caches results in Redis using an audio-content hash.

---

## 1. What you are receiving

| Item | Source | Notes |
|---|---|---|
| Source code + `Dockerfile.deploy` | Repo `lukasnorland/Mapperatorinator`, branch `amaremix` | Pin to a commit SHA for reproducible builds (see §3). |
| HuggingFace token (`HF_TOKEN`) | Shared privately | Fine-grained, scoped to **Read** on `lukasnorland/rhythm-skeleton-mt3-v2`. Required at **build** time to bake weights into the image. |
| Redis connection info | Provided by infra | Used at runtime for result/status caching. |

---

## 2. Host prerequisites

GPU host requirements (build + run):

- Linux x86_64 (tested on Ubuntu).
- NVIDIA GPU (≥ 12 GB VRAM recommended).
- NVIDIA driver + `nvidia-container-toolkit`.
- Docker Engine ≥ 24 with BuildKit.

Verification commands:

```bash
nvidia-smi
docker version --format '{{.Server.Version}}'
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If the last command fails, install `nvidia-container-toolkit` and restart Docker:
`https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`

---

## 3. One-time setup

```bash
git clone git@github.com:lukasnorland/Mapperatorinator.git
cd Mapperatorinator
git checkout amaremix
# Optional but recommended: pin to exact commit from handoff
# git checkout <commit-sha>
```

Store secrets in your secret manager. For an ad-hoc build:

```bash
export HF_TOKEN=hf_xxx_from_handoff
```

Do **not** commit tokens to git.

---

## 4. Build (bake base model + LoRA)

Build using BuildKit secret so `HF_TOKEN` never lands in an image layer:

```bash
DOCKER_BUILDKIT=1 docker build \
  -f Dockerfile.deploy \
  --secret id=hf_token,env=HF_TOKEN \
  -t snapbeat-api:mt3-v2 .
```

The Dockerfile downloads (bakes) both:

- `BASE_MODEL` (default: `OliBomby/Mapperatorinator-v31`)
- `LORA_REPO` (default: `lukasnorland/rhythm-skeleton-mt3-v2`)

Optional override:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.deploy \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg BASE_MODEL=OliBomby/Mapperatorinator-v31 \
  --build-arg LORA_REPO=lukasnorland/rhythm-skeleton-mt3-v2 \
  -t snapbeat-api:mt3-v2 .
```

---

## 5. Run

### 5.1 Redis config (required for caching)

The API uses these runtime environment variables:

| Var | Default | Meaning |
|---|---:|---|
| `REDIS_HOST` | `127.0.0.1` | Redis hostname/IP reachable from the container |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis DB index |
| `REDIS_TTL_SECONDS` | `86400` | TTL for results + status keys |

### 5.2 Start the API container

Run with GPU + published port:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -e PORT=8080 \
  -e REDIS_HOST=<redis-host> \
  -e REDIS_PORT=6379 \
  -e REDIS_DB=0 \
  -e REDIS_TTL_SECONDS=86400 \
  snapbeat-api:mt3-v2
```

API URL:

- `http://127.0.0.1:8080/api/skeleton-design`

---

## 6. API usage (SSE)

Request:

```bash
curl -N -H 'Content-Type: application/json' \
  -X POST 'http://127.0.0.1:8080/api/skeleton-design' \
  -d '{"audio_url":"https://.../song.mp3"}'
```

SSE events:

- `status`: lifecycle stages (download, hash computed, inference start/running, cache hit)
- `log`: subprocess stdout lines
- `end`: final payload

Final payload is always sent as:

```json
{ "status": "success", "results": { /* snapbeat json */ } }
```

Or on failure:

```json
{ "status": "error", "message": "...", "detail": "..." }
```

---

## 7. Caching (Redis keys)

The API computes `hashcode = sha256(audio_bytes)[:16]` (content-addressed; no `song_name` in keys).

Keys:

- Results: `skeleton_data:{hashcode}:results`
- Status: `skeleton_data:{hashcode}:status`

Poll using `hashcode` from SSE (`hash_computed` / `cache_hit`). Download failures occur before hash and are SSE-only (no Redis key).

Payloads:

- `:status` JSON includes `audio_url` (and `job_id`, `song_name`, etc.) on success and error paths after hash.
- `:results` JSON is an envelope `{"audio_url": "<request url>", "results": <SnapBeat object>}`. Older cached keys may still hold a raw SnapBeat object without `audio_url`.

Behavior:

- If `:results` exists for that `hashcode`, the API returns immediately with `cache_hit` (still downloads to compute hash).
- Otherwise it runs inference and writes `:results` + `:status`.

SSE final `end` events include `audio_url` alongside `results` / error fields where applicable.

---

## 8. Disk behavior / cleanup

The API downloads the audio into a job scratch directory:

- `tmp/__jobs__/{song_name}__{job_id}/...`

After completion (success or error), the API removes job scratch directories.
The recommended deployment is **Redis-only** caching (no persistent output files).

---

## 9. Troubleshooting

**SSE client timeouts**

- The server emits periodic `status` keep-alives while inference runs.

**401/403 when downloading LoRA**

- If you see auth errors during build, validate `HF_TOKEN` has **Read** access to
  the private LoRA repo.

**OOM**

- Reduce `max_batch_size` by passing Hydra overrides in the subprocess (requires
  updating the API to forward overrides). If you hit OOM frequently, use a larger GPU.

**Redis unreachable**

- Ensure `REDIS_HOST` is reachable from inside the container network.
- If using Docker for Redis, put both containers on the same network and use the
  Redis service/container name as `REDIS_HOST`.

