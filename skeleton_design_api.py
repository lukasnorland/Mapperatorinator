from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import selectors
import uuid
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse, unquote

import requests
import redis
from flask import Flask, Response, request


APP_ROOT = Path(__file__).resolve().parent
TMP_ROOT = APP_ROOT / "tmp"

app = Flask(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1").strip() or "127.0.0.1"
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_TTL_SECONDS = int(os.environ.get("REDIS_TTL_SECONDS", "86400"))
REDIS_PASSWORD = (os.environ.get("REDIS_PASSWORD") or "").strip() or None
REDIS_CLUSTER_MODE = (os.environ.get("REDIS_CLUSTER_MODE") or "").strip().lower() in {"1", "true", "yes", "y"}
REDIS_TLS = (os.environ.get("REDIS_TLS") or "").strip().lower() in {"1", "true", "yes", "y"}


def _redis_client() -> redis.Redis | None:
    try:
        # Fail fast if Redis is misconfigured/unreachable. This API should still work
        # (just without caching) when Redis isn't available.
        if REDIS_CLUSTER_MODE:
            from redis.cluster import RedisCluster

            r = RedisCluster(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=0.3,
                socket_timeout=0.3,
                ssl=REDIS_TLS,
                require_full_coverage=False,
            )
            r.ping()
            return r  # type: ignore[return-value]

        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=0.3,
            socket_timeout=0.3,
            ssl=REDIS_TLS,
        )
        r.ping()
        return r
    except Exception:
        return None


def _build_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    hf_token = (env.get("HF_TOKEN") or "").strip()
    if hf_token and not (env.get("HUGGINGFACE_HUB_TOKEN") or "").strip():
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
    return env


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _slugify_song_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "audio"


def _song_name_from_audio_url(audio_url: str) -> str:
    parsed = urlparse(audio_url)
    base = unquote(os.path.basename(parsed.path))
    stem = Path(base).stem if base else ""
    return _slugify_song_name(stem)


def _filename_from_audio_url(audio_url: str) -> str:
    parsed = urlparse(audio_url)
    base = unquote(os.path.basename(parsed.path)).strip()
    return base or "audio.bin"


def _guess_ext_from_url(audio_url: str) -> str:
    parsed = urlparse(audio_url)
    base = unquote(os.path.basename(parsed.path))
    ext = Path(base).suffix.lower()
    if ext and len(ext) <= 8:
        return ext
    return ""


def _sse(event: str, data_obj) -> str:
    payload = json.dumps(data_obj, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _download_audio(audio_url: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    def _stream_to_file(resp: requests.Response) -> None:
        resp.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    try:
        with requests.get(audio_url, stream=True, timeout=(10, 300)) as r:
            _stream_to_file(r)
    except requests.exceptions.ProxyError:
        # Some environments (notably Docker Desktop + corporate networking) can end up with
        # proxy settings applied implicitly. Retry once with trust_env disabled to force
        # a direct connection.
        with requests.Session() as s:
            s.trust_env = False
            with s.get(audio_url, stream=True, timeout=(10, 300)) as r:
                _stream_to_file(r)


def _run_snapbeat_inference(audio_path: Path, output_path: Path) -> tuple[int, Iterator[str]]:
    cmd = [
        sys.executable,
        str(APP_ROOT / "snapbeat_inference.py"),
        f"audio_path={str(audio_path)}",
        f"output_path={str(output_path)}",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    def iter_lines() -> Iterator[str]:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip("\n")
        proc.stdout.close()
        return_code = proc.wait()
        yield f"__EXIT_CODE__={return_code}"

    return 0, iter_lines()


@app.post("/api/skeleton-design")
def snapbeat_sse() -> Response:
    body = request.get_json(silent=True) or {}
    audio_url = (body.get("audio_url") or "").strip()
    if not audio_url:
        return Response(_sse("error", {"message": "Missing audio_url"}), mimetype="text/event-stream")

    job_id = uuid.uuid4().hex
    song_name = _song_name_from_audio_url(audio_url)
    filename = _filename_from_audio_url(audio_url)

    def generate() -> Iterator[str]:
        r: redis.Redis | None = None
        hashcode: str | None = None
        try:
            yield _sse("status", {"stage": "init", "job_id": job_id, "song_name": song_name, "audio_url": audio_url})
            r = _redis_client()
            yield _sse("status", {"stage": "download", "audio_url": audio_url})

            # Download into a per-job folder first so we can hash the bytes.
            job_dir = TMP_ROOT / "__jobs__" / f"{song_name}__{job_id}"
            audio_path = job_dir / filename
            output_path = job_dir

            _download_audio(audio_url, audio_path)
            yield _sse("status", {"stage": "download_done", "audio_path": str(audio_path)})

            hashcode_full = _file_sha256(audio_path)
            hashcode = hashcode_full[:16]
            yield _sse("status", {"stage": "hash_computed", "hashcode": hashcode, "audio_url": audio_url})

            # Content-addressed keys (same bytes → same hash): reuse across jobs without duplicate storage.
            redis_results_key = f"skeleton_data:{hashcode}:results"
            redis_status_key = f"skeleton_data:{hashcode}:status"
            if r is not None:
                cached = r.get(redis_results_key)
                if cached:
                    try:
                        cached_parsed = json.loads(cached)
                        r.setex(
                            redis_status_key,
                            REDIS_TTL_SECONDS,
                            json.dumps(
                                {
                                    "status": "success",
                                    "stage": "cache_hit",
                                    "job_id": job_id,
                                    "song_name": song_name,
                                    "hashcode": hashcode,
                                    "audio_url": audio_url,
                                }
                            ),
                        )
                        yield _sse(
                            "status",
                            {
                                "stage": "cache_hit",
                                "hashcode": hashcode,
                                "results_key": redis_results_key,
                                "status_key": redis_status_key,
                                "audio_url": audio_url,
                            },
                        )
                        if isinstance(cached_parsed, dict) and "results" in cached_parsed:
                            results_out = cached_parsed["results"]
                            out_audio_url = cached_parsed.get("audio_url", audio_url)
                        else:
                            results_out = cached_parsed
                            out_audio_url = audio_url
                        yield _sse("end", {"status": "success", "results": results_out, "audio_url": out_audio_url})
                        return
                    except Exception:
                        # If cache is corrupted, ignore and recompute.
                        pass
                r.setex(
                    redis_status_key,
                    REDIS_TTL_SECONDS,
                    json.dumps(
                        {
                            "status": "running",
                            "stage": "download_done",
                            "job_id": job_id,
                            "song_name": song_name,
                            "audio_url": audio_url,
                        }
                    ),
                )

            yield _sse("status", {"stage": "inference_start"})

            # Stream subprocess output as log events.
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(APP_ROOT / "snapbeat_inference.py"),
                    f"audio_path={str(audio_path)}",
                    f"output_path={str(output_path)}",
                ],
                env=_build_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None

            # Non-blocking stdout reader that can handle tqdm-style '\r' updates.
            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ)
            os.set_blocking(proc.stdout.fileno(), False)
            buf = ""

            last_ping = time.time()
            ping_every_s = 5.0

            def _emit_ready_lines() -> Iterator[str]:
                nonlocal buf
                # Split on either newline or carriage return.
                while True:
                    m = re.search(r"[\r\n]", buf)
                    if not m:
                        return
                    idx = m.start()
                    line = buf[:idx]
                    buf = buf[idx + 1 :]
                    if line:
                        yield _sse("log", {"line": line})

            while True:
                events = sel.select(timeout=1.0)
                if events:
                    try:
                        chunk = proc.stdout.read()  # type: ignore[call-arg]
                    except BlockingIOError:
                        chunk = ""
                    if chunk:
                        buf += chunk
                        for msg in _emit_ready_lines():
                            yield msg
                        last_ping = time.time()

                if proc.poll() is not None:
                    # Flush whatever is left in the buffer.
                    if buf.strip():
                        yield _sse("log", {"line": buf.strip()})
                    break

                if (time.time() - last_ping) >= ping_every_s:
                    yield _sse("status", {"stage": "inference_running"})
                    if r is not None:
                        r.setex(
                            redis_status_key,
                            REDIS_TTL_SECONDS,
                            json.dumps(
                                {
                                    "status": "running",
                                    "stage": "inference_running",
                                    "job_id": job_id,
                                    "song_name": song_name,
                                    "audio_url": audio_url,
                                }
                            ),
                        )
                    last_ping = time.time()

            proc.stdout.close()
            exit_code = proc.wait()

            if exit_code != 0:
                if r is not None:
                    r.setex(
                        redis_status_key,
                        REDIS_TTL_SECONDS,
                        json.dumps(
                            {
                                "status": "error",
                                "message": "Inference failed",
                                "exit_code": exit_code,
                                "job_id": job_id,
                                "song_name": song_name,
                                "audio_url": audio_url,
                            }
                        ),
                    )
                yield _sse(
                    "end",
                    {"status": "error", "message": "Inference failed", "exit_code": exit_code, "audio_url": audio_url},
                )
                return

            # Default naming from snapbeat_inference.py:
            # <audio_stem>_snapbeat.json in output_path.
            snapbeat_json_path = output_path / f"{audio_path.stem}_snapbeat.json"
            if not snapbeat_json_path.exists():
                if r is not None:
                    r.setex(
                        redis_status_key,
                        REDIS_TTL_SECONDS,
                        json.dumps(
                            {
                                "status": "error",
                                "message": "Missing snapbeat json output",
                                "path": str(snapbeat_json_path),
                                "job_id": job_id,
                                "song_name": song_name,
                                "audio_url": audio_url,
                            }
                        ),
                    )
                yield _sse(
                    "end",
                    {
                        "status": "error",
                        "message": "Missing snapbeat json output",
                        "path": str(snapbeat_json_path),
                        "audio_url": audio_url,
                    },
                )
                return

            with open(snapbeat_json_path, "r", encoding="utf-8") as f:
                snapbeat_obj = json.load(f)

            if r is not None:
                envelope = {"audio_url": audio_url, "results": snapbeat_obj}
                payload = json.dumps(envelope, ensure_ascii=False)
                r.setex(redis_results_key, REDIS_TTL_SECONDS, payload)
                r.setex(
                    redis_status_key,
                    REDIS_TTL_SECONDS,
                    json.dumps({"status": "success", "job_id": job_id, "song_name": song_name, "audio_url": audio_url}),
                )

            yield _sse("end", {"status": "success", "results": snapbeat_obj, "audio_url": audio_url})
        except requests.RequestException as e:
            yield _sse(
                "end",
                {"status": "error", "message": "Download failed", "detail": str(e), "audio_url": audio_url},
            )
        except Exception as e:
            yield _sse(
                "end",
                {"status": "error", "message": "Unhandled error", "detail": str(e), "audio_url": audio_url},
            )
        finally:
            # Always cleanup any disk artifacts for this audio.
            try:
                shutil.rmtree(TMP_ROOT / "__jobs__" / f"{song_name}__{job_id}", ignore_errors=True)
            except Exception:
                pass
            if hashcode:
                try:
                    shutil.rmtree(TMP_ROOT / hashcode, ignore_errors=True)
                except Exception:
                    pass

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    # Bind to all interfaces for containerized deployment.
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)

