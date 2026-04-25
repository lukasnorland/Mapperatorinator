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


def _redis_client() -> redis.Redis | None:
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
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
    with requests.get(audio_url, stream=True, timeout=(10, 300)) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


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
        r = _redis_client()
        hashcode: str | None = None
        try:
            yield _sse("status", {"stage": "init", "job_id": job_id, "song_name": song_name})
            yield _sse("status", {"stage": "download", "audio_url": audio_url})

            # Download into a per-job folder first so we can hash the bytes.
            job_dir = TMP_ROOT / "__jobs__" / f"{song_name}__{job_id}"
            audio_path = job_dir / filename
            output_path = job_dir

            _download_audio(audio_url, audio_path)
            yield _sse("status", {"stage": "download_done", "audio_path": str(audio_path)})

            hashcode_full = _file_sha256(audio_path)
            hashcode = hashcode_full[:16]
            yield _sse("status", {"stage": "hash_computed", "hashcode": hashcode})

            redis_base_key = f"skeleton_data:{song_name}_{hashcode}"
            redis_results_key = f"{redis_base_key}:results"
            redis_status_key = f"{redis_base_key}:status"
            if r is not None:
                cached = r.get(redis_results_key)
                if cached:
                    try:
                        cached_obj = json.loads(cached)
                        yield _sse(
                            "status",
                            {"stage": "cache_hit", "hashcode": hashcode, "key": redis_results_key},
                        )
                        yield _sse("end", {"status": "success", "results": cached_obj})
                        return
                    except Exception:
                        # If cache is corrupted, ignore and recompute.
                        pass
                r.setex(
                    redis_status_key,
                    REDIS_TTL_SECONDS,
                    json.dumps({"status": "running", "stage": "download_done", "job_id": job_id, "song_name": song_name}),
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
                                {"status": "running", "stage": "inference_running", "job_id": job_id, "song_name": song_name}
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
                            }
                        ),
                    )
                yield _sse("end", {"status": "error", "message": "Inference failed", "exit_code": exit_code})
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
                            }
                        ),
                    )
                yield _sse(
                    "end",
                    {
                        "status": "error",
                        "message": "Missing snapbeat json output",
                        "path": str(snapbeat_json_path),
                    },
                )
                return

            with open(snapbeat_json_path, "r", encoding="utf-8") as f:
                snapbeat_obj = json.load(f)

            if r is not None:
                r.setex(redis_results_key, REDIS_TTL_SECONDS, json.dumps(snapbeat_obj, ensure_ascii=False))
                r.setex(
                    redis_status_key,
                    REDIS_TTL_SECONDS,
                    json.dumps({"status": "success", "job_id": job_id, "song_name": song_name}),
                )

            yield _sse("end", {"status": "success", "results": snapbeat_obj})
        except requests.RequestException as e:
            if r is not None:
                r.setex(
                    f"{song_name}__{job_id}:status",
                    REDIS_TTL_SECONDS,
                    json.dumps({"status": "error", "message": "Download failed", "detail": str(e), "job_id": job_id}),
                )
            yield _sse("end", {"status": "error", "message": "Download failed", "detail": str(e)})
        except Exception as e:
            yield _sse("end", {"status": "error", "message": "Unhandled error", "detail": str(e)})
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
    app.run(host="127.0.0.1", port=5050, threaded=True, debug=False)

