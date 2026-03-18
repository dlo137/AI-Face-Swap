"""
storage.py — Supabase Storage upload

Public API:
  upload_to_supabase(file_path, supabase_url, supabase_key, bucket) → str (public URL)
  upload_video(file_path)                                            → str (reads env vars)
"""
from __future__ import annotations

import os
import uuid


def upload_to_supabase(
    file_path: str,
    supabase_url: str,
    supabase_key: str,
    bucket: str,
) -> str:
    """
    Upload a file to Supabase Storage and return its public URL.

    The bucket must already exist and have public read access enabled
    (Supabase dashboard → Storage → bucket → Make Public).

    Args:
        file_path:    Absolute path to the local file to upload.
        supabase_url: Project URL, e.g. https://xxxx.supabase.co
        supabase_key: service_role secret key (NOT the anon key —
                      anon key lacks storage write permissions).
        bucket:       Target bucket name, e.g. "faceswap-uploads".

    Returns:
        Public HTTPS URL for the uploaded file.

    Raises:
        FileNotFoundError: if file_path does not exist.
        StorageException:  if the upload is rejected by Supabase.
    """
    from supabase import create_client

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File to upload not found: {file_path}")

    client = create_client(supabase_url, supabase_key)

    # Unique object key so concurrent jobs never collide
    ext = os.path.splitext(file_path)[1] or ".mp4"
    object_path = f"outputs/{uuid.uuid4()}{ext}"

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"[storage] uploading {file_size_mb:.1f} MB → {bucket}/{object_path}")

    with open(file_path, "rb") as f:
        client.storage.from_(bucket).upload(
            path=object_path,
            file=f,
            file_options={"content-type": _content_type(ext)},
        )

    public_url = client.storage.from_(bucket).get_public_url(object_path)
    print(f"[storage] public URL: {public_url}")
    return public_url


def upload_video(file_path: str) -> str:
    """
    Convenience wrapper that reads credentials from environment variables.
    Called by main.py inside the Modal function where the Secret is injected.

    Required env vars (set via Modal Secret 'supabase-secret'):
      SUPABASE_URL    — https://xxxx.supabase.co
      SUPABASE_KEY    — service_role key
      SUPABASE_BUCKET — bucket name (default: faceswap-uploads)
    """
    url    = _require_env("SUPABASE_URL")
    key    = _require_env("SUPABASE_KEY")
    bucket = os.environ.get("SUPABASE_BUCKET", "faceswap-uploads")

    return upload_to_supabase(file_path, url, key, bucket)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _content_type(ext: str) -> str:
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }.get(ext.lower(), "application/octet-stream")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Missing required env var: {name}. "
            "Add it to the Modal Secret 'supabase-secret'."
        )
    return value
