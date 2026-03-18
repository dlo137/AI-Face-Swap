"""
storage.py — Supabase Storage upload
Uploads the output video and returns its public URL.

Required env vars (set via Modal Secret "supabase-secret"):
  SUPABASE_URL      — e.g. https://xxxx.supabase.co
  SUPABASE_KEY      — service_role key (NOT anon — needs storage write access)
  SUPABASE_BUCKET   — e.g. "face-swap-results"
"""
import os
import uuid
from pathlib import Path


def upload_video(local_path: str) -> str:
    """
    Upload a video file to Supabase Storage.

    Args:
        local_path: Absolute path to the MP4 file on disk.

    Returns:
        Public URL string for the uploaded file.
    """
    from supabase import create_client, Client

    url = _require_env("SUPABASE_URL")
    key = _require_env("SUPABASE_KEY")
    bucket = os.environ.get("SUPABASE_BUCKET", "face-swap-results")

    client: Client = create_client(url, key)

    # Generate a unique object key
    file_name = f"{uuid.uuid4()}.mp4"
    object_path = f"outputs/{file_name}"

    with open(local_path, "rb") as f:
        video_bytes = f.read()

    print(f"[storage] uploading {len(video_bytes) / (1024*1024):.1f} MB → {bucket}/{object_path}")

    client.storage.from_(bucket).upload(
        path=object_path,
        file=video_bytes,
        file_options={"content-type": "video/mp4"},
    )

    public_url = client.storage.from_(bucket).get_public_url(object_path)
    print(f"[storage] public URL: {public_url}")
    return public_url


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}. "
            f"Add it to the Modal Secret 'supabase-secret'."
        )
    return value
