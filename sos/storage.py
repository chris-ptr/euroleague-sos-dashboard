"""
Minimal Supabase Storage client using plain HTTP (via `requests`) instead of
the full supabase-py SDK — all we need is upload/download/list of objects.

Used only by scripts/refresh_and_publish.py and scripts/seed_supabase.py
(server-side, with the service-role key). Never imported by the frontend.
"""
import json
from typing import Optional

import requests

from .config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY


class SupabaseStorage:
    def __init__(self, base_url: str = SUPABASE_URL, service_role_key: str = SUPABASE_SERVICE_ROLE_KEY):
        if not base_url or not service_role_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set (env vars or .env)."
            )
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {service_role_key}",
            "apikey": service_role_key,
        }

    def upload_bytes(self, bucket: str, path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        url = f"{self.base_url}/storage/v1/object/{bucket}/{path}"
        headers = {
            **self.headers,
            "Content-Type": content_type,
            "x-upsert": "true",
        }
        resp = requests.post(url, headers=headers, data=data, timeout=60)
        if resp.status_code >= 300:
            raise RuntimeError(f"Upload failed for {bucket}/{path}: {resp.status_code} {resp.text}")

    def upload_json(self, bucket: str, path: str, obj) -> None:
        self.upload_bytes(bucket, path, json.dumps(obj).encode("utf-8"), content_type="application/json")

    def download_bytes(self, bucket: str, path: str) -> Optional[bytes]:
        url = f"{self.base_url}/storage/v1/object/{bucket}/{path}"
        resp = requests.get(url, headers=self.headers, timeout=60)
        if resp.status_code == 404:
            return None
        if resp.status_code >= 300:
            raise RuntimeError(f"Download failed for {bucket}/{path}: {resp.status_code} {resp.text}")
        return resp.content

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """Return object names (not full paths) directly under `prefix`."""
        url = f"{self.base_url}/storage/v1/object/list/{bucket}"
        resp = requests.post(
            url,
            headers={**self.headers, "Content-Type": "application/json"},
            json={"prefix": prefix, "limit": 1000, "offset": 0},
            timeout=60,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"List failed for {bucket}/{prefix}: {resp.status_code} {resp.text}")
        return [item["name"] for item in resp.json() if item.get("id") is not None]
