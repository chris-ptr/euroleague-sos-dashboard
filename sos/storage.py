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

    def _list_entries(self, bucket: str, prefix: str) -> list[dict]:
        """Raw listing directly under `prefix`, paginated. Folders have id=None."""
        url = f"{self.base_url}/storage/v1/object/list/{bucket}"
        entries: list[dict] = []
        offset, page = 0, 1000
        while True:
            resp = requests.post(
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                json={"prefix": prefix, "limit": page, "offset": offset},
                timeout=60,
            )
            if resp.status_code >= 300:
                raise RuntimeError(f"List failed for {bucket}/{prefix}: {resp.status_code} {resp.text}")
            batch = resp.json()
            entries.extend(batch)
            if len(batch) < page:
                return entries
            offset += page

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """Return object names (not full paths) directly under `prefix`."""
        return [e["name"] for e in self._list_entries(bucket, prefix) if e.get("id") is not None]

    def list_objects_recursive(self, bucket: str, prefix: str = "") -> list[str]:
        """Return full object paths under `prefix`, descending into subfolders."""
        paths: list[str] = []
        for entry in self._list_entries(bucket, prefix):
            child = f"{prefix}/{entry['name']}" if prefix else entry["name"]
            if entry.get("id") is None:  # folder placeholder, not a real object
                paths.extend(self.list_objects_recursive(bucket, child))
            else:
                paths.append(child)
        return paths

    def delete_objects(self, bucket: str, paths: list[str], batch_size: int = 100) -> int:
        """Delete objects by full path. Returns the number deleted."""
        url = f"{self.base_url}/storage/v1/object/{bucket}"
        deleted = 0
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            resp = requests.delete(
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                json={"prefixes": chunk},
                timeout=120,
            )
            if resp.status_code >= 300:
                raise RuntimeError(f"Delete failed for {bucket}: {resp.status_code} {resp.text}")
            deleted += len(chunk)
        return deleted
