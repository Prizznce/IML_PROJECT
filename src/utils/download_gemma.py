import sys
import os
import urllib.request
import time
from pathlib import Path
from src.utils.model_registry import get_hf_token

def download_file():
    token = get_hf_token()
    snap_dir = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub" / "models--google--gemma-3-1b-it" / "snapshots" / "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_file = snap_dir / "model.safetensors"
    part_file = snap_dir / "model.safetensors.part"

    if out_file.exists() and out_file.stat().st_size == 1999811208:
        print(f"model.safetensors already completely downloaded at {out_file}", flush=True)
        return

    url = "https://huggingface.co/google/gemma-3-1b-it/resolve/main/model.safetensors"
    headers = {"Authorization": f"Bearer {token}"}

    existing_bytes = 0
    if part_file.exists():
        existing_bytes = part_file.stat().st_size
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"Resuming download from byte {existing_bytes}...", flush=True)
    else:
        print("Starting fresh download of model.safetensors (1.86 GB)...", flush=True)

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(part_file, "ab" if existing_bytes > 0 else "wb") as f:
        total_len = int(resp.headers.get("Content-Length", 0)) + existing_bytes
        downloaded = existing_bytes
        chunk_size = 4 * 1024 * 1024  # 4 MB
        last_log = time.time()
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_log > 3:
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_len / (1024 * 1024) if total_len else 0
                pct = (downloaded / total_len * 100) if total_len else 0
                print(f"Downloaded {mb_down:.1f} MB / {mb_tot:.1f} MB ({pct:.1f}%)", flush=True)
                last_log = time.time()

    if part_file.stat().st_size == 1999811208:
        part_file.replace(out_file)
        print(f"Download verified and saved to {out_file}", flush=True)
    else:
        print(f"Warning: downloaded size {part_file.stat().st_size} differs from expected 1999811208", flush=True)

if __name__ == "__main__":
    download_file()
