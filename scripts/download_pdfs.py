"""Download PDFs listed in data.json.

data.json structure:
    { "<Category>": { "<Title>": "<pdf url>", ... }, ... }

For each top-level key (category) an output folder is created under OUTPUT_DIR,
and every PDF in that category is downloaded into it.
"""

import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

DATA_FILE = Path("data/data.json")
OUTPUT_DIR = Path("data/pdfs")


def safe_name(name: str) -> str:
    """Make a string safe to use as a file/folder name."""
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name[:200]


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return True
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  saved: {dest.name} ({len(data) // 1024} KB)")
        return True
    except (URLError, HTTPError) as e:
        print(f"  FAILED: {dest.name} -> {e}")
        return False


def main() -> int:
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(exist_ok=True)

    ok = failed = 0
    for category, items in data.items():
        folder = OUTPUT_DIR / safe_name(category)
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n[{category}] -> {folder}")
        for title, url in items.items():
            filename = safe_name(title) + ".pdf"
            if download(url, folder / filename):
                ok += 1
            else:
                failed += 1

    print(f"\nDone. {ok} downloaded/existing, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
