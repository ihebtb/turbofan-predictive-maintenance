"""Download and extract NASA C-MAPSS into data/raw/CMAPSSData (about 12 MB)."""

import io
import urllib.request
import zipfile

from turbofan.data import DATASET_URL, DEFAULT_ROOT


def main() -> None:
    if (DEFAULT_ROOT / "train_FD001.txt").exists():
        print(f"already present: {DEFAULT_ROOT}")
        return
    print(f"downloading {DATASET_URL}")
    outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(DATASET_URL, timeout=300).read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)
    inner.extractall(DEFAULT_ROOT)
    print(f"extracted to {DEFAULT_ROOT}: {sorted(p.name for p in DEFAULT_ROOT.iterdir())}")


if __name__ == "__main__":
    main()
