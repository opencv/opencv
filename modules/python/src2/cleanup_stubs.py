import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1])

for p in root.rglob("*.pyi"):
    d = p.with_suffix("")
    if d.is_dir():
        shutil.rmtree(d)
