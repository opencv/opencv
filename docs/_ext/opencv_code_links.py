"""Build a symbol→URL map from the Doxygen tagfile and write it to
``_static/opencv-symbols.json`` so client-side JS can link identifiers
inside code blocks back to docs.opencv.org."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

DOCS_BASE = "https://docs.opencv.org/5.x/"

SKIP_NAMES = {
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "return", "break", "continue", "void", "int", "char", "float", "double",
    "bool", "true", "false", "this", "new", "delete", "throw", "try", "catch",
    "namespace", "class", "struct", "enum", "typedef", "using", "template",
    "typename", "public", "private", "protected", "virtual", "const",
    "static", "extern", "inline", "auto", "size_t", "include",
    "main", "argc", "argv", "vector", "string", "map", "pair",
    "ptr", "cap", "frame", "edges", "img", "mtx", "cmtx", "grayscale",
    "image", "src", "dst", "tmp", "buf", "data", "mat",
    "i", "j", "k", "n", "m", "x", "y", "z", "w", "h",
}


def _load_symbols(tag_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not tag_path.exists():
        return out
    root = ET.parse(tag_path).getroot()
    for compound in root.iter("compound"):
        kind = compound.get("kind")
        cname = (compound.findtext("name") or "").strip()
        cfile = (compound.findtext("filename") or "").strip()
        if not (cname and cfile):
            continue
        url = DOCS_BASE + cfile
        if kind in ("class", "struct"):
            simple = cname.split("::")[-1]
            if len(simple) > 1 and simple not in SKIP_NAMES:
                out.setdefault(simple, url)
                out.setdefault(cname, url)
        if kind == "file":
            cpath = (compound.findtext("path") or "").strip()
            full_path = cpath + cname if cpath else cname
            if full_path:
                out.setdefault(full_path, url)
                out.setdefault('"' + full_path + '"', url)
                out.setdefault("<" + full_path + ">", url)

        for m in compound.findall("member"):
            mname = (m.findtext("name") or "").strip()
            mfile = (m.findtext("anchorfile") or "").strip()
            manchor = (m.findtext("anchor") or "").strip()
            if not (mname and mfile and manchor):
                continue
            if len(mname) <= 1 or mname in SKIP_NAMES:
                continue
            mkind = m.get("kind") or ""
            if mkind in ("variable", "typedef") and mname.islower():
                continue
            full_url = DOCS_BASE + mfile + "#" + manchor
            out.setdefault(mname, full_url)
            if kind in ("class", "struct") and cname:
                out.setdefault(f"{cname}::{mname}", full_url)
    return out


def _on_build_finished(app, exception):
    if exception is not None:
        return
    tag = Path(app.config.opencv_tagfile)
    symbols = _load_symbols(tag)
    payload = json.dumps(symbols, separators=(",", ":"))
    out_static = Path(app.outdir) / "_static"
    out_static.mkdir(parents=True, exist_ok=True)
    (out_static / "opencv-symbols.json").write_text(payload, encoding="utf-8")


def setup(app):
    app.add_config_value("opencv_tagfile", "/tmp/opencv.tag", "html")
    app.connect("build-finished", _on_build_finished)
    return {"version": "0.1", "parallel_read_safe": True}
