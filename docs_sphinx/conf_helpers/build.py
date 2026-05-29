"""Import-time orchestration: populate the shared indexes.

Importing this module runs the one-time build steps in their original order:
stage the bibliography page, scan tutorial / contrib / standalone anchors,
generate the API stub tree, run external scans, then build the image and
snippet basename indexes. All results land in the shared maps owned by
``state``. conf.py imports this module purely for its import-time effect.
"""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *
from .xml_render import _patch_namespace_xml_for_breathe
from .stubs import _generate_api_stubs

# Stage the bibliography page into the Sphinx srcdir.
# Stage the bibliography into the Sphinx srcdir so `@subpage citelist`
# below resolves to an internal docname. Skipped when SPHINX_INPUT_ROOT
# is DOC_ROOT (ad-hoc sphinx-build) — writing into opencv/doc/ is
# forbidden, and `@cite` falls back to the external Doxygen URL in that
# case (see _cite_repl).
if _BIB_ENTRIES_SORTED and SPHINX_INPUT_ROOT != DOC_ROOT:
    try:
        SPHINX_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (SPHINX_INPUT_ROOT / "citelist.markdown").write_text(
            _bib_render_all(_BIB_ENTRIES_SORTED, _CITE_NUMBER),
            encoding="utf-8")
    except OSError:
        pass

# Internal scan: master + enabled main/js/py/contrib subtrees + standalone pages.
_scan_internal(SPHINX_INPUT_ROOT / "tutorials" / "tutorials.markdown")
for _m in DOC_MODULES:
    _scan_internal(SPHINX_INPUT_ROOT / "tutorials" / _m)
if JS_DOC_MODULES:
    _scan_internal(DOC_ROOT / "js_tutorials" / "js_tutorials.markdown",
                   base=DOC_ROOT)
for _m in JS_DOC_MODULES:
    _scan_internal(DOC_ROOT / "js_tutorials" / _m, base=DOC_ROOT)
if PY_DOC_MODULES:
    _scan_internal(DOC_ROOT / "py_tutorials" / "py_tutorials.markdown",
                   base=DOC_ROOT)
for _m in PY_DOC_MODULES:
    _scan_internal(DOC_ROOT / "py_tutorials" / _m, base=DOC_ROOT)
_contrib_root_md = SPHINX_INPUT_ROOT / "tutorials_contrib" / "tutorials_contrib.markdown"
if _contrib_root_md.is_file():
    _scan_internal(_contrib_root_md)
for _m in CONTRIB_MODULES:
    _scan_internal(SPHINX_INPUT_ROOT / "tutorials_contrib" / _m)
# Standalone top-level pages (siblings of tutorials/ in the staged tree).
# Registers their {#anchor} in _ANCHOR_TO_DOC so the master-doc @subpage
# injection below resolves to an internal docname instead of being dropped.
_scan_internal(SPHINX_INPUT_ROOT / "faq.markdown")
_scan_internal(SPHINX_INPUT_ROOT / "citelist.markdown")
_scan_internal(SPHINX_INPUT_ROOT / "intro.markdown")

if API_MODULES:
    # 1) Build a patched XML tree breathe will read (inlines group-only
    #    <memberdef>s into namespace XML so name lookups succeed).
    if _API_XML_DIR.is_dir():
        _patch_namespace_xml_for_breathe(_API_XML_DIR, _PATCHED_XML_DIR)
    # 2) Generate the api/ stub tree from the ORIGINAL XML — the stub
    #    generator only reads group XML, which is unchanged.
    _generate_api_stubs(API_MODULES, _API_XML_DIR, SPHINX_INPUT_ROOT / "api")
    # Recursive scan picks up api_root.markdown + every group stub.
    _scan_internal(SPHINX_INPUT_ROOT / "api")


def _write_root_index() -> None:
    """Generate the Sphinx landing page at ``index.html``.

    The legacy tutorials root remains focused on C++ tutorials. Cross-family
    entry points live here so the site root no longer redirects users straight
    to ``tutorials/tutorials.html``.

    Each entry renders as a section heading (the category) with the page link
    on the line beneath it. FAQ and Bibliography are direct links whose heading
    *is* the link. A hidden toctree mirrors the same order to drive the sidebar.
    """
    if SPHINX_INPUT_ROOT == DOC_ROOT:
        return

    # (heading, link_text, docname). link_text=None => the heading itself is
    # the link (FAQ / Bibliography). This order is both the rendered order and
    # the hidden-toctree order.
    entries: list[tuple[str, str | None, str]] = []

    def add(heading: str, link_text: str | None, docname: str,
            condition: bool = True) -> None:
        if condition:
            entries.append((heading, link_text, docname))

    add("Introduction", "Introduction", "intro", "intro" in _ANCHOR_TO_DOC)
    add("OpenCV Tutorials", "OpenCV tutorials", "tutorials/tutorials")
    add("Python Tutorials", "OpenCV-Python tutorials",
        "py_tutorials/py_tutorials", bool(PY_DOC_MODULES))
    add("Javascript Tutorials", "OpenCV.js tutorials",
        "js_tutorials/js_tutorials", bool(JS_DOC_MODULES))
    add("Contrib Tutorials", "tutorials for contrib module",
        "tutorials_contrib/tutorials_contrib",
        bool(CONTRIB_MODULES) and _contrib_root_md.is_file())
    add("Main modules", "main modules", "api/api_root",
        bool(API_MODULES) and "api_root" in _ANCHOR_TO_DOC)
    add("Frequently Asked Questions", None, "faq", "faq" in _ANCHOR_TO_DOC)
    add("Bibliography", None, "citelist", "citelist" in _ANCHOR_TO_DOC)

    toctree = "\n".join(
        f"{heading} <{docname}>" for heading, _link, docname in entries)

    # Body is raw HTML, NOT markdown links. A markdown `[x](intro.html)` is
    # resolved by MyST as an internal cross-reference and emitted as
    # `href="#intro.html"` (i.e. index.html#intro.html), which never
    # navigates. A raw `<a href>` is passed through verbatim and resolves
    # relative to index.html → the correct page. Raw `<h2>` headings (rather
    # than `##`) also keep these entries out of the page-local TOC, so the
    # "On this page" secondary sidebar stays empty here (see conf.py, where
    # the index page's secondary_sidebar_items is emptied). No blank lines
    # inside the block so MyST treats it as one passthrough HTML block.
    html_lines = ['<div class="ocv-landing">']
    for heading, link_text, docname in entries:
        if link_text is None:
            html_lines.append(
                f'<h2><a href="{docname}.html">{heading}</a></h2>')
        else:
            html_lines.append(f'<h2>{heading}</h2>')
            html_lines.append(f'<p><a href="{docname}.html">{link_text}</a></p>')
    html_lines.append("</div>")
    body = "\n".join(html_lines)

    text = (
        "OpenCV modules\n"
        "==============\n\n"
        "```{toctree}\n"
        ":hidden:\n"
        ":maxdepth: 1\n"
        ":titlesonly:\n\n"
        f"{toctree}\n"
        "```\n\n"
        f"{body}\n"
    )
    try:
        (SPHINX_INPUT_ROOT / "index.markdown").write_text(text, encoding="utf-8")
    except OSError:
        pass


_write_root_index()

# External scan: every OTHER main module's top-level table_of_content_*.markdown.
# Sources live under DOC_ROOT (the staged tree only contains *enabled* main
# modules, not the rest), so scan DOC_ROOT directly here.
for _toc in (DOC_ROOT / "tutorials").glob("*/table_of_content_*.markdown"):
    if _toc.parent.name not in DOC_MODULES:
        _scan_external(_toc)
# Same for js_tutorials (files are named js_table_of_contents_*.markdown there).
for _toc in (DOC_ROOT / "js_tutorials").glob("*/js_table_of_contents_*.markdown"):
    if _toc.parent.name not in JS_DOC_MODULES:
        _scan_external(_toc)
# py_tutorials uses the `py_table_of_contents_*.markdown` naming variant.
for _toc in (DOC_ROOT / "py_tutorials").glob("*/py_table_of_contents_*.markdown"):
    if _toc.parent.name not in PY_DOC_MODULES:
        _scan_external(_toc)

# Image basename index (mirrors Doxygen's flat IMAGE_PATH lookup).
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp"}
for _root in ((DOC_ROOT / "tutorials").rglob("images/*"),
              (DOC_ROOT / "js_tutorials").rglob("images/*"),
              (DOC_ROOT / "js_tutorials" / "js_assets").glob("*"),
              (DOC_ROOT / "py_tutorials").rglob("images/*"),
              (DOC_ROOT / "images").glob("*")):
    for _img in _root:
        if _img.is_file():
            _IMAGE_INDEX.setdefault(_img.name, _img.relative_to(DOC_ROOT).as_posix())
for _m in CONTRIB_MODULES:
    # <m>/tutorials/**/images/* — same shape as main, reachable through
    # the existing tutorials_contrib/<m> symlink CMake stages.
    _tut = CONTRIB_ROOT / _m / "tutorials"
    if _tut.is_dir():
        for _img in _tut.rglob("images/*"):
            if _img.is_file():
                _rel = _img.relative_to(_tut).as_posix()
                _IMAGE_INDEX.setdefault(_img.name,
                                        f"tutorials_contrib/{_m}/{_rel}")
    # Contrib images outside <m>/tutorials/ (<m>/doc/pics, <m>/samples).
    # URL is /contrib_modules/<m>/<rest>. Files are served from there via
    # html_extra_path set below — no copies in srcdir.
    for _sub in ("doc", "samples"):
        _src = CONTRIB_ROOT / _m / _sub
        if _src.is_dir():
            for _img in _src.rglob("*"):
                if _img.is_file() and _img.suffix.lower() in _IMAGE_EXTS:
                    _rel = _img.relative_to(CONTRIB_ROOT).as_posix()
                    _IMAGE_INDEX.setdefault(_img.name,
                                            f"contrib_modules/{_rel}")

# Snippet basename index (mirrors Doxygen EXAMPLE_RECURSIVE lookup).
_SNIPPET_EXTENSIONS = {
    ".cpp", ".hpp", ".h", ".c", ".cc", ".cxx",
    ".py", ".java", ".kt", ".scala", ".clj", ".groovy",
    ".sh", ".bash", ".bat", ".ps1",
    ".cmake", ".gradle",
    ".xml", ".yaml", ".yml", ".json", ".html", ".css",
    ".js", ".ts", ".rb",
}
_snippet_scan_roots = [OPENCV_ROOT / "samples", OPENCV_ROOT / "apps"] + [
    CONTRIB_ROOT / _m / "samples" for _m in CONTRIB_MODULES]
for _root in _snippet_scan_roots:
    if _root.is_dir():
        for _f in _root.rglob("*"):
            if _f.is_file() and _f.suffix.lower() in _SNIPPET_EXTENSIONS:
                _SNIPPET_INDEX.setdefault(_f.name, _f)
