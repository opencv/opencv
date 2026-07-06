# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""Import-time orchestration: populate the shared indexes."""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *
from .xml_render import _patch_namespace_xml_for_breathe
from .stubs import _generate_api_stubs


def _discover_orphan_groups(xml_dir):
    if not xml_dir.is_dir():
        return [], []
    folders = set()
    for _root in (OPENCV_ROOT / "modules", CONTRIB_ROOT):
        if _root.is_dir():
            folders.update(d.name for d in _root.iterdir()
                           if (d / "include" / "opencv2").is_dir())
    skip = set(folders) | {f.replace("_", "__") for f in folders}
    skip |= {_module_group_stem(m) for m in folders}
    all_groups, child = set(), set()
    for gx in xml_dir.glob("group__*.xml"):
        all_groups.add(gx.stem)
        try:
            xml = gx.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        child.update(re.findall(r'<innergroup refid="(group__[^"]+)"', xml))
    main, extra = [], []
    for g in sorted(all_groups - child):
        stem = g[len("group__"):]
        name = stem.replace("__", "_")
        if stem in skip or name in skip:
            continue
        xml = (xml_dir / f"{g}.xml").read_text(encoding="utf-8", errors="ignore")
        loc = re.search(r'<location file="([^"]*)"', xml)
        (extra if loc and "opencv_contrib/" in loc.group(1) else main).append(name)
    return main, extra


# Skip when input root is DOC_ROOT: writing there is forbidden.
if _BIB_ENTRIES_SORTED and SPHINX_INPUT_ROOT != DOC_ROOT:
    try:
        SPHINX_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (SPHINX_INPUT_ROOT / "citelist.markdown").write_text(
            _bib_render_all(_BIB_ENTRIES_SORTED, _CITE_NUMBER),
            encoding="utf-8")
    except OSError:
        pass

# Internal scan: enabled subtrees + standalone pages.
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

_contrib_dir = SPHINX_INPUT_ROOT / "tutorials_contrib"
_contrib_root_md = next(
    (p for p in (_contrib_dir / "contrib_root.markdown",
                 _contrib_dir / "tutorials_contrib.markdown") if p.is_file()),
    _contrib_dir / "contrib_root.markdown")
if _contrib_root_md.is_file():
    _scan_internal(_contrib_root_md)
for _m in CONTRIB_MODULES:
    _scan_internal(SPHINX_INPUT_ROOT / "tutorials_contrib" / _m)
# Standalone top-level pages.
_scan_internal(SPHINX_INPUT_ROOT / "faq.markdown")
_scan_internal(SPHINX_INPUT_ROOT / "citelist.markdown")
_scan_internal(SPHINX_INPUT_ROOT / "intro.markdown")

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
    # <m>/tutorials/**/images/*
    _tut = CONTRIB_ROOT / _m / "tutorials"
    if _tut.is_dir():
        for _img in _tut.rglob("images/*"):
            if _img.is_file():
                _rel = _img.relative_to(_tut).as_posix()
                _IMAGE_INDEX.setdefault(_img.name,
                                        f"tutorials_contrib/{_m}/{_rel}")
    # Contrib images outside <m>/tutorials/.
    for _sub in ("doc", "samples"):
        _src = CONTRIB_ROOT / _m / _sub
        if _src.is_dir():
            for _img in _src.rglob("*"):
                if _img.is_file() and _img.suffix.lower() in _IMAGE_EXTS:
                    _rel = _img.relative_to(CONTRIB_ROOT).as_posix()
                    _IMAGE_INDEX.setdefault(_img.name,
                                            f"contrib_modules/{_rel}")

# Tutorials can reference images that live only under samples/ (+ apps/), which
# Doxygen's IMAGE_PATH covered but the tutorial-only index misses (renders
# broken). Index+stage ONLY sample images a tutorial actually references and
# that a tutorial `images/` dir doesn't already provide, to avoid copying the
# whole samples image set. Staged under sample_pics/ like api_pics above.
_referenced_images: set[str] = set()
for _md in (DOC_ROOT / "tutorials").rglob("*.markdown"):
    try:
        _txt = _md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    for _m in re.finditer(r'!\[[^\]]*\]\((?:[^)\s]*?/)?images/([^)\s]+)\)', _txt):
        _referenced_images.add(pathlib.Path(_m.group(1)).name)
_missing_images = {n for n in _referenced_images if n not in _IMAGE_INDEX}
if _missing_images:
    _sample_pics = SPHINX_INPUT_ROOT / "sample_pics"
    _stage_samples = SPHINX_INPUT_ROOT != DOC_ROOT
    for _base in (OPENCV_ROOT / "samples", OPENCV_ROOT / "apps"):
        if not _base.is_dir() or not _missing_images:
            continue
        for _img in _base.rglob("*"):
            if (_img.name in _missing_images and _img.is_file()
                    and _img.suffix.lower() in _IMAGE_EXTS):
                _IMAGE_INDEX[_img.name] = f"sample_pics/{_img.name}"
                _missing_images.discard(_img.name)   # first match wins; stop looking
                if _stage_samples:
                    _sample_pics.mkdir(parents=True, exist_ok=True)
                    _link = _sample_pics / _img.name
                    if not _link.exists():
                        try:
                            _os.symlink(_img, _link)
                        except (OSError, NotImplementedError):
                            try:
                                _shutil.copy2(_img, _link)
                            except OSError:
                                pass
            if not _missing_images:
                break


# Hand-authored dnn topic; content lives in the dnn module's own doc dir.
_DNN_ENGINE_SELECTION_MD = (
    OPENCV_ROOT / "modules" / "dnn" / "doc" / "dnn_engine.markdown").read_text(
    encoding="utf-8")


def _add_dnn_engine_selection_topic(out_dir: pathlib.Path) -> None:
    """Add the 'DNN Engine Selection' page as a dnn-module topic.
    Must run after stub generation (its stale-file sweep) and before the anchor
    scan, so the new {#anchor} + @subpage get picked up."""
    dnn = out_dir / "dnn.md"
    if not dnn.is_file():
        return
    (out_dir / "dnn_engine_selection.md").write_text(
        _DNN_ENGINE_SELECTION_MD, encoding="utf-8")
    text = dnn.read_text(encoding="utf-8")
    if "api_dnn_engine_selection" in text:
        return
    new = re.sub(
        r"(## Topics\n\n(?:- @subpage [^\n]*\n)+)",
        lambda m: m.group(1) + "- @subpage api_dnn_engine_selection\n",
        text, count=1)
    if new != text:
        dnn.write_text(new, encoding="utf-8")


# Hand-authored standalone HAL page; content lives in core's own doc dir.
_HAL_MD = (
    OPENCV_ROOT / "modules" / "core" / "doc" / "hal.markdown").read_text(
    encoding="utf-8")


def _add_hal_page(out_dir: pathlib.Path) -> None:
    """Write the HAL page and add a 'Learn about HAL' link on the api_root page.
    Call after stub generation and before the anchor scan."""
    api_root = out_dir / "api_root.markdown"
    if not api_root.is_file():
        return
    (out_dir / "hal.md").write_text(_HAL_MD, encoding="utf-8")
    text = api_root.read_text(encoding="utf-8")
    if "](hal.md)" in text:
        return
    text = re.sub(r"(```\{toctree\}\n.*?\n)(```\n)", r"\1hal\n\2",
                  text, count=1, flags=re.S)
    text = text.rstrip() + (
        "\n\n## Learn about HAL\n\n"
        "OpenCV ships a Hardware Acceleration Layer that lets hardware vendors "
        "inject tuned, silicon-specific implementations behind a stable C "
        "interface. See [OpenCV Hardware Acceleration Layer (HAL)](hal.md).\n")
    api_root.write_text(text, encoding="utf-8")


if API_MODULES:
    _api_pics = SPHINX_INPUT_ROOT / "api_pics"
    _stage_pics = SPHINX_INPUT_ROOT != DOC_ROOT
    if _stage_pics:
        _api_pics.mkdir(parents=True, exist_ok=True)
    _modules_root = DOC_ROOT.parent / "modules"
    if _modules_root.is_dir():
        for _doc_dir in sorted(_modules_root.glob("*/doc")):
            for _img in _doc_dir.rglob("*"):
                if not (_img.is_file() and _img.suffix.lower() in _IMAGE_EXTS):
                    continue
                if _img.name in _IMAGE_INDEX:
                    continue
                _IMAGE_INDEX[_img.name] = f"api_pics/{_img.name}"
                if _stage_pics:
                    _link = _api_pics / _img.name
                    if not _link.exists():
                        try:
                            _os.symlink(_img, _link)
                        except (OSError, NotImplementedError):
                            try:
                                _shutil.copy2(_img, _link)
                            except OSError:
                                pass

    if _API_XML_DIR.is_dir():
        _patch_namespace_xml_for_breathe(_API_XML_DIR, _PATCHED_XML_DIR)

    from conf_helpers.state import OPENCV_ROOT, CONTRIB_ROOT
    _is_contrib = lambda m: (CONTRIB_ROOT / m).is_dir() and not (
        OPENCV_ROOT / "modules" / m).is_dir()
    _main_api = [m for m in API_MODULES if not _is_contrib(m)]
    _extra_api = [m for m in API_MODULES if _is_contrib(m)]
    _main_orphans, _extra_orphans = _discover_orphan_groups(_API_XML_DIR)
    _generate_api_stubs(_main_api, _API_XML_DIR, SPHINX_INPUT_ROOT / "main_modules",
                        root_anchor="api_root", root_title="Main modules",
                        extra_groups=_main_orphans)
    _add_dnn_engine_selection_topic(SPHINX_INPUT_ROOT / "main_modules")
    _add_hal_page(SPHINX_INPUT_ROOT / "main_modules")
    _scan_internal(SPHINX_INPUT_ROOT / "main_modules")
    if _extra_api or _extra_orphans:
        _generate_api_stubs(_extra_api, _API_XML_DIR, SPHINX_INPUT_ROOT / "extra_modules",
                            root_anchor="extra_api_root", root_title="Extra modules",
                            extra_groups=_extra_orphans)
        _scan_internal(SPHINX_INPUT_ROOT / "extra_modules")


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

    entries: list[tuple[str, str | None, str]] = []

    def add(heading: str, link_text: str | None, docname: str,
            condition: bool = True) -> None:
        if condition:
            entries.append((heading, link_text, docname))

    add("Introduction", "Introduction", "intro", "intro" in _ANCHOR_TO_DOC)
    # Its own landing section rather than a bullet in the intro's Usage list.
    _prebuilt = _ANCHOR_TO_DOC.get("tutorial_using_prebuilt_binaries")
    add("How to use pre-built OpenCV binaries",
        "Using OpenCV pre-built binaries in your own projects",
        _prebuilt or "", bool(_prebuilt))
    add("OpenCV Tutorials", "OpenCV tutorials", "tutorials/tutorials")
    add("Python Tutorials", "OpenCV-Python tutorials",
        "py_tutorials/py_tutorials", bool(PY_DOC_MODULES))
    add("Javascript Tutorials", "OpenCV.js tutorials",
        "js_tutorials/js_tutorials", bool(JS_DOC_MODULES))
    add("Contrib Tutorials", "Tutorials for contrib module",
        f"tutorials_contrib/{_contrib_root_md.stem}",
        bool(CONTRIB_MODULES) and _contrib_root_md.is_file())
    add("Main modules", "Documentation for main modules",
        "main_modules/api_root", "api_root" in _ANCHOR_TO_DOC)
    add("Extra modules", "Documentation for extra modules",
        "extra_modules/api_root", "extra_api_root" in _ANCHOR_TO_DOC)
    add("Frequently Asked Questions", None, "faq", "faq" in _ANCHOR_TO_DOC)
    add("Bibliography", None, "citelist", "citelist" in _ANCHOR_TO_DOC)

    toctree = "\n".join(
        f"{heading} <{docname}>" for heading, _link, docname in entries)

    # Markdown H2 headings so each shows in the "On this page" TOC; links stay
    # raw HTML to resolve relative to index.html (headings can't sit in a <div>).
    body_lines: list[str] = []
    for heading, link_text, docname in entries:
        body_lines.append(f"## {heading}\n")
        body_lines.append(
            f'<ul><li><a href="{docname}.html">{link_text or heading}</a></li></ul>\n')
    body = "\n".join(body_lines).rstrip()

    # Landing-page intro prose lives in a hand-editable Markdown file next to
    # conf.py (docs_sphinx/index_intro.md), not inline here.
    _intro_md = pathlib.Path(__file__).resolve().parent.parent / "index_intro.md"
    try:
        intro = _intro_md.read_text(encoding="utf-8").strip()
    except OSError:
        intro = ""

    text = (
        "OpenCV documentation\n"
        "====================\n\n"
        "```{toctree}\n"
        ":hidden:\n"
        ":maxdepth: 1\n"
        ":titlesonly:\n\n"
        f"{toctree}\n"
        "```\n\n"
        f"{intro}\n\n"
        f"{body}\n"
    )
    try:
        (SPHINX_INPUT_ROOT / "index.markdown").write_text(text, encoding="utf-8")
    except OSError:
        pass


def _write_related_pages_index() -> None:
    """Generate `related_pages.markdown` — the local analog of Doxygen's
    pages.html (the header "Related Pages" target).

    Lists every standalone documentation page (\\page) that has a *local*
    Sphinx docname, so nothing points off-site. Titles and the canonical set
    come from the Doxygen tag page index (`_DOC_PAGE_TITLES`); a page is
    emitted only when its name resolves through `_ANCHOR_TO_DOC`, so the list
    contains exactly what this build actually rendered and grows automatically
    as more modules are enabled. Marked `orphan` — reached via the header link,
    not the sidebar toctree (intro/faq/citelist already live in the index toc).
    """
    if SPHINX_INPUT_ROOT == DOC_ROOT:
        return
    rows: list[tuple[str, str]] = []        # (title, docname)
    seen: set[str] = set()

    def add(anchor: str) -> None:
        doc = _ANCHOR_TO_DOC.get(anchor)
        if doc and anchor not in seen:
            title = (_DOC_PAGE_TITLES.get(anchor)
                     or _ANCHOR_TO_TITLE.get(anchor) or anchor)
            rows.append((title, doc))
            seen.add(anchor)

    # Core standalone pages first, in a stable, friendly order.
    for _a in ("intro", "faq", "citelist"):
        add(_a)
    # Then every other \page that resolves locally, alphabetical by title.
    for _name in sorted(_DOC_PAGE_TITLES,
                        key=lambda n: (_DOC_PAGE_TITLES.get(n) or n).lower()):
        add(_name)

    items = "\n".join(f'<li><a href="{_d}.html">{_t}</a></li>' for _t, _d in rows)
    text = (
        "---\norphan: true\n---\n"
        "# Related Pages\n\n"
        "All standalone documentation pages available in this build.\n\n"
        f'<ul class="ocv-related-pages">\n{items}\n</ul>\n'
    )
    try:
        (SPHINX_INPUT_ROOT / "related_pages.markdown").write_text(
            text, encoding="utf-8")
    except OSError:
        pass


def _write_examples_index() -> None:
    """Generate `examples/examples_root.markdown` — the local analog of
    Doxygen's examples.html (the header "Examples" target).

    The per-sample example pages are orphan pages reached from class "Examples"
    blocks; this index links them all in one place. Sourced from
    `_EXAMPLE_PAGES_NEEDED` (populated during API-stub generation), so it lists
    exactly the samples this build emitted. Also `orphan` (header-only entry).
    """
    if SPHINX_INPUT_ROOT == DOC_ROOT:
        return
    from .examples import _EXAMPLE_PAGES_NEEDED, _example_pagename
    if not _EXAMPLE_PAGES_NEEDED:
        return
    items = "\n".join(
        f'<li><a href="{_example_pagename(_d)}.html">{_d}</a></li>'
        for _d in sorted(_EXAMPLE_PAGES_NEEDED))
    text = (
        "---\norphan: true\n---\n"
        "# Examples\n\n"
        "All example programs referenced in the API documentation.\n\n"
        f'<ul class="ocv-examples-index">\n{items}\n</ul>\n'
    )
    try:
        (SPHINX_INPUT_ROOT / "examples").mkdir(parents=True, exist_ok=True)
        (SPHINX_INPUT_ROOT / "examples" / "examples_root.markdown").write_text(
            text, encoding="utf-8")
    except OSError:
        pass


def _esc(s: str) -> str:
    """Minimal HTML escape for brief text injected into the index <li> markup."""
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_namespace_list_index() -> None:
    """Generate `namespace_list.markdown` — local analog of Doxygen's
    namespaces.html (the header "Namespaces" target).

    Renders the namespace tree (cv → cv::cuda → …) as a nested list, each node
    linking to its local namespace page with the brief description alongside.
    Intermediate namespaces with no page of their own render as plain text.
    Sourced from `_ALL_NAMESPACES` (populated during API-stub generation).
    """
    if SPHINX_INPUT_ROOT == DOC_ROOT or not _ALL_NAMESPACES:
        return
    # Nested tree keyed by path component; each node tracks its full name.
    tree: dict = {}
    for _name in _ALL_NAMESPACES:
        node = tree
        parts = _name.split("::")
        for _i, _part in enumerate(parts):
            node = node.setdefault(
                _part, {"_full": "::".join(parts[:_i + 1]), "_kids": {}})["_kids"]

    def render(node: dict) -> list[str]:
        out = ["<ul>"]
        for _part in sorted(node, key=str.lower):
            child = node[_part]
            info = _ALL_NAMESPACES.get(child["_full"])
            if info:
                label = f'<a href="{info["docname"]}.html">{_part}</a>'
                if info.get("brief"):
                    label += f' — {_esc(info["brief"])}'
            else:
                label = _part
            out.append(f"<li>{label}")
            if child["_kids"]:
                out += render(child["_kids"])
            out.append("</li>")
        out.append("</ul>")
        return out

    text = (
        "---\norphan: true\n---\n"
        "# Namespace List\n\n"
        "Here is a list of all documented namespaces with brief descriptions.\n\n"
        f'<div class="ocv-namespace-list">\n{chr(10).join(render(tree))}\n</div>\n'
    )
    try:
        (SPHINX_INPUT_ROOT / "namespace_list.markdown").write_text(
            text, encoding="utf-8")
    except OSError:
        pass


def _write_class_list_index() -> None:
    """Generate `class_list.markdown` — local analog of Doxygen's annotated.html
    (the header "Classes" target).

    Lists every documented class/struct grouped by its enclosing namespace,
    each linking to its local page with the brief description. Sourced from
    `_ALL_CLASSES` (populated during API-stub generation).
    """
    if SPHINX_INPUT_ROOT == DOC_ROOT or not _ALL_CLASSES:
        return
    by_ns: dict[str, list[tuple[str, dict]]] = {}
    for _info in _ALL_CLASSES.values():
        qualified = _info.get("qualified", "")
        if not qualified:
            continue
        ns, _, leaf = qualified.rpartition("::")
        by_ns.setdefault(ns, []).append((leaf, _info))

    body = ['<ul class="ocv-class-list">']
    for ns in sorted(by_ns, key=lambda n: (n == "", n.lower())):
        heading = ns if ns else "(global namespace)"
        ns_info = _ALL_NAMESPACES.get(ns)
        if ns_info:
            heading = f'<a href="{ns_info["docname"]}.html">{ns}</a>'
        body.append(f"<li><b>{heading}</b>")
        body.append("<ul>")
        for leaf, info in sorted(by_ns[ns], key=lambda t: t[0].lower()):
            entry = f'<a href="{info["docname"]}.html">{leaf}</a>'
            if info.get("brief"):
                entry += f' — {_esc(info["brief"])}'
            body.append(f"<li>{entry}</li>")
        body.append("</ul></li>")
    body.append("</ul>")

    text = (
        "---\norphan: true\n---\n"
        "# Class List\n\n"
        "Here are the classes, structs and unions with brief descriptions.\n\n"
        f'<div class="ocv-class-list-wrap">\n{chr(10).join(body)}\n</div>\n'
    )
    try:
        (SPHINX_INPUT_ROOT / "class_list.markdown").write_text(
            text, encoding="utf-8")
    except OSError:
        pass


_write_root_index()
_write_related_pages_index()
if API_MODULES:
    _write_examples_index()
    _write_namespace_list_index()
    _write_class_list_index()

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

_REFERENCED_ANCHORS.update({
    "intro", "faq", "citelist",
    "tutorial_js_root", "tutorial_py_root", "tutorial_contrib_root",
    "api_root", "extra_api_root",
})

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
