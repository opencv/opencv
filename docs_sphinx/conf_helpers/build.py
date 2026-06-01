"""Import-time orchestration: populate the shared indexes."""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *
from .xml_render import _patch_namespace_xml_for_breathe
from .stubs import _generate_api_stubs

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
# Contrib root page filename differs by staging vintage: the current CMake
# (docs_sphinx/CMakeLists.txt) auto-generates `contrib_root.markdown` (heading
# anchor `tutorial_contrib_root`); older staged trees named it
# `tutorials_contrib.markdown`. Accept whichever exists so the landing-page
# link and the anchor scan both resolve regardless of which produced the tree.
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

if API_MODULES:
    # 0) Module API figures (e.g. modules/calib/doc/pics/pinhole_camera_model.png,
    #    referenced by Doxygen `@image html …` in group docs) live OUTSIDE
    #    DOC_ROOT, so the tutorial scan above never indexed or staged them.
    #    Mirror them flat into `api_pics/` under the srcdir and index by
    #    filename so the XML `<image>` converter resolves them as
    #    `/api_pics/<name>`. (Doxygen's IMAGE_PATH is flat, so basenames are
    #    effectively unique; first writer wins on the rare clash.)
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
    # 1) Build a patched XML tree breathe will read (inlines group-only
    #    <memberdef>s into namespace XML so name lookups succeed).
    if _API_XML_DIR.is_dir():
        _patch_namespace_xml_for_breathe(_API_XML_DIR, _PATCHED_XML_DIR)
    # 2) Generate stub trees from the ORIGINAL XML. Split by origin: main tree
    #    (opencv/modules) → main_modules/, contrib tree → extra_modules/.
    from conf_helpers.state import OPENCV_ROOT, CONTRIB_ROOT
    _is_contrib = lambda m: (CONTRIB_ROOT / m).is_dir() and not (
        OPENCV_ROOT / "modules" / m).is_dir()
    _main_api = [m for m in API_MODULES if not _is_contrib(m)]
    _extra_api = [m for m in API_MODULES if _is_contrib(m)]
    _generate_api_stubs(_main_api, _API_XML_DIR, SPHINX_INPUT_ROOT / "main_modules",
                        root_anchor="api_root", root_title="Main modules")
    _scan_internal(SPHINX_INPUT_ROOT / "main_modules")
    if _extra_api:
        _generate_api_stubs(_extra_api, _API_XML_DIR, SPHINX_INPUT_ROOT / "extra_modules",
                            root_anchor="extra_api_root", root_title="Extra modules")
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

    # Body: raw HTML so links resolve correctly relative to index.html.
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

# These roots reach the master toctree via @subpage lines injected at
# translate/source-read time (not present in any scanned source file), so add
# them to the referenced set explicitly — otherwise orphan detection would
# wrongly flag the js/py/contrib roots and the standalone pages.
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
