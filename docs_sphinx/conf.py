"""Sphinx wrapper for opencv/doc/.

The wrapper lives in opencv/docs_sphinx/ as a single conf.py. Sphinx is
invoked with config-dir / source-dir separation so the wrapper never
duplicates the legacy tree. Build via the CMake `sphinx` target:

    cmake --build <build> --target sphinx
    # output -> <build>/docs_sphinx/html/

opencv/doc/ stays untouched: Doxygen-flavored directives in the .markdown
sources are translated to MyST in a `source-read` hook below.

To enable additional tutorial modules, append their directory names (the
folder under opencv/doc/tutorials/) to DOC_MODULES below. The root index
(tutorials/tutorials.markdown) lists every module, but only modules in
DOC_MODULES are actually compiled — entries for the rest are dropped
from toctrees automatically.
"""

from __future__ import annotations
import pathlib, re

HERE = pathlib.Path(__file__).parent.resolve()
DOC_ROOT = (HERE.parent / "doc").resolve()
OPENCV_ROOT = HERE.parent.resolve()

# ---------------------------------------------------------------------------
# SCOPE — add module folder names from opencv/doc/tutorials/ here.
# Override via env var to avoid editing this file:
#     OPENCV_DOC_MODULES=photo,imgproc cmake --build <build> --target sphinx
# ---------------------------------------------------------------------------
import os as _os
DOC_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_DOC_MODULES") or "photo,objdetect,dnn,gpu,others,core,calib3d,features,introduction").split(",")
    if m.strip()
]

# Sibling list for opencv/doc/js_tutorials/ modules. Same env-var override
# pattern (OPENCV_JS_DOC_MODULES). The js_tutorials root is pulled in as a
# top-level toctree entry of the master doc when this list is non-empty.
JS_DOC_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_JS_DOC_MODULES") or "js_setup").split(",")
    if m.strip()
]

# Same pattern for opencv/doc/py_tutorials/. OpenCV-Python's tree mirrors
# js_tutorials (root index + per-module table_of_content + sub-tutorials).
PY_DOC_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_PY_DOC_MODULES") or "py_setup,py_core,py_imgproc,py_video,py_photo,py_objdetect").split(",")
    if m.strip()
]

# -- Project ----------------------------------------------------------------
project = "OpenCV"
author = "OpenCV Team"
release = "5.x"

# -- Sphinx core ------------------------------------------------------------
extensions = ["myst_parser"]
for _ext in ("sphinx_design", "sphinx_copybutton"):
    try:
        __import__(_ext)
        extensions.append(_ext)
    except ImportError:
        pass
HAVE_SPHINX_DESIGN = "sphinx_design" in extensions

source_suffix = {".md": "markdown", ".markdown": "markdown"}

# Root tutorial index (lists all modules via @subpage). Stays the master
# regardless of how many modules are in DOC_MODULES.
master_doc = "tutorials/tutorials"

# Source dir is opencv/doc/ — scope to the master + enabled modules only.
include_patterns = ["tutorials/tutorials.markdown"] + [
    f"tutorials/{m}/**" for m in DOC_MODULES
] + (["js_tutorials/js_tutorials.markdown"] if JS_DOC_MODULES else []) + [
    f"js_tutorials/{m}/**" for m in JS_DOC_MODULES
] + (["py_tutorials/py_tutorials.markdown"] if PY_DOC_MODULES else []) + [
    f"py_tutorials/{m}/**" for m in PY_DOC_MODULES
]
exclude_patterns = [
    "**/Thumbs.db", "**/.DS_Store", "**/_old/**",
    "tutorials/core/how_to_use_OpenCV_parallel_for_/**",
    "tutorials/introduction/load_save_image/**",
]

myst_enable_extensions = [
    "colon_fence", "deflist", "dollarmath", "amsmath",
    "attrs_inline", "attrs_block", "smartquotes",
]
myst_heading_anchors = 4
suppress_warnings = [
    "myst.header", "myst.xref_missing", "toc.not_included",
    "misc.highlighting_failure",
    "image.not_readable",
]

# -- Doxygen integration -----------------------------------------------------
# External links in the navbar and unbuilt-module sidebar entries point at
# the existing Doxygen build. Override the base URL or tagfile via env vars.
DOXYGEN_BASE_URL = (
    _os.environ.get("OPENCV_DOXYGEN_BASE_URL", "https://docs.opencv.org/5.x/")
    .rstrip("/") + "/")
# Try both common layouts: build/ inside opencv/ (in-tree, the usual CMake
# invocation `cmake -B build` from the repo root) and build/ sibling to
# opencv/ (out-of-tree, used by some CI setups). Whichever exists wins; env
# var still overrides everything.
_TAG_CANDIDATES = (
    HERE.parent / "build" / "doc" / "doxygen" / "html" / "opencv.tag",
    HERE.parent.parent / "build" / "doc" / "doxygen" / "html" / "opencv.tag",
)
_TAG_FILE = pathlib.Path(_os.environ.get(
    "OPENCV_DOXYGEN_TAGFILE",
    str(next((p for p in _TAG_CANDIDATES if p.is_file()), _TAG_CANDIDATES[0])),
))

# anchor -> doxygen URL filename (from opencv.tag if available).
_TAG_FILENAMES: dict[str, str] = {}
# anchor -> human-readable title (from <title> in opencv.tag). Used when the
# bullet handler falls back to an external Doxygen URL: we want the link
# text to be "Video analysis (video module)" rather than the raw anchor
# name "tutorial_table_of_content_video".
_TAG_TITLES: dict[str, str] = {}
# cv-namespace short-name -> full doxygen URL (function, enum value, typedef,
# class, struct). Python tutorials reference these as `cv.cvtColor`,
# `cv.INTER_LINEAR`, `cv.Mat`, etc. — Doxygen auto-links them but CommonMark
# doesn't, so step 7c (`_linkify_cv_symbols`) reads this map to replicate it.
_CV_SYMBOL_URL: dict[str, str] = {}
if _TAG_FILE.is_file():
    try:
        import xml.etree.ElementTree as _ET
        _tag_root = _ET.parse(str(_TAG_FILE)).getroot()
        for _c in _tag_root.iter("compound"):
            _kind = _c.get("kind")
            if _kind == "page":
                _n, _f = _c.findtext("name"), _c.findtext("filename")
                _t = _c.findtext("title")
                if _n and _f:
                    _TAG_FILENAMES[_n] = _f if _f.endswith(".html") else _f + ".html"
                if _n and _t:
                    _TAG_TITLES[_n] = _t
            elif _kind == "namespace" and _c.findtext("name") == "cv":
                for _m in _c.findall("member"):
                    _n = _m.findtext("name")
                    _af = _m.findtext("anchorfile")
                    _an = _m.findtext("anchor") or ""
                    if not (_n and _af):
                        continue
                    _CV_SYMBOL_URL.setdefault(
                        _n, DOXYGEN_BASE_URL + _af + (f"#{_an}" if _an else "")
                    )
            elif _kind in ("class", "struct"):
                _full = _c.findtext("name") or ""
                if _full.startswith("cv::"):
                    _short = _full.split("::")[-1]
                    _af = _c.findtext("filename")
                    if _short and _af:
                        _CV_SYMBOL_URL.setdefault(_short, DOXYGEN_BASE_URL + _af)
            else:
                # `CV_*` C macros (e.g. CV_8U, CV_64F, CV_16S) live as
                # `kind="define"` members of source-file or group compounds.
                # Python bindings re-export them as `cv.CV_*`; capture them
                # so tutorials writing `cv.CV_8U` get linked too.
                for _m in _c.findall("member"):
                    if _m.get("kind") != "define":
                        continue
                    _n = _m.findtext("name") or ""
                    if not _n.startswith("CV_"):
                        continue
                    _af = _m.findtext("anchorfile")
                    _an = _m.findtext("anchor") or ""
                    if _af:
                        _CV_SYMBOL_URL.setdefault(
                            _n, DOXYGEN_BASE_URL + _af + (f"#{_an}" if _an else "")
                        )
    except Exception:
        pass

def _doxygen_url(page: str) -> str:
    return DOXYGEN_BASE_URL + _TAG_FILENAMES.get(page, page)


# ---- Citation numbering --------------------------------------------------
# `@cite KEY` resolves to `[N]` where N is the entry's position in
# `doc/opencv.bib` sorted case-insensitively by key (Doxygen's default
# ordering). Doxygen's live citelist.html numbers map keys to integers the
# same way; reading from the bib means our build is self-contained and
# doesn't need a network fetch.
_CITE_NUMBER: dict[str, int] = {}
_BIB_FILE = DOC_ROOT / "opencv.bib"
if _BIB_FILE.is_file():
    try:
        _bib_text = _BIB_FILE.read_text(encoding="utf-8", errors="replace")
        _bib_keys = re.findall(r"@\w+\s*\{\s*(\S+?)\s*,", _bib_text)
        for _i, _k in enumerate(sorted(_bib_keys, key=str.lower), 1):
            _CITE_NUMBER[_k] = _i
    except OSError:
        pass


# ---- Redirect-page map ---------------------------------------------------
# OpenCV's docs include many "stub" pages whose entire body is
# `Content has been moved: @ref destination`. Following these chains at
# render time means inline `@ref X` ends up pointing at the actual content
# instead of an intermediate redirect (which would itself need clicking
# through). Built from anything under `doc/{tutorials,py_tutorials,
# js_tutorials}/**/*.markdown`; `_old/` subtrees are intentionally included
# because that's where many of the canonical redirect stubs live.
_REDIRECT_MAP: dict[str, str] = {}
_REDIRECT_RE = re.compile(
    r"\{#(?P<src>[\w-]+)\}\s*\n[=\-]+\s*\n+"
    r"\s*(?:Content|Tutorial\s+content)\s+has\s+been\s+moved\b"
    r"[^@]{0,300}@ref\s+(?P<dst>[\w-]+)",
    re.IGNORECASE,
)
for _scan_dir in ("tutorials", "py_tutorials", "js_tutorials"):
    _root = DOC_ROOT / _scan_dir
    if not _root.is_dir():
        continue
    for _md in _root.rglob("*.markdown"):
        try:
            _t = _md.read_text(encoding="utf-8", errors="replace")[:2000]
        except OSError:
            continue
        _m = _REDIRECT_RE.search(_t)
        if _m:
            _REDIRECT_MAP[_m.group("src")] = _m.group("dst")

def _resolve_redirect(anchor: str) -> str:
    """Follow `_REDIRECT_MAP` transitively. Cycles bail safely."""
    seen: set[str] = set()
    while anchor in _REDIRECT_MAP and anchor not in seen:
        seen.add(anchor)
        anchor = _REDIRECT_MAP[anchor]
    return anchor

# -- HTML / PyData theme ----------------------------------------------------
try:
    import pydata_sphinx_theme  # noqa: F401
    html_theme = "pydata_sphinx_theme"
except ImportError:
    html_theme = "alabaster"

html_title = "OpenCV Tutorials"
html_show_sourcelink = False
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500&display=swap",
    "custom.css",
]
html_theme_options = {
    "logo": {"text": f"OpenCV {release}"},
    # Show all 7 Doxygen-style external links inline (no "More" dropdown).
    "header_links_before_dropdown": 7,
    # Doxygen-style top-level nav (the legacy site's MAIN PAGE / RELATED
    # PAGES / NAMESPACES / CLASSES / FILES / EXAMPLES / JAVA DOCUMENTATION).
    # All external — they target the existing Doxygen build.
    "external_links": [
        {"url": _doxygen_url("index.html"),       "name": "Main Page"},
        {"url": _doxygen_url("pages.html"),       "name": "Related Pages"},
        {"url": _doxygen_url("namespaces.html"),  "name": "Namespaces"},
        {"url": _doxygen_url("annotated.html"),   "name": "Classes"},
        {"url": _doxygen_url("files.html"),       "name": "Files"},
        {"url": _doxygen_url("examples.html"),    "name": "Examples"},
        {"url": DOXYGEN_BASE_URL + "javadoc/",    "name": "Java Documentation"},
    ],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": True,
    "show_nav_level": 2,
    "navigation_depth": 4,
    "secondary_sidebar_items": ["page-toc"],
    "back_to_top_button": True,
    "show_version_warning_banner": False,
    "icon_links": [{"name": "GitHub",
                    "url": "https://github.com/opencv/opencv",
                    "icon": "fa-brands fa-github"}],
}

# ===========================================================================
#  Doxygen-flavored .markdown  ->  MyST translation via source-read.
#  Nothing on disk under opencv/doc/ is modified.
# ===========================================================================

# Build anchor maps. Three kinds:
#   _ANCHOR_TO_DOC      anchor -> docname  (internal — for enabled modules)
#   _ANCHOR_TO_EXTERNAL anchor -> (title, url)  (external — for the rest)
#   _ANCHOR_TO_TITLE    anchor -> first-heading title (used when rendering a
#                       subpage list as a visible bulleted list with link text)
# Disabled modules still appear in the master toctree as external links to
# the Doxygen build, so the left sidebar shows the full module list.
_ANCHOR_TO_DOC: dict[str, str] = {}
_ANCHOR_TO_EXTERNAL: dict[str, tuple[str, str]] = {}
_ANCHOR_TO_TITLE: dict[str, str] = {}

_HEAD_RE = re.compile(
    r"^(?P<title1>[^\n]+?)\s*\{#(?P<anchor1>[\w-]+)\}\s*\n[=\-]{3,}\s*$"
    r"|"
    r"^#+\s+(?P<title2>[^\n]+?)\s*\{#(?P<anchor2>[\w-]+)\}\s*$",
    re.MULTILINE)

def _scan_internal(path: pathlib.Path) -> None:
    """Add every {#anchor} and standalone `@anchor NAME` in `path` (file
    or dir) to _ANCHOR_TO_DOC. Picks up both `.markdown` (the bulk of the
    tree) and `.md` (the form used by ports like dnn/dnn_pytorch_tf_*)."""
    _SUFFIXES = (".markdown", ".md")
    if path.is_file():
        files = [path] if path.suffix in _SUFFIXES else []
    elif path.is_dir():
        # Skip `_old/**` — matches exclude_patterns so we don't register
        # anchors whose target docs Sphinx never compiles.
        files = [p for s in _SUFFIXES for p in path.rglob(f"*{s}")
                 if "_old" not in p.parts]
    else:
        files = []
    for md in files:
        try:
            body = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = md.relative_to(DOC_ROOT).with_suffix("").as_posix()
        for m in re.finditer(r"\{#([\w-]+)\}", body):
            _ANCHOR_TO_DOC[m.group(1)] = rel
        for m in re.finditer(r"^@anchor\s+([\w-]+)\s*$", body, re.MULTILINE):
            _ANCHOR_TO_DOC[m.group(1)] = rel
        # Capture the first heading's title alongside its anchor so subpage
        # lists with descriptions can render with real link text.
        tm = _HEAD_RE.search(body[:4000])
        if tm:
            anchor = tm.group("anchor1") or tm.group("anchor2")
            title = (tm.group("title1") or tm.group("title2") or "").strip()
            if anchor and title:
                _ANCHOR_TO_TITLE[anchor] = title

def _scan_external(toc_file: pathlib.Path) -> None:
    """Pull the top heading's (title, anchor) from a module's table_of_content
    file and add it to _ANCHOR_TO_EXTERNAL with a URL into the Doxygen build."""
    try:
        head = toc_file.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return
    m = _HEAD_RE.search(head)
    if not m:
        return
    anchor = m.group("anchor1") or m.group("anchor2")
    title = (m.group("title1") or m.group("title2") or "").strip()
    if not anchor:
        return
    url = DOXYGEN_BASE_URL + _TAG_FILENAMES.get(anchor, "index.html")
    _ANCHOR_TO_EXTERNAL[anchor] = (title, url)

# Internal scan: master + every enabled module subtree.
_scan_internal(DOC_ROOT / "tutorials" / "tutorials.markdown")
for _m in DOC_MODULES:
    _scan_internal(DOC_ROOT / "tutorials" / _m)
if JS_DOC_MODULES:
    _scan_internal(DOC_ROOT / "js_tutorials" / "js_tutorials.markdown")
for _m in JS_DOC_MODULES:
    _scan_internal(DOC_ROOT / "js_tutorials" / _m)
if PY_DOC_MODULES:
    _scan_internal(DOC_ROOT / "py_tutorials" / "py_tutorials.markdown")
for _m in PY_DOC_MODULES:
    _scan_internal(DOC_ROOT / "py_tutorials" / _m)

# External scan: every OTHER module's top-level table_of_content_*.markdown.
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

# Doxygen flattens IMAGE_PATH across every `images/` folder under the tutorial
# tree plus the top-level `doc/images/` (per Doxyfile.in), so a tutorial can
# reference `images/foo.png` even when `foo.png` lives in a sibling module's
# `images/` directory or in `doc/images/`. Mirror that behavior by building a
# basename -> doc-root-relative-path index once at import time.
_IMAGE_INDEX: dict[str, str] = {}
# js_tutorials/ uses `js_assets/` (flat directory at the js_tutorials root)
# instead of per-tutorial `images/` folders. Include both lookup roots so
# `![alt](js_assets/foo.jpg)` references resolve the same way.
for _root in ((DOC_ROOT / "tutorials").rglob("images/*"),
              (DOC_ROOT / "js_tutorials").rglob("images/*"),
              (DOC_ROOT / "js_tutorials" / "js_assets").glob("*"),
              (DOC_ROOT / "py_tutorials").rglob("images/*"),
              (DOC_ROOT / "images").glob("*")):
    for _img in _root:
        if _img.is_file():
            _IMAGE_INDEX.setdefault(_img.name, _img.relative_to(DOC_ROOT).as_posix())

_TOGGLE_LABELS = {"cpp": "C++", "java": "Java", "python": "Python"}


# Mirror of Doxygen's EXAMPLE_PATH (see opencv/doc/Doxyfile.in) — the bases a
# bare `@snippet some/path.cpp` is resolved against. OPENCV_ROOT comes first so
# fully-qualified paths like `samples/cpp/...` keep working.
_SNIPPET_BASES = [
    OPENCV_ROOT,
    OPENCV_ROOT / "samples",
    OPENCV_ROOT / "apps",
]

# Doxygen's Doxyfile has EXAMPLE_RECURSIVE = YES, so a bare basename like
# `@snippet linux_quick_install.sh body` resolves to
# `samples/install/linux_quick_install.sh` even though the directive omits
# the `install/` qualifier. Mirror that with a basename -> path index built
# once at import time. Restricted to common source-file extensions to keep
# the scan fast.
_SNIPPET_EXTENSIONS = {
    ".cpp", ".hpp", ".h", ".c", ".cc", ".cxx",
    ".py", ".java", ".kt", ".scala", ".clj", ".groovy",
    ".sh", ".bash", ".bat", ".ps1",
    ".cmake", ".gradle",
    ".xml", ".yaml", ".yml", ".json", ".html", ".css",
    ".js", ".ts", ".rb",
}
_SNIPPET_INDEX: dict[str, pathlib.Path] = {}
for _root in (OPENCV_ROOT / "samples", OPENCV_ROOT / "apps"):
    if _root.is_dir():
        for _f in _root.rglob("*"):
            if _f.is_file() and _f.suffix.lower() in _SNIPPET_EXTENSIONS:
                _SNIPPET_INDEX.setdefault(_f.name, _f)


# Doxygen accepts language names that Pygments doesn't recognize (or wraps
# them with a leading `.` in the `@code{.lang}` and ```.lang fenced forms).
# Strip the dot and remap a few aliases so Pygments stays warning-free.
_LANG_ALIASES = {
    "none": "text",
    "unparsed": "text",
    "guess": "text",
    "gradle": "groovy",
}

def _normalize_lang(lang: str) -> str:
    lang = (lang or "").strip(".").strip().lower() or "text"
    return _LANG_ALIASES.get(lang, lang)


def _read_snippet(rel_path: str, label: str | None) -> tuple[str, str]:
    """Return (code_text, language) for an @include / @snippet directive."""
    # Some sources write the path with a leading slash (e.g. `@include
    # /samples/android/.../tutorial1_surface_view.xml`). pathlib's `/` would
    # treat that as absolute and lose the snippet base, so strip it.
    rel_norm = rel_path.lstrip("/")
    p = next((b / rel_norm for b in _SNIPPET_BASES
              if (b / rel_norm).is_file()), None)
    # Doxygen does a recursive basename lookup across EXAMPLE_PATH (see
    # opencv/doc/Doxyfile.in: EXAMPLE_RECURSIVE = YES). If the direct join
    # doesn't find the file, fall back to the prebuilt basename index.
    if p is None:
        hit = _SNIPPET_INDEX.get(pathlib.Path(rel_norm).name)
        if hit and hit.is_file():
            p = hit
    if p is None:
        return f"// not found: {rel_path}\n", "text"
    text = p.read_text(encoding="utf-8", errors="replace")
    ext = p.suffix.lower()
    lang = {".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".c": "c",
            ".py": "python", ".java": "java",
            ".xml": "xml", ".html": "html",
            ".sh": "bash", ".bash": "bash"}.get(ext, "text")
    if label is None:
        return text, lang
    # Doxygen matches `[label]` after any comment-style marker anywhere on a
    # line: //, //! and // for C/C++/Java/Kotlin, # and ## for Python/shell,
    # <!-- for XML/HTML. Block-comment-wrapped labels like
    # `/* //! [label] */` are matched via the `//`-prefix branch too.
    pat = re.compile(r"^[^\[\n]*(?://!|//|##|#|<!--)[^\[\n]*\[" + re.escape(label)
                     + r"\][^\n]*$", re.MULTILINE)
    matches = list(pat.finditer(text))
    if len(matches) < 2:
        return f"// snippet not found: {rel_path} [{label}]\n", lang
    body = text[matches[0].end():matches[1].start()].strip("\n")
    lines = body.split("\n")
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    if indents:
        dedent = min(indents)
        lines = [l[dedent:] if len(l) >= dedent else l for l in lines]
    return "\n".join(lines), lang


def _emit_toggles(tabs: list[tuple[str, str]]) -> str:
    if HAVE_SPHINX_DESIGN:
        out = ["", "``````{tab-set}"]
        for lang, body in tabs:
            label = _TOGGLE_LABELS.get(lang, lang.title())
            out += [f"`````{{tab-item}} {label}", body, "`````"]
        out += ["``````", ""]
        return "\n".join(out)
    # Fallback: render each toggle as a labeled section.
    out = [""]
    for lang, body in tabs:
        label = _TOGGLE_LABELS.get(lang, lang.title())
        out += [f"**{label}**", "", body, ""]
    return "\n".join(out)


def _translate(text: str, docname: str | None = None) -> str:
    # 0v. @verbatim ... @endverbatim — stash content first so neither math
    #     markers, @code, nor any other rule below mangles the body. Used
    #     heavily in introduction/documenting_opencv/documentation_tutorial,
    #     which shows Doxygen syntax (so the body contains literal directives,
    #     `\f[...\f]` math, and code-fences as examples). Body is restored at
    #     the very end of this function so the inserted text is safe from
    #     re-processing.
    _verbatim_stash: dict[str, str] = {}
    def _verbatim_save(body: str, inline: bool) -> str:
        key = f"VERBATIM_{len(_verbatim_stash)}"
        if inline:
            _verbatim_stash[key] = f"`{body.strip()}`"
        else:
            _verbatim_stash[key] = f"\n```text\n{body.strip()}\n```\n"
        return key
    # Block form (markers on separate lines) — run first; DOTALL across body.
    text = re.sub(
        r"@verbatim[ \t]*\n(?P<body>.*?)\n[ \t]*@endverbatim",
        lambda m: _verbatim_save(m.group("body"), inline=False),
        text, flags=re.DOTALL)
    # Inline form (both markers on the same line).
    text = re.sub(
        r"@verbatim[ \t]+(?P<body>[^\n]+?)[ \t]+@endverbatim",
        lambda m: _verbatim_save(m.group("body"), inline=True),
        text)

    # 0. Master doc: synthesize @subpage entries for the js_tutorials and/or
    #    py_tutorials roots so step 9 picks them up as toctree entries.
    #    tutorials.markdown has no direct reference to either, and editing it
    #    is forbidden, so injection here is the only way to surface those
    #    trees in the master sidebar.
    if docname == "tutorials/tutorials":
        if JS_DOC_MODULES:
            text += "\n- @subpage tutorial_js_root\n"
        if PY_DOC_MODULES:
            text += "\n- @subpage tutorial_py_root\n"

    # 0a. py_tutorials root: rewrite specific cross-tree `@ref` items to
    #     `@subpage` so the targets join the sidebar nav. The author of
    #     py_tutorials.markdown used `@ref` for these (rather than @subpage)
    #     because they live in the C++ tutorial tree, but pyData's sidebar
    #     hierarchy benefits from surfacing them under Python tutorials too.
    #     Doing this only for known cases avoids the cycle that a generic
    #     @ref-to-toctree promotion would trigger from reciprocal links
    #     between py_setup pages.
    if docname == "py_tutorials/py_tutorials":
        if "py_video" in PY_DOC_MODULES:
            text = re.sub(
                r"@ref\s+tutorial_table_of_content_video\b",
                "@subpage tutorial_py_table_of_contents_video",
                text,
            )
        # Object Detection lives in the C++ `objdetect` module (which is in
        # DOC_MODULES by default); promote the @ref so the section appears
        # under Python Tutorials in the sidebar.
        if "objdetect" in DOC_MODULES:
            text = re.sub(
                r"@ref\s+tutorial_table_of_content_objdetect\b",
                "@subpage tutorial_table_of_content_objdetect",
                text,
            )

    # 0d. py_video and py_objdetect are pure "Content has been moved" stub
    #     trees — every page just redirects to the corresponding C++
    #     tutorial. Mark them as `:orphan:` so they're compiled (for legacy
    #     URL compatibility) but stay out of the sidebar nav and don't
    #     trigger "not in any toctree" warnings. The py_tutorials root's
    #     visible link for these sections is already routed (via step 0a)
    #     to the actual destination.
    if docname and (docname.startswith("py_tutorials/py_video/")
                    or docname.startswith("py_tutorials/py_objdetect/")):
        text = "---\norphan: true\n---\n\n" + text

    # 0b. Doxygen automatic-numbered list items: "-# foo" -> "1. foo". MyST /
    #     CommonMark sequentially numbers identical-marker ordered lists, so
    #     successive "1." items render as 1, 2, 3, ...
    text = re.sub(r"^(?P<indent>[ \t]*)-#[ \t]+",
                  lambda m: f"{m.group('indent')}1. ", text, flags=re.MULTILINE)

    # 0c. Dedent "orphan" indented bullets that sit directly under a
    #     paragraph (e.g. `In this chapter, you will learn\n    -   foo` in
    #     py_imgproc/py_template_matching). Doxygen renders these as a real
    #     bullet list; CommonMark treats them as paragraph continuation or
    #     (after a blank line) as a code block. Insert a paragraph break and
    #     strip the leading indent so CommonMark sees a proper list.
    _orphan_bullets_re = re.compile(
        r"^(?P<before>(?![ \t#=*\->]).+\n)"
        r"(?P<bullets>(?:[ \t]{2,}-[ \t]+[^\n]+\n)+)",
        re.MULTILINE,
    )
    def _dedent_orphan_bullets(m: re.Match) -> str:
        before = m.group("before")
        raw = m.group("bullets")
        lines = raw.split("\n")
        nonempty = [l for l in lines if l.strip()]
        if not nonempty:
            return m.group(0)
        min_indent = min(len(l) - len(l.lstrip(" \t")) for l in nonempty)
        dedented = "\n".join(
            l[min_indent:] if len(l) >= min_indent else l for l in lines
        )
        return before + "\n" + dedented
    text = _orphan_bullets_re.sub(_dedent_orphan_bullets, text)

    # 1. Heading anchors: "Title {#name}\n===" (setext) and "## Title {#name}" (ATX).
    #    Strip the anchor from the rendered heading and emit a MyST label
    #    "(name)=" immediately above. Setext converted to ATX for simplicity.
    def _setext_repl(m: re.Match) -> str:
        title = m.group("title").strip()
        level = 1 if m.group("bar") == "=" else 2
        return f"({m.group('anchor')})=\n{'#' * level} {title}"
    text = re.sub(
        r"^(?P<title>[^\n]+?)\s*\{#(?P<anchor>[\w-]+)\}\s*\n(?P<bar>[=\-])[=\-]{2,}\s*$",
        _setext_repl, text, flags=re.MULTILINE)
    text = re.sub(
        r"^(?P<hashes>#+)\s+(?P<title>[^\n]+?)\s*\{#(?P<anchor>[\w-]+)\}\s*$",
        lambda m: f"({m.group('anchor')})=\n{m.group('hashes')} {m.group('title')}",
        text, flags=re.MULTILINE)

    # 2. Doxygen LaTeX math markers
    text = re.sub(r"\\f\[(.+?)\\f\]",
                  lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
                  text, flags=re.DOTALL)
    text = re.sub(r"\\f\$(.+?)\\f\$", lambda m: f"${m.group(1)}$",
                  text, flags=re.DOTALL)

    # 2b. \bordermatrix{...} is a Plain-TeX macro (not LaTeX), so MathJax
    #     leaves it raw. Convert to a standard `matrix` environment and
    #     translate `\cr` row separators to `\\`. Loses the bracket lines
    #     of bordermatrix but the contents render correctly.
    text = re.sub(
        r"\\bordermatrix\s*\{([^}]*)\}",
        lambda m: r"\begin{matrix}" + m.group(1).replace(r"\cr", r"\\")
                  + r"\end{matrix}",
        text)

    # 3. @code{.lang} ... @endcode
    def _code_repl(m: re.Match) -> str:
        lang = _normalize_lang(m.group("lang") or "")
        return f"\n```{lang}\n{m.group('body').strip()}\n```\n"
    text = re.sub(
        r"@code(?:\{(?P<lang>[^}]*)\})?\s*\n(?P<body>.*?)\n\s*@endcode",
        _code_repl, text, flags=re.DOTALL)

    # 3b. Tilde-fenced code blocks: `~~~~{.lang} ... ~~~~` (Doxygen-style).
    #     MyST treats `{.lang}` as a directive name, so `.py` raises an
    #     "Unknown directive" warning. Normalise to backtick-fenced output.
    def _tilde_repl(m: re.Match) -> str:
        lang = (m.group("lang") or "").strip().lstrip(".") or "text"
        if lang == "none":
            lang = "text"
        return f"\n```{lang}\n{m.group('body').strip()}\n```\n"
    text = re.sub(
        r"^~~~+(?:[ \t]*\{(?P<lang>[^}\n]+)\})?[ \t]*\n"
        r"(?P<body>.*?)\n^~~~+[ \t]*$",
        _tilde_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 3c. Normalize `yml` fence lexer -> `yaml`.
    #     Pygments emits "Lexer succeeded for unknown lexer 'yml'" otherwise.
    #     Runs after 3 and 3b so @code{.yml} / ~~~~{.yml} fences caught here too.
    text = re.sub(
        r"^(?P<fence>`{3,}|~{3,})yml\b(?P<rest>[ \t]*)$",
        lambda m: f"{m.group('fence')}yaml{m.group('rest')}",
        text, flags=re.MULTILINE)

    # 3d. \htmlonly ... \endhtmlonly  ->  raw HTML embed (used by js_tutorials
    #     for the interactive `<iframe>` "Try it" panels). MyST's `{raw} html`
    #     directive passes the body through to the HTML writer unchanged.
    text = re.sub(
        r"\\htmlonly\s*\n(?P<body>.*?)\n\s*\\endhtmlonly",
        lambda m: f"\n```{{raw}} html\n{m.group('body').strip()}\n```\n",
        text, flags=re.DOTALL)

    # 3e. Plain Markdown fences with a Doxygen-flavored language spec
    #     (e.g. "```.sh") confuse Pygments — strip the leading dot and apply
    #     the same alias map as @code{.lang}.
    text = re.sub(
        r"^(?P<fence>`{3,})(?P<lang>\.?[\w-]+)[ \t]*$",
        lambda m: f"{m.group('fence')}{_normalize_lang(m.group('lang'))}",
        text, flags=re.MULTILINE)

    # @include and @snippet both emit a plain backtick fence with the
    # captured leading indent prefixed to every body line so the fence
    # stays within an enclosing list-item content scope.
    def _emit_codeblock(indent: str, lang: str, body: str) -> str:
        body_lines = body.rstrip().splitlines()
        body_indented = "\n".join(indent + line for line in body_lines)
        return f"\n{indent}```{lang}\n{body_indented}\n{indent}```\n"

    # 4. @include path  /  @includelineno path
    #    (line numbering hint is dropped — MyST fenced blocks don't take :linenos:
    #    and PyData's code-block styling is already legible without it.)
    def _include_repl(m: re.Match) -> str:
        code, lang = _read_snippet(m.group("path"), None)
        return _emit_codeblock(m.group("indent") or "", lang, code)
    text = re.sub(r"^(?P<indent>[ \t]*)@include(?:lineno)?\s+(?P<path>\S+)",
                  _include_repl, text, flags=re.MULTILINE)

    # 4b. Remove stray @snippet that immediately follows @end_toggle at the same
    #     indent (no blank line between them). These are Doxygen fallback snippets
    #     for non-toggle Doxygen mode; the Sphinx build already shows them inside
    #     the tab-set, so the stray duplicate must be dropped before step 5
    #     would otherwise emit a second copy at document level.
    text = re.sub(
        r"(^([ \t]*)@end_toggle[ \t]*\n)\2@snippet[^\n]*\n",
        r"\1",
        text, flags=re.MULTILINE)

    # 4b. @htmlinclude path  -> raw HTML embed.
    #     Doxygen's EXAMPLE_PATH search is recursive, so a bare basename
    #     like `js_face_recognition.html` still resolves to its sample
    #     directory. If the file is a complete HTML document, lift just
    #     the <body>...</body> content to avoid nesting <html> inside
    #     the rendered page.
    def _htmlinclude_repl(m: re.Match) -> str:
        rel = m.group("path")
        p = next((b / rel for b in _SNIPPET_BASES
                  if (b / rel).is_file()), None)
        if p is None:
            name = pathlib.Path(rel).name
            for base in _SNIPPET_BASES:
                hit = next(iter(base.rglob(name)), None)
                if hit is not None:
                    p = hit
                    break
        if p is None:
            return f"<!-- htmlinclude not found: {rel} -->"
        raw = p.read_text(encoding="utf-8", errors="replace")
        # `<body ...>` opener: allow quoted attribute values so a `=>`
        # arrow function inside `onload="..."` doesn't terminate the tag.
        body = re.search(
            r"<body\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*>(.*?)</body>",
            raw, re.DOTALL | re.IGNORECASE)
        inner = body.group(1).strip() if body else raw
        return f"\n```{{raw}} html\n{inner}\n```\n"
    text = re.sub(r"^@htmlinclude\s+(?P<path>\S+)\s*$",
                  _htmlinclude_repl, text, flags=re.MULTILINE)

    # 5. @snippet path [Label]
    def _snippet_repl(m: re.Match) -> str:
        code, lang = _read_snippet(m.group("path"), m.group("label"))
        return _emit_codeblock(m.group("indent") or "", lang, code)
    text = re.sub(
        r"^(?P<indent>[ \t]*)@snippet\s+(?P<path>\S+)\s+(?P<label>[^\n]+?)\s*$",
        _snippet_repl, text, flags=re.MULTILINE)

    # 6. @add_toggle_LANG ... @end_toggle  (coalesce runs into one tab-set)
    def _toggle_collapse(src: str) -> str:
        out, i = [], 0
        opener = re.compile(r"^\s*@add_toggle_(\w+)\s*$", re.MULTILINE)
        while True:
            m = opener.search(src, i)
            if not m:
                out.append(src[i:]); break
            out.append(src[i:m.start()])
            tabs, j = [], m.start()
            while True:
                m2 = re.match(
                    r"\s*@add_toggle_(\w+)\s*\n(.*?)\n\s*@end_toggle\s*\n?",
                    src[j:], flags=re.DOTALL)
                if not m2:
                    break
                tabs.append((m2.group(1), m2.group(2)))
                j += m2.end()
                k = re.match(r"\s*", src[j:])
                if not k or not re.match(r"@add_toggle_", src[j + k.end():]):
                    break
                j += k.end()
            if not tabs:
                out.append(src[m.start():m.start() + 1]); i = m.start() + 1; continue
            out.append(_emit_toggles(tabs))
            i = j
        return "".join(out)
    text = _toggle_collapse(text)

    # 6b. @link target [display text] @endlink  ->  @ref target ["display"]
    #     Doxygen's @link/@endlink pair is the verbose form of @ref. MyST has
    #     no equivalent, so normalise to @ref and let step 7 resolve the anchor
    #     against _ANCHOR_TO_DOC (covers e.g. dnn_googlenet automatically).
    #     Display text is non-greedy and may span lines; embedded `"` chars are
    #     stripped because step 7's disp parser is `"[^"]+"`.
    def _link_repl(m: re.Match) -> str:
        target = m.group("target")
        disp = (m.group("disp") or "").strip().replace('"', '')
        return f'@ref {target} "{disp}"' if disp else f"@ref {target}"
    text = re.sub(
        r"@link\s+(?P<target>[\w-]+)(?P<disp>.*?)@endlink",
        _link_repl, text, flags=re.DOTALL)

    # 6c. Bullet lists of @subpage / @ref items (collected runs -> toctree +
    #     visible list). Runs BEFORE step 7 so @ref items are still in raw
    #     form. @subpage entries register in the toctree (navigation tree);
    #     @ref entries are render-only links — useful for module roots like
    #     py_tutorials.markdown that mix both. When items carry indented
    #     descriptions (the `js`/`py` table_of_contents format), emits a
    #     hidden toctree plus a visible link+description list so the
    #     descriptions render as paragraphs, not code blocks.
    def _subpage_list_to_toctree(src: str) -> str:
        # A bullet item: `-` + optional prefix text + @subpage/@ref + anchor.
        # E.g. `-   stitching. @subpage tutorial_stitcher` (tutorials/others/).
        bullet  = r"^[ \t]*-\s+[^\n@]*?@(?:subpage|ref)\s+[\w-]+(?:[^\n]*)\n"
        # Description: 0+ "description blocks", each being optional blank
        # lines followed by 1+ indented content lines. Accepts both
        # `- @subpage X\n\n    desc` (blank line before desc) and
        # `- @subpage X\n    desc` (desc on immediately-next line, as in
        # py_imgproc/py_transforms/py_table_of_contents_transforms.markdown).
        desc_re = r"(?:(?:[ \t]*\n)*(?:[ \t]+[^\n]+\n)+)*"
        pat = re.compile(rf"((?:{bullet}{desc_re}(?:[ \t]*\n)*)+)", re.MULTILINE)
        item_pat = re.compile(
            rf"^[ \t]*-\s+(?P<prefix>[^\n@]*?)@(?P<kind>subpage|ref)\s+(?P<anchor>[\w-]+)[^\n]*\n"
            rf"(?P<desc>{desc_re})",
            re.MULTILINE)

        def repl(m: re.Match) -> str:
            resolved: list[tuple[str, str, str, str, str, str]] = []  # kind, doctype, target, title, desc, prefix
            for im in item_pat.finditer(m.group(1)):
                kind = im.group("kind")  # "subpage" or "ref"
                anchor = im.group("anchor")
                # Bullet prefix (e.g. `stitching. ` in `-   stitching. @subpage X`)
                # — kept as plain text before the link so the "others" TOC
                # reads "stitching. High level stitching API" the way Doxygen
                # renders it.
                prefix = (im.group("prefix") or "").strip()
                desc_lines = [l.strip() for l in (im.group("desc") or "").splitlines() if l.strip()]
                description = " ".join(desc_lines)
                # `@ref` follows redirect chains; `@subpage` does not (it
                # determines navigation, so we want the literal target).
                lookup = _resolve_redirect(anchor) if kind == "ref" else anchor
                if lookup in _ANCHOR_TO_DOC:
                    resolved.append((kind, "internal", _ANCHOR_TO_DOC[lookup],
                                     _ANCHOR_TO_TITLE.get(lookup, lookup), description, prefix))
                elif lookup in _ANCHOR_TO_EXTERNAL:
                    title, url = _ANCHOR_TO_EXTERNAL[lookup]
                    resolved.append((kind, "external", url, title, description, prefix))
                elif lookup in _TAG_FILENAMES:
                    title = _TAG_TITLES.get(lookup, lookup)
                    resolved.append((kind, "external", _doxygen_url(lookup),
                                     title, description, prefix))
            if not resolved:
                return ""

            # toctree gets only @subpage entries (navigation). Putting all
            # internal @ref targets in the toctree would create cycles —
            # py_setup pages cross-reference each other (`tutorial_py_root`,
            # `tutorial_py_pip_install`, etc.) and folding those into the
            # nav tree makes Sphinx hit RecursionError. When a particular
            # cross-tree @ref needs to surface in the sidebar (e.g. the
            # py_tutorials root's reference to objdetect), inject a
            # synthetic @subpage in step 0 instead.
            tt_lines = []
            for kind, doctype, target, title, _, _ in resolved:
                if kind != "subpage":
                    continue
                tt_lines.append("/" + target if doctype == "internal"
                                else f"{title} <{target}>")
            tt_body = "\n".join(tt_lines)

            has_descriptions = any(d for *_, d, _ in resolved)
            has_prefixes = any(p for *_, p in resolved)
            if not (has_descriptions or has_prefixes):
                # Plain rendering -> Sphinx-style toctree directive (kept
                # for the photo / objdetect / etc. TOCs whose bullets are
                # bare `- @subpage X`). `:titlesonly:` blocks H2 expansion.
                if tt_body:
                    return f"\n```{{toctree}}\n:maxdepth: 1\n:titlesonly:\n\n{tt_body}\n```\n"
                # All @ref + no descriptions/prefixes: drop the run.
                return ""

            # Hidden toctree (subpages only) + visible list (all items).
            # Prefix sits OUTSIDE the link so Doxygen-style category labels
            # (`stitching. <link>`, `video. <link>`) render as plain text.
            list_lines = []
            for _kind, doctype, target, title, desc, prefix in resolved:
                href = f"/{target}" if doctype == "internal" else target
                prefix_text = f"{prefix} " if prefix else ""
                list_lines.append(f"- {prefix_text}[{title}]({href})")
                if desc:
                    list_lines.append("")
                    list_lines.append(f"  {desc}")
                list_lines.append("")
            preamble = (f"\n```{{toctree}}\n:hidden:\n:maxdepth: 1\n:titlesonly:\n\n{tt_body}\n```\n"
                        if tt_body else "")
            return f"{preamble}\n{chr(10).join(list_lines).rstrip()}\n"
        return pat.sub(repl, src)
    text = _subpage_list_to_toctree(text)

    # 7. @ref name [optional "Display Text"]
    #    Resolution order: enabled-module anchor (internal docname) ->
    #    Doxygen tag (external URL) -> fragment-only fallback. The Doxygen
    #    fallback keeps cross-module refs (e.g. js_setup -> js_imgproc) from
    #    rendering as broken in-page anchors when the target module isn't
    #    onboarded into the Sphinx wrapper yet.
    def _ref_repl(m: re.Match) -> str:
        name = m.group("name"); disp = m.group("disp")
        # Follow chained "Content has been moved: @ref X" redirects so
        # users land directly at the final destination — e.g. the inline
        # ref `@ref tutorial_table_of_content_video` inside the py_video
        # TOC resolves through `tutorial_table_of_content_video` (itself
        # a stub) to `tutorial_table_of_content_other` (the real page).
        resolved = _resolve_redirect(name)
        target = _ANCHOR_TO_DOC.get(resolved)
        if target:
            link_text = (disp or _ANCHOR_TO_TITLE.get(resolved)
                         or _TAG_TITLES.get(resolved) or resolved)
            return f"[{link_text}](/{target})"
        if resolved in _TAG_FILENAMES:
            link_text = disp or _TAG_TITLES.get(resolved, resolved)
            return f"[{link_text}]({_doxygen_url(resolved)})"
        return f"[{disp or resolved}](#{resolved})"
    # Names may be qualified C++ identifiers like `cv::saturate_cast`, so
    # the character class allows `:` in addition to word chars and `-`.
    text = re.sub(r'@ref\s+(?P<name>[\w:-]+)(?:\s+"(?P<disp>[^"]+)")?',
                  _ref_repl, text)

    # 8. @cite KEY -> `[N]` HTML anchor linking to the Doxygen citelist page,
    #    where N is the entry's alphabetical position in doc/opencv.bib
    #    (built into `_CITE_NUMBER` at module load). HTML anchor so the
    #    brackets survive markdown processing. Falls back to the key when
    #    not in the bib map. `_apply_outside_code` keeps citations inside
    #    code blocks literal.
    def _cite_repl(m: re.Match) -> str:
        key = m.group("key")
        num = _CITE_NUMBER.get(key)
        label = f"[{num}]" if num is not None else f"[{key}]"
        return (f'<a href="{DOXYGEN_BASE_URL}citelist.html#CITEREF_'
                f'{key}">{label}</a>')
    text = _apply_outside_code(text, lambda chunk: re.sub(
        r"@cite\s+(?P<key>[\w-]+)", _cite_repl, chunk))

    # 8b. @youtube{ID}  -> responsive embed (raw HTML, passed through by MyST).
    text = re.sub(
        r"^@youtube\{(?P<id>[\w-]+)\}\s*$",
        lambda m: (
            '\n<div class="opencv-youtube">'
            f'<iframe src="https://www.youtube-nocookie.com/embed/{m.group("id")}" '
            'title="YouTube video player" frameborder="0" '
            'allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
            'gyroscope; picture-in-picture" allowfullscreen></iframe></div>\n'
        ),
        text, flags=re.MULTILINE)

    # 8c. @note ... / @see ... / @warning ... -> MyST admonitions. Each body
    #     runs until the next blank line, the next @directive, or end of file
    #     (Doxygen's paragraph-level semantics). Leading whitespace before
    #     `@note` is preserved so admonitions inside `-#` list items render as
    #     nested admonitions instead of indented code blocks.
    _ADMON_KIND = {"note": "note", "see": "seealso", "warning": "warning"}
    def _admon_repl(m: re.Match) -> str:
        kind = _ADMON_KIND[m.group("dir")]
        indent = m.group("indent") or ""
        body = m.group("body").rstrip()
        # When the fence is indented (admonition inside a `-#`/`1.` list
        # item) the body's content block must also sit at that column for
        # MyST to recognize it. Re-indent lines that don't already match:
        # this covers same-line forms like `    @note Foo...\n    bar...`
        # whose first body line has zero leading whitespace because the
        # regex's `[ \t]*` ate the space after the directive name.
        if indent:
            re_indented = []
            for line in body.split("\n"):
                if not line.strip() or line.startswith(indent):
                    re_indented.append(line)
                else:
                    re_indented.append(indent + line.lstrip(" \t"))
            body = "\n".join(re_indented)
        return f"\n{indent}:::{{{kind}}}\n{body}\n{indent}:::\n"
    # The optional `:?` after the directive name accepts the (non-standard
    # but common in OpenCV-Python docs) form `@note: text` alongside the
    # canonical `@note text` / `@note\n text`.
    text = re.sub(
        r"^(?P<indent>[ \t]*)@(?P<dir>note|see|warning):?[ \t]*\n?"
        r"(?P<body>.+?)(?=\n[ \t]*\n|\n[ \t]*@[A-Za-z]|\Z)",
        _admon_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 10. @next_tutorial / @prev_tutorial  -> drop
    text = re.sub(r"^@(?:next|prev)_tutorial\{[^}]*\}\s*$", "",
                  text, flags=re.MULTILINE)

    # 11. @tableofcontents -> drop (PyData right sidebar replaces it)
    text = re.sub(r"^@tableofcontents\s*$", "", text, flags=re.MULTILINE)

    # 11b. @cond NAME ... @endcond  -> strip just the markers; if the
    #      enclosed @subpage points to a disabled module it gets dropped
    #      by _subpage_list_to_toctree above.  Same treatment for @parblock /
    #      @endparblock — they exist only to let Doxygen accept multi-
    #      paragraph arguments to directives like @note, which Markdown
    #      already handles natively, so the markers can be dropped.
    text = re.sub(r"^@cond\s+\S+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^@endcond\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*@parblock\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*@endparblock\s*$", "", text, flags=re.MULTILINE)

    # 11c. @anchor NAME  ->  MyST label "(NAME)=" so the following block
    #      element (typically a heading) becomes the cross-reference target.
    text = re.sub(r"^@anchor\s+(?P<name>[\w-]+)\s*$",
                  lambda m: f"({m.group('name')})=",
                  text, flags=re.MULTILINE)

    # 11d. Doxygen numbered-list marker `-#` at line start -> Markdown `1.`.
    #      Markdown auto-increments numbered lists, so emitting `1.` for every
    #      item produces the right output. Preserves leading indent for nested
    #      lists.
    text = re.sub(r"^(?P<indent>[ \t]*)-#(?P<sp>[ \t]+)",
                  lambda m: f"{m.group('indent')}1.{m.group('sp')}",
                  text, flags=re.MULTILINE)

    # 11e. Bullet markers with 5+ spaces between marker and content cause MyST
    #      to treat the continuation lines as code blocks. Normalize such
    #      bullets to 3-space separation and re-flow the continuation column
    #      by the same delta so nested content stays aligned with the marker.
    def _normalize_over_indented_markers(src: str) -> str:
        lines_in = src.split("\n")
        out: list[str] = []
        i = 0
        while i < len(lines_in):
            m = re.match(r"^([ \t]*)([-*+])( {5,})(.*)", lines_in[i])
            if m:
                outer, marker, spaces, content = (
                    m.group(1), m.group(2), m.group(3), m.group(4))
                old_col = len(outer) + 1 + len(spaces)
                new_col = len(outer) + 1 + 3
                delta = old_col - new_col
                out.append(f"{outer}{marker}   {content}")
                i += 1
                while i < len(lines_in):
                    line = lines_in[i]
                    stripped = line.lstrip(" \t")
                    actual = len(line) - len(stripped)
                    if not stripped:
                        out.append(line); i += 1; continue
                    if actual >= old_col:
                        out.append(" " * (actual - delta) + stripped); i += 1
                    else:
                        break
            else:
                out.append(lines_in[i]); i += 1
        return "\n".join(out)
    text = _normalize_over_indented_markers(text)

    # 11f. Bullet lists immediately after a heading are sometimes indented by
    #      4 spaces in Doxygen sources — Markdown would interpret that as a
    #      code block. Strip exactly one level of 4-space indent off such
    #      runs so MyST renders a proper list.
    text = re.sub(
        r"(^#{1,6}[ \t][^\n]+\n(?:[ \t]*\n)*)((?:    [ \t]*[-*+][^\n]*\n)+)",
        lambda m: m.group(1) + re.sub(r"^    ", "", m.group(2), flags=re.MULTILINE),
        text, flags=re.MULTILINE)

    # 12. Image paths "images/foo.png" or "js_assets/foo.jpg" — resolve like
    #     Doxygen's flat IMAGE_PATH: prefer the doc's own asset sibling, then
    #     fall back to a global basename lookup across every tutorial `images/`
    #     folder plus `js_tutorials/js_assets/`. As a final fallback, point at
    #     the consolidated `tutorials/others/images/` dir (where modules like
    #     `photo` store their assets). Also matches `<prefix>/images/foo.png`
    #     — some tutorials prefix the directory name (multiview_calibration
    #     does this) which Doxygen IMAGE_PATH ignores but MyST would resolve
    #     literally and miss.
    def _img_repl(m: re.Match) -> str:
        rel = m.group("rel")
        asset_dir = m.group("dir")
        if docname:
            local = DOC_ROOT / pathlib.Path(docname).parent / asset_dir / rel
            if local.is_file():
                return f'{m.group("pre")}images/{rel})'
        hit = _IMAGE_INDEX.get(pathlib.Path(rel).name)
        if hit:
            return f'{m.group("pre")}/{hit})'
        return f'{m.group("pre")}/tutorials/others/images/{rel})'
    text = re.sub(
        r'(?P<pre>!\[[^\]]*\]\()(?:[^)]*?/)?(?P<dir>images|js_assets)/(?P<rel>[^)]+)\)',
        _img_repl, text)

    # 13. Front-matter table: OpenCV tutorials use the "| -: | :- |"
    #     alignment pattern for the Original-author/Compatibility block.
    #     Wrap it in a {div} carrying .opencv-meta-table so custom.css
    #     can pin the rounded card + label-column styling without us
    #     modifying the .markdown source.
    def _wrap_front_matter(src: str) -> str:
        pat = re.compile(
            r"(^\|[^\n]*\|[ \t]*\n"     # header row (often empty)
            r"\|[ \t]*-:[ \t]*\|[ \t]*:-[ \t]*\|[ \t]*\n"  # alignment row
            r"(?:\|[^\n]*\|[ \t]*\n)+)",  # one or more body rows
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            return f":::{{div}} opencv-meta-table\n\n{m.group(1)}\n:::\n"
        return pat.sub(repl, src, count=1)
    text = _wrap_front_matter(text)

    # 14a. Auto-link `cv.SymbolName` references in Python tutorial prose so
    #      they point at the matching Doxygen API page (Doxygen's auto-linker
    #      does this magic out of the box; CommonMark doesn't). Skip when the
    #      symbol isn't in the tag-file map so unknown names stay literal.
    text = _linkify_cv_symbols(text)

    # 14b. Auto-linkify bare URLs. CommonMark requires explicit `<URL>` or
    #      `[text](URL)` markup to make a URL clickable — Doxygen's renderer
    #      was lenient and turned bare `https://...` into links. Mirror that
    #      by wrapping bare URLs in `<...>`. Runs after the cv-symbol step
    #      so the new `[cv.X](URL)` links aren't picked up as bare URLs.
    text = _linkify_bare_urls(text)

    # 15. Restore @verbatim stash (see step 0v).
    for _vk, _vv in _verbatim_stash.items():
        text = text.replace(_vk, _vv)

    return text


_BARE_URL_RE = re.compile(
    r"(?<![<\[(\w\"'=])"
    r"(?P<url>https?://[^\s<>()`\"']+[^\s<>()`\"'.,;:!?])"
)
# Match `cv.X` (Python) or `cv::X` (C++) — both reference the same OpenCV
# symbol map. Negative lookbehind blocks `frame.cv.X`, `mycv.X`, `[cv::X`,
# and `foo::cv::X`-style false positives. Captures the separator so the
# link text preserves the source's style (`cv.cvtColor` vs `cv::cvtColor`).
# Optionally consumes a following `()` so the link reads `cv.foo()` /
# `cv::foo()` when the source used the parens-only call form.
_CV_SYMBOL_RE = re.compile(
    r"(?<![/\w.:\[])(?P<sep>cv(?:\.|::))(?P<sym>[A-Za-z_]\w*)(?P<parens>\(\))?"
)
# Bare `funcName()` references — Doxygen auto-linked these too, even without
# the `cv.` prefix. The angle-bracket exclusions in the lookbehind avoid
# nested-anchor double-wrapping after `_CV_SYMBOL_RE` already replaced
# `cv.X()` -> `<a ...>cv.X()</a>` (the X part inside the anchor is preceded
# by `.`, but ensure other anchor internals are also off-limits).
_BARE_FN_RE = re.compile(r"(?<![/\w.<>\"])(?P<sym>[A-Za-z_]\w{2,})\(\)")
_FENCED_BLOCK_RE = re.compile(
    r"^(?P<fence>[`~]{3,})[^\n]*\n[\s\S]*?\n(?P=fence)[ \t]*$",
    re.MULTILINE,
)
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*?`+")


def _apply_outside_code(src: str, transform) -> str:
    """Apply `transform(str) -> str` to every region of `src` that is not
    inside a fenced code block or an inline code span."""
    def _segment(text: str) -> str:
        out, last = [], 0
        for cm in _INLINE_CODE_RE.finditer(text):
            out.append(transform(text[last:cm.start()]))
            out.append(cm.group(0))
            last = cm.end()
        out.append(transform(text[last:]))
        return "".join(out)
    out, last = [], 0
    for fm in _FENCED_BLOCK_RE.finditer(src):
        out.append(_segment(src[last:fm.start()]))
        out.append(fm.group(0))
        last = fm.end()
    out.append(_segment(src[last:]))
    return "".join(out)


def _linkify_bare_urls(src: str) -> str:
    return _apply_outside_code(src,
        lambda chunk: _BARE_URL_RE.sub(r"<\g<url>>", chunk))


def _linkify_cv_symbols(src: str) -> str:
    if not _CV_SYMBOL_URL:
        return src
    def repl_cv(m: re.Match) -> str:
        sym = m.group("sym")
        url = _CV_SYMBOL_URL.get(sym)
        if not url:
            return m.group(0)
        sep = m.group("sep")  # "cv." (Python) or "cv::" (C++)
        parens = m.group("parens") or ""
        # HTML anchor (not markdown `[text](url)`) so the link survives when
        # the source embeds the reference inside raw HTML — e.g. the
        # `<center><em>cv.calcHist(...)</em></center>` function-signature
        # blocks in py_histograms. CommonMark doesn't re-parse markdown
        # inside raw HTML blocks; inline HTML inside markdown does render.
        return f'<a href="{url}">{sep}{sym}{parens}</a>'
    def repl_bare(m: re.Match) -> str:
        sym = m.group("sym")
        url = _CV_SYMBOL_URL.get(sym)
        if not url:
            return m.group(0)
        return f'<a href="{url}">{sym}()</a>'
    def transform(chunk: str) -> str:
        chunk = _CV_SYMBOL_RE.sub(repl_cv, chunk)
        chunk = _BARE_FN_RE.sub(repl_bare, chunk)
        return chunk
    return _apply_outside_code(src, transform)


def _source_read(app, docname, source):
    # Translate any tutorial doc — the root index plus everything under a
    # module we enabled in DOC_MODULES / JS_DOC_MODULES / PY_DOC_MODULES.
    if not (docname.startswith("tutorials/")
            or docname.startswith("js_tutorials/")
            or docname.startswith("py_tutorials/")):
        return
    source[0] = _translate(source[0], docname)


def setup(app):
    app.connect("source-read", _source_read)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
