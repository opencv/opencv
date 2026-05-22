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
import pathlib, re, textwrap as _textwrap

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
    for m in (_os.environ.get("OPENCV_DOC_MODULES") or "photo,objdetect,core,calib3d,features,3d,app,introduction,imgproc,ios").split(",")
    if m.strip()
]

# ---------------------------------------------------------------------------
# SCOPE — contrib tree.  Folder names under opencv_contrib/modules/.
# Override via env var to avoid editing this file:
#     OPENCV_CONTRIB_MODULES=ml,bgsegm cmake --build <build> --target sphinx
# Empty list = main-only build (legacy behavior, no contrib site).
# ---------------------------------------------------------------------------
CONTRIB_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_CONTRIB_MODULES") or "ml,bgsegm,bioinspired,cannops,ccalib,cnn_3dobj,cvv,dnn_objdetect,dnn_superres,gapi,hdf,julia,line_descriptor,phase_unwrapping,structured_light").split(",")
    if m.strip()
]
CONTRIB_ROOT = pathlib.Path(
    _os.environ.get("OPENCV_CONTRIB_ROOT")
    or str(HERE.parent.parent / "opencv_contrib" / "modules")
).resolve()

# Sphinx srcdir as seen by conf.py.  CMake stages a merged tree at
# ${CMAKE_BINARY_DIR}/docs_sphinx_input/ and forwards this env var.
# Default = DOC_ROOT so ad-hoc sphinx-build runs keep working. The `or`
# idiom (rather than dict.get's default) treats an empty-string env var
# the same as unset — CMake forwards "" when contrib is disabled.
SPHINX_INPUT_ROOT = pathlib.Path(
    _os.environ.get("OPENCV_SPHINX_INPUT_ROOT") or str(DOC_ROOT)
).resolve()

# -- Project ----------------------------------------------------------------
project = "OpenCV"
author = "OpenCV Team"
release = "5.x"

# -- Sphinx core ------------------------------------------------------------
extensions = ["myst_parser", "sphinx.ext.graphviz"]
# Render Doxygen \dot ... \enddot blocks as inline SVG (matches Doxygen's
# DOT_IMAGE_FORMAT=svg default — keeps text crisp and selectable).
graphviz_output_format = "svg"
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

# Source dir is the staged tree (or DOC_ROOT for legacy ad-hoc runs).
# Scope: master + enabled main modules + (optionally) enabled contrib modules.
include_patterns = ["tutorials/tutorials.markdown"] + [
    f"tutorials/{m}/**" for m in DOC_MODULES
]
if CONTRIB_MODULES and (SPHINX_INPUT_ROOT / "tutorials_contrib").is_dir():
    include_patterns.append("tutorials_contrib/contrib_root.markdown")
    include_patterns += [f"tutorials_contrib/{m}/**" for m in CONTRIB_MODULES]
exclude_patterns = [
    "**/Thumbs.db", "**/.DS_Store",
    "tutorials/core/how_to_use_OpenCV_parallel_for_/**",
    "tutorials/introduction/load_save_image/**",
    "tutorials/app/_old/**",
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
_TAG_FILE = pathlib.Path(_os.environ.get(
    "OPENCV_DOXYGEN_TAGFILE",
    str(HERE.parent.parent / "build" / "doc" / "doxygen" / "html" / "opencv.tag"),
))

# anchor -> doxygen URL filename (from opencv.tag if available).
_TAG_FILENAMES: dict[str, str] = {}
if _TAG_FILE.is_file():
    try:
        import xml.etree.ElementTree as _ET
        for _c in _ET.parse(str(_TAG_FILE)).getroot().iter("compound"):
            if _c.get("kind") == "page":
                _n, _f = _c.findtext("name"), _c.findtext("filename")
                if _n and _f:
                    _TAG_FILENAMES[_n] = _f if _f.endswith(".html") else _f + ".html"
    except Exception:
        pass

def _doxygen_url(page: str) -> str:
    return DOXYGEN_BASE_URL + _TAG_FILENAMES.get(page, page)

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

# Doxygen defines \fork{a}{b}{c}{d} as a piecewise-function shorthand.
# Define it as a MathJax macro so threshold.markdown renders correctly.
mathjax3_config = {
    "tex": {
        "macros": {
            "fork": [r"\left\{ \begin{array}{ll} #1 & \mbox{#2}\\ #3 & \mbox{#4}\end{array} \right.", 4],
        }
    }
}

# ===========================================================================
#  Doxygen-flavored .markdown  ->  MyST translation via source-read.
#  Nothing on disk under opencv/doc/ is modified.
# ===========================================================================

# Build anchor maps. Two kinds:
#   _ANCHOR_TO_DOC      anchor -> docname  (internal — for enabled modules)
#   _ANCHOR_TO_EXTERNAL anchor -> (title, url)  (external — for the rest)
# Disabled modules still appear in the master toctree as external links to
# the Doxygen build, so the left sidebar shows the full module list.
_ANCHOR_TO_DOC: dict[str, str] = {}
_ANCHOR_TO_EXTERNAL: dict[str, tuple[str, str]] = {}

_HEAD_RE = re.compile(
    r"^(?P<title1>[^\n]+?)\s*\{#(?P<anchor1>[\w-]+)\}\s*\n[=\-]{3,}\s*$"
    r"|"
    r"^#+\s+(?P<title2>[^\n]+?)\s*\{#(?P<anchor2>[\w-]+)\}\s*$",
    re.MULTILINE)

def _scan_internal(path: pathlib.Path, base: pathlib.Path | None = None) -> None:
    """Add every {#anchor} and standalone `@anchor NAME` in `path` (file or
    dir) to _ANCHOR_TO_DOC. Docname is computed relative to `base` (default
    SPHINX_INPUT_ROOT) so the same scanner serves both main and contrib."""
    base = base or SPHINX_INPUT_ROOT
    files = [path] if (path.is_file() and path.suffix == ".markdown") \
        else (list(path.rglob("*.markdown")) if path.is_dir() else [])
    for md in files:
        try:
            body = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Use the unresolved path so symlinks in the staged input tree
        # produce docnames relative to the staging root, not to their
        # real source location (opencv/doc/ or opencv_contrib/modules/).
        rel = md.relative_to(base).with_suffix("").as_posix()
        for m in re.finditer(r"\{#([\w-]+)\}", body):
            _ANCHOR_TO_DOC[m.group(1)] = rel
        for m in re.finditer(r"^@anchor\s+([\w-]+)\s*$", body, re.MULTILINE):
            _ANCHOR_TO_DOC[m.group(1)] = rel

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

# Internal scan: master + every enabled main and contrib module subtree.
# Walk the staged tree so docnames stay relative to SPHINX_INPUT_ROOT (Sphinx
# srcdir), regardless of where the actual source files live on disk.
_scan_internal(SPHINX_INPUT_ROOT / "tutorials" / "tutorials.markdown")
for _m in DOC_MODULES:
    _scan_internal(SPHINX_INPUT_ROOT / "tutorials" / _m)
_contrib_root_md = SPHINX_INPUT_ROOT / "tutorials_contrib" / "contrib_root.markdown"
if _contrib_root_md.is_file():
    _scan_internal(_contrib_root_md)
for _m in CONTRIB_MODULES:
    _scan_internal(SPHINX_INPUT_ROOT / "tutorials_contrib" / _m)

# External scan: every OTHER main module's top-level table_of_content_*.markdown.
# Sources live under DOC_ROOT (the staged tree only contains *enabled* main
# modules, not the rest), so scan DOC_ROOT directly here.
for _toc in (DOC_ROOT / "tutorials").glob("*/table_of_content_*.markdown"):
    if _toc.parent.name not in DOC_MODULES:
        _scan_external(_toc)

# Basename -> srcdir-relative URL index for image lookup, mirroring
# Doxygen's flat IMAGE_PATH. Walks source trees directly (not the staged
# tree) because pathlib.rglob in Python <3.13 doesn't follow symlinks.
_IMAGE_INDEX: dict[str, str] = {}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp"}
for _img in (DOC_ROOT / "tutorials").rglob("images/*"):
    if _img.is_file():
        _IMAGE_INDEX.setdefault(_img.name,
                                _img.relative_to(DOC_ROOT).as_posix())
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


# Expose each enabled contrib module as a symlink under a build-dir subdir
# and let Sphinx's html_extra_path publish the tree to the output. Output
# URLs are /contrib_modules/<m>/... — no files duplicated in srcdir.
# Skipped when SPHINX_INPUT_ROOT lives inside a source tree (i.e. ad-hoc
# sphinx-build without CMake's OPENCV_SPHINX_INPUT_ROOT) — matches the
# documented "always build through CMake" expectation.
html_extra_path: list[str] = []
def _in_source_tree(p: pathlib.Path) -> bool:
    for _root in (DOC_ROOT, CONTRIB_ROOT):
        try: p.relative_to(_root); return True
        except ValueError: pass
    return False
if not _in_source_tree(SPHINX_INPUT_ROOT):
    _extras = SPHINX_INPUT_ROOT.parent / "contrib_extras"
    _prefix = _extras / "contrib_modules"
    _prefix.mkdir(parents=True, exist_ok=True)
    for _m in CONTRIB_MODULES:
        _src, _link = CONTRIB_ROOT / _m, _prefix / _m
        if _src.is_dir() and not _link.exists():
            try: _os.symlink(_src, _link, target_is_directory=True)
            except (OSError, NotImplementedError): pass
    html_extra_path = [str(_extras)]

_TOGGLE_LABELS = {"cpp": "C++", "java": "Java", "python": "Python"}


# Mirror of Doxygen's EXAMPLE_PATH (see opencv/doc/Doxyfile.in) — the bases a
# bare `@snippet some/path.cpp` is resolved against. OPENCV_ROOT comes first so
# fully-qualified paths like `samples/cpp/...` keep working. Contrib module
# samples are appended so `@snippet introduction_to_svm.cpp ...` in a contrib
# tutorial resolves to opencv_contrib/modules/<m>/samples/...
_SNIPPET_BASES = [
    OPENCV_ROOT,
    OPENCV_ROOT / "samples",
    OPENCV_ROOT / "apps",
] + [CONTRIB_ROOT / _m / "samples" for _m in CONTRIB_MODULES]

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
_snippet_scan_roots = [OPENCV_ROOT / "samples", OPENCV_ROOT / "apps"] + [
    CONTRIB_ROOT / _m / "samples" for _m in CONTRIB_MODULES]
for _root in _snippet_scan_roots:
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
    # `run` is a custom convention some contrib tutorials use to mean
    # "this is a shell command you run" (e.g. dnn_superres/upscale_image_*).
    # Pygments has no `run` lexer — map to bash so it highlights as shell.
    "run": "bash",
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


def _emit_toggles(tabs: list[tuple[str, str]], indent: str = "") -> str:
    def _dedent(body: str) -> str:
        # Fully dedent the tab-item body to column 0. The {tab-item} directive
        # resets the parser context, so directives (```{code-block}```) must be
        # at column 0 to be recognised — not at 4-space where CommonMark treats
        # them as indented code blocks. Using the actual minimum indent of all
        # non-empty lines (instead of exactly len(indent)) handles the case
        # where @snippet/@include is indented deeper than @add_toggle (e.g.
        # 8-space snippet inside a 4-space toggle block).
        lines = body.split("\n")
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return body
        min_ind = min(len(l) - len(l.lstrip()) for l in non_empty)
        if min_ind == 0:
            return body
        return "\n".join(l[min_ind:] if l.strip() else l for l in lines)
    if HAVE_SPHINX_DESIGN:
        out = ["", "``````{tab-set}"]
        for lang, body in tabs:
            label = _TOGGLE_LABELS.get(lang, lang.title())
            out += [f"`````{{tab-item}} {label}", _dedent(body), "`````"]
        out += ["``````", ""]
        return "\n".join(out)
    # Fallback: render each toggle as a labeled section.
    out = [""]
    for lang, body in tabs:
        label = _TOGGLE_LABELS.get(lang, lang.title())
        out += [f"**{label}**", "", _dedent(body), ""]
    return "\n".join(out)


def _translate(text: str, docname: str | None = None) -> str:
    # 0. @verbatim ... @endverbatim — stash content first so neither math
    #    markers, @code, nor any other rule below mangles the body. Used
    #    heavily in introduction/documenting_opencv/documentation_tutorial,
    #    which shows Doxygen syntax (so the body contains literal directives,
    #    `\f[...\f]` math, and code-fences as examples). Body is restored at
    #    the very end of this function with a private-use placeholder so the
    #    inserted text is safe from re-processing.
    _verbatim_stash: dict[str, str] = {}
    def _verbatim_save(body: str, inline: bool) -> str:
        key = f"VERBATIM_{len(_verbatim_stash)}"
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

    # 1b. Convert a trailing setext heading at EOF to ATX. Otherwise
    #     docutils rejects the doc as ending with a transition.
    text = re.sub(
        r"^(?P<title>[^\n#=\-][^\n]*?)[ \t]*\n(?P<bar>[=\-])[=\-]{2,}[ \t]*$\s*\Z",
        lambda m: f"{'#' if m.group('bar') == '=' else '##'} {m.group('title').strip()}\n",
        text, flags=re.MULTILINE)

    # 1c. Convert remaining mid-doc setext H1s to ATX so 1d can see them.
    text = re.sub(
        r"^(?P<title>[^\n#=\-][^\n]*?)[ \t]*\n=[=]{2,}[ \t]*$",
        lambda m: f"# {m.group('title').strip()}",
        text, flags=re.MULTILINE)

    # 1d. Demote every H1 after the first to H2 so multi-H1 Doxygen docs
    #     (one `# Heading` per section) end up with a proper "1 title +
    #     N sections" outline. Without this, Sphinx's toctree lists every
    #     H1 as a separate entry on the parent TOC page.
    def _demote_extra_h1s(src: str) -> str:
        fence_open_re = re.compile(r'^[ \t]*(?:`{3,}|~{3,})')
        atx_h1_re = re.compile(r'^#\s')
        h1_count = 0
        in_fence = False
        out = []
        for line in src.split('\n'):
            if fence_open_re.match(line):
                in_fence = not in_fence
                out.append(line)
                continue
            if in_fence:
                out.append(line)
                continue
            if atx_h1_re.match(line):
                h1_count += 1
                if h1_count > 1:
                    line = '#' + line   # H1 → H2
            out.append(line)
        return '\n'.join(out)
    text = _demote_extra_h1s(text)

    # 2. Doxygen LaTeX math markers
    # 1a-i. Over-indented list markers.
    #   A bullet marker (-/*/+) followed by 5+ spaces causes CommonMark to use
    #   marker_col+2 as the continuation indent (the extra spaces beyond 4 are
    #   treated as content), making the actual text 6+ spaces into the content —
    #   above the 4-space threshold for indented code blocks.  Normalise to
    #   exactly 3 spaces after the marker and reduce all continuation lines by
    #   the same delta so nested structure is preserved.
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

    # 1a-ii. 4-space indented list items directly under a heading.
    #   Doxygen ignores visual indentation; CommonMark treats 4-space-indented
    #   lines as indented code blocks.  Strip the 4-space prefix when list items
    #   immediately follow a heading (with optional blank lines between).
    text = re.sub(
        r"(^#{1,6}[ \t][^\n]+\n(?:[ \t]*\n)*)((?:    [ \t]*[-*+][^\n]*\n)+)",
        lambda m: m.group(1) + re.sub(r"^    ", "", m.group(2), flags=re.MULTILINE),
        text, flags=re.MULTILINE)

    # 1a-iii. 4-space list items under a plain paragraph line → strip to fix lazy continuation.
    text = re.sub(
        r"(^(?![ \t#@`]|-#|[-*+]\s|\d+[.)]\s)[^\n]+\n)((?:    [-*+][ \t][^\n]*\n(?:[ \t]{5,}[^\n]*\n)*)+)",
        lambda m: m.group(1) + re.sub(r"^    ", "", m.group(2), flags=re.MULTILINE),
        text, flags=re.MULTILINE)

    # 1b. @note ... / @see ...  -> MyST admonitions.
    #     Runs BEFORE math conversion so that \f[...\f] inside a note body is
    #     still on one logical line and does not create a blank-line terminator
    #     that would cut the body short.
    #     Allow optional leading indent and bare @note (body on next line).
    #     Dedent the body so indented lines don't become code blocks inside
    #     the directive.
    _ADMON_KIND = {"note": "note", "see": "seealso", "warning": "warning"}
    def _admon_repl(m: re.Match) -> str:
        indent = m.group("indent")
        kind = _ADMON_KIND[m.group("dir")]
        raw = m.group("body")
        lines = raw.split("\n")
        min_ind = min(
            (len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        body = "\n".join(l[min_ind:] for l in lines).strip()
        return f"\n{indent}:::{{{kind}}}\n{indent}{body}\n{indent}:::\n"
    text = re.sub(
        r"^(?P<indent>[ \t]*)@(?P<dir>note|see|warning)[ \t]*\n?(?P<body>.+?)(?=\n[ \t]*\n|\n[ \t]*@[A-Za-z]|\Z)",
        _admon_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 2. Doxygen LaTeX math markers.
    #    Block \f[...\f]: consume leading indent and re-emit it on the $$
    #    fence lines so the block stays inside any enclosing list item and
    #    the text that follows (at the same indent) is not misread as a code block.
    #
    #    Preprocess: when two \f[...\f] blocks are adjacent on the same source
    #    line (e.g. \end{bmatrix}\f]\f[G_{y}), split them onto separate lines
    #    and prefix the second \f[ with the line's own leading indent.  Without
    #    this the primary regex below would convert the first block correctly but
    #    leave the second \f[ at column 0 in the output; the fallback then emits
    #    that block at column 0, breaking any enclosing list structure.
    def _split_adj_math(m: re.Match) -> str:
        indent = m.group("indent")
        return m.group(0).replace("\\f]\\f[", f"\\f]\n{indent}\\f[")
    text = re.sub(r"^(?P<indent>[ \t]*)[^\n]*\\f\]\\f\[",
                  _split_adj_math, text, flags=re.MULTILINE)

    def _block_math_repl(m: re.Match) -> str:
        ind = m.group("indent")
        return f"\n{ind}$$\n{m.group('body').strip()}\n{ind}$$\n"
    text = re.sub(r"^(?P<indent>[ \t]*)\\f\[(?P<body>.+?)\\f\]",
                  _block_math_repl, text, flags=re.DOTALL | re.MULTILINE)
    # Fallback for any \f[...\f] not at line-start (e.g. two adjacent blocks).
    text = re.sub(r"\\f\[(.+?)\\f\]",
                  lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
                  text, flags=re.DOTALL)
    # Inline math.
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

    # 2c. Normalise unknown Pygments lexer names: plaintext/bash/sh → text.
    text = re.sub(r"^([ \t]*)```plaintext\b", r"\1```text", text, flags=re.MULTILINE)
    text = re.sub(r"^([ \t]*)```(?:bash|sh)\b", r"\1```text", text, flags=re.MULTILINE)

    # 3. @code{.lang} ... @endcode → fenced block. Preserve the indent
    #    so blocks nested under a bullet item stay inside the list; for
    #    col-0 @code keep the legacy .strip() form (byte-identical).
    def _code_repl(m: re.Match) -> str:
        indent = m.group("indent") or ""
        lang = _normalize_lang(m.group("lang") or "")
        if lang == "m":
            lang = "objc"
        body = m.group("body")
        if indent:
            body = _textwrap.dedent(body).strip("\n")
            body = "\n".join((indent + line) if line else line
                             for line in body.split("\n"))
            return f"\n{indent}```{lang}\n{body}\n{indent}```\n"
        return f"\n```{lang}\n{body.strip()}\n```\n"

    text = re.sub(
        r"^(?P<indent>[ \t]*)@code(?:\{(?P<lang>[^}]*)\})?\s*\n(?P<body>.*?)\n[ \t]*@endcode",
        _code_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 3a. \dot ... \enddot → MyST `{graphviz}` fenced directive. Body is
    #     raw DOT source; the fenced form keeps it out of MyST's smartquotes
    #     and URL-autolink passes.
    def _dot_repl(m: re.Match) -> str:
        indent = m.group("indent") or ""
        body = _textwrap.dedent(m.group("body")).strip("\n")
        if indent:
            body = "\n".join((indent + line) if line else line
                             for line in body.split("\n"))
            return f"\n{indent}```{{graphviz}}\n{body}\n{indent}```\n"
        return f"\n```{{graphviz}}\n{body}\n```\n"
    text = re.sub(
        r"^(?P<indent>[ \t]*)\\dot[ \t]*\n(?P<body>.*?)\n[ \t]*\\enddot[ \t]*$",
        _dot_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 3b. Plain Markdown fences with a Doxygen-flavored language spec
    #     (e.g. "```.sh") confuse Pygments — strip the leading dot and apply
    #     the same alias map as @code{.lang}.
    text = re.sub(
        r"^(?P<fence>`{3,})(?P<lang>\.?[\w-]+)[ \t]*$",
        lambda m: f"{m.group('fence')}{_normalize_lang(m.group('lang'))}",
        text, flags=re.MULTILINE)

    # Plain backtick fence with leading indent applied to every body
    # line so the fence stays inside an enclosing list-item scope.
    # ({code-block} and `:::` colon-fence forms break inside tab-items.)
    def _emit_codeblock(indent: str, lang: str, body: str) -> str:
        body_lines = body.rstrip().splitlines()
        body_indented = "\n".join(indent + line for line in body_lines)
        return f"\n{indent}```{lang}\n{body_indented}\n{indent}```\n"

    # 4. @include path  /  @includelineno path
    #    Indent preserved so code blocks inside list items don't break the list.
    def _include_repl(m: re.Match) -> str:
        indent = m.group("indent")
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

    # 5. @snippet path [Label]
    # Indent preserved so code blocks inside list items don't break the list.
    def _snippet_repl(m: re.Match) -> str:
        indent = m.group("indent")
        code, lang = _read_snippet(m.group("path"), m.group("label"))
        return _emit_codeblock(m.group("indent") or "", lang, code)
    text = re.sub(
        r"^(?P<indent>[ \t]*)@snippet\s+(?P<path>\S+)\s+(?P<label>[^\n]+?)\s*$",
        _snippet_repl, text, flags=re.MULTILINE)

    # 5b. @snippetlineno — same as @snippet with :linenos:.
    def _snippetlineno_repl(m: re.Match) -> str:
        indent = m.group("indent")
        code, lang = _read_snippet(m.group("path"), m.group("label"))
        body = "\n".join(indent + l for l in code.rstrip().split("\n"))
        return f"\n{indent}```{{code-block}} {lang}\n{indent}:linenos:\n{body}\n{indent}```\n"
    text = re.sub(r"^(?P<indent>[ \t]*)@snippetlineno\s+(?P<path>\S+)\s+(?P<label>[^\n]+?)\s*$",
                  _snippetlineno_repl, text, flags=re.MULTILINE)

    # 6. @add_toggle_LANG ... @end_toggle  (coalesce runs into one tab-set)
    #    Capture the leading indent of each toggle block and emit the tab-set
    #    fence lines at the same indent, so toggles inside list items stay as
    #    list-item continuation content (where 4-space fences are valid).
    #    Body content is dedented so code blocks at column 0 inside the
    #    directive body are parsed correctly by myst-parser.
    def _toggle_collapse(src: str) -> str:
        out, i = [], 0
        opener = re.compile(r"^([ \t]*)@add_toggle_(\w+)[ \t]*$", re.MULTILINE)
        while True:
            m = opener.search(src, i)
            if not m:
                out.append(src[i:]); break
            out.append(src[i:m.start()])
            block_ind, tabs, j = m.group(1), [], m.start()
            while True:
                m2 = re.match(
                    r"[ \t]*@add_toggle_(\w+)[ \t]*\n(.*?)\n[ \t]*@end_toggle[ \t]*\n?",
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
            out.append(_emit_toggles(tabs, block_ind))
            i = j
        return "".join(out)
    text = _toggle_collapse(text)

    # 6b. Strip list-item continuation indent stranded after a col-0 tab-set close.
    def _strip_tabset_continuations(src: str) -> str:
        lines = src.split("\n")
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line == "``````":
                out.append(line)
                i += 1
                while i < len(lines):
                    ln = lines[i]
                    if ln.startswith("    "):
                        out.append(ln[4:])
                        i += 1
                    elif not ln.strip():
                        out.append(ln)
                        i += 1
                    else:
                        break
            else:
                out.append(line)
                i += 1
        return "\n".join(out)
    text = _strip_tabset_continuations(text)

    # 7. @ref name [optional "Display Text"]
    def _ref_repl(m: re.Match) -> str:
        name = m.group("name"); disp = m.group("disp")
        target = _ANCHOR_TO_DOC.get(name)
        if target:
            return f"[{disp or name}]({'/' + target})"
        return f"[{disp or name}](#{name})"
    # Names may be qualified C++ identifiers like `cv::saturate_cast`, so
    # the character class allows `:` in addition to word chars and `-`.
    text = re.sub(r'@ref\s+(?P<name>[\w:-]+)(?:\s+"(?P<disp>[^"]+)")?',
                  _ref_repl, text)

    # 8. @cite KEY → [[KEY]](link to docs.opencv.org citelist)
    text = re.sub(
        r"@cite\s+([\w-]+)",
        lambda m: f"[[{m.group(1)}]](https://docs.opencv.org/5.x/d0/de3/citelist.html#CITEREF_{m.group(1)})",
        text)

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

    # 8c. @note / @see / @warning  -> MyST admonitions.  Each directive body runs
    #     until the next blank line, the next @directive at start-of-line, or
    #     end of file (matches Doxygen's paragraph-level semantics).
    _ADMON_KIND = {"note": "note", "see": "seealso", "warning": "warning"}
    def _admon_repl(m: re.Match) -> str:
        kind = _ADMON_KIND[m.group("dir")]
        body = m.group("body").strip()
        return f"\n:::{{{kind}}}\n{body}\n:::\n"
    text = re.sub(
        r"^@(?P<dir>note|see|warning)\s+(?P<body>.+?)(?=\n[ \t]*\n|\n@[A-Za-z]|\Z)",
        _admon_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 8d. Dedent indented description paragraphs after `- @subpage X`
    #     so they render as normal text, not as code blocks.
    def _dedent_subpage_descriptions(src: str) -> str:
        # Accept either 4+ spaces/tabs OR a single leading tab as the
        # continuation indent (one tab = bullet content column in
        # CommonMark — phase_unwrapping uses this).
        pat = re.compile(
            r"^(?P<bullet>[ \t]*-\s+[^\n]*@subpage\s+[\w-]+[^\n]*)\n"
            r"(?P<desc>(?:[ \t]*\n|(?:\t|[ \t]{4,})[^\n]+(?:\n|$))+)",
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            desc = _textwrap.dedent(m.group("desc")).strip("\n")
            # All-blank description (e.g. `- @subpage X\n\n##### Section`):
            # don't rewrite, or we'd accumulate extra blank lines.
            if not desc.strip():
                return m.group(0)
            return f"{m.group('bullet')}\n\n{desc}\n\n"
        return pat.sub(repl, src)
    text = _dedent_subpage_descriptions(text)

    # 9. Bullet `@subpage` lists → real toctree. Enabled modules become
    #    internal entries; disabled ones become external Doxygen links.
    #    Allows any text between `-` and `@subpage` to accept the
    #    `- <module>. @subpage <id>` form used by contrib_root.markdown.
    def _subpage_list_to_toctree(src: str) -> str:
        pat = re.compile(
            r"((?:^[ \t]*-\s+[^\n]*?@subpage\s+[\w-]+(?:[^\n]*)\n)+)",
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            entries = re.findall(r"@subpage\s+([\w-]+)", m.group(1))
            lines = []
            for e in entries:
                if e in _ANCHOR_TO_DOC:
                    lines.append("/" + _ANCHOR_TO_DOC[e])
                elif e in _ANCHOR_TO_EXTERNAL:
                    title, url = _ANCHOR_TO_EXTERNAL[e]
                    lines.append(f"{title} <{url}>")
            if not lines:
                return ""
            body = "\n".join(lines)
            return f"\n```{{toctree}}\n:maxdepth: 1\n\n{body}\n```\n"
        return pat.sub(repl, src)
    text = _subpage_list_to_toctree(text)

    # 10. @next_tutorial / @prev_tutorial  -> drop
    text = re.sub(r"^@(?:next|prev)_tutorial\{[^}]*\}\s*$", "",
                  text, flags=re.MULTILINE)

    # 10b. Doxygen ordered-list marker: "-#" -> "1."
    #      Doxygen uses -# for numbered lists; MyST uses standard 1. notation.
    text = re.sub(r"^([ \t]*)-#([ \t]+)", r"\g<1>1.\g<2>",
                  text, flags=re.MULTILINE)

    # 11. @tableofcontents -> drop (PyData right sidebar replaces it)
    text = re.sub(r"^(?:@tableofcontents|\[TOC\])\s*$", "", text, flags=re.MULTILINE)

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

    # 11g. Wrap bare http(s) URLs in <> for CommonMark autolink.
    def _autolink_repl(m: re.Match) -> str:
        url = m.group(0)
        trail = ""
        while url and url[-1] in ".,;:!?)":
            trail = url[-1] + trail
            url = url[:-1]
        return f"<{url}>{trail}" if url else m.group(0)
    text = re.sub(
        r'(?<!\]\()(?<![<"])https?://\S+',
        _autolink_repl, text)

    # Depth-relative prefix from the current doc back to the site root,
    # used to point `<img src>` at `<output>/contrib_modules/...` files
    # that html_extra_path publishes (Sphinx can't pathto-rewrite URLs
    # for files outside srcdir, so we compute the ../ count ourselves).
    _depth = docname.count("/") if docname else 0
    _contrib_url_prefix = ("../" * _depth) + "contrib_modules/"

    def _emit_contrib_img(rel_url: str, alt: str) -> str:
        src = _contrib_url_prefix + rel_url
        img = f'<img src="{src}" alt="{alt}"/>'
        if alt.startswith("Figure "):
            return (f'<figure>{img}'
                    f'<figcaption>{alt}</figcaption></figure>')
        return img

    # 12. Image paths `images/foo.png`. Try the doc's local `images/`
    #     sibling first, then the global basename index, then a final
    #     well-known fallback dir (mirrors Doxygen flat IMAGE_PATH).
    def _img_repl(m: re.Match) -> str:
        alt, rel = m.group("alt"), m.group("rel")
        if docname:
            parts = pathlib.Path(docname).parent.parts
            local = None
            if parts and parts[0] == "tutorials":
                local = DOC_ROOT / pathlib.Path(docname).parent / "images" / rel
            elif len(parts) >= 2 and parts[0] == "tutorials_contrib":
                # Contrib doc → resolve under <m>/tutorials/<rest>/images/.
                rest = pathlib.Path(*parts[2:]) if len(parts) > 2 else pathlib.Path()
                local = CONTRIB_ROOT / parts[1] / "tutorials" / rest / "images" / rel
            if local is not None and local.is_file():
                return f'![{alt}](images/{rel})'
        hit = _IMAGE_INDEX.get(pathlib.Path(rel).name)
        if hit:
            if hit.startswith("contrib_modules/"):
                return _emit_contrib_img(hit[len("contrib_modules/"):], alt)
            return f'![{alt}](/{hit})'
        return f'![{alt}](/tutorials/others/images/{rel})'
    text = re.sub(
        r'!\[(?P<alt>[^\]]*)\]\((?:[^)]*?/)?images/(?P<rel>[^)]+)\)',
        _img_repl, text)

    # 12b. Cross-tree image refs for contrib pages (Doxygen IMAGE_PATH
    #      flattening): `pics/foo.jpg` → <m>/doc/pics/, `<m>/samples/...`,
    #      etc. Try module-relative bases; first match becomes raw-HTML
    #      <img> with a depth-relative /contrib_modules/<m>/<rest> URL.
    def _img_xtree(m: re.Match) -> str:
        alt, rel = m.group("alt"), m.group("rel")
        if rel.startswith("/") or "://" in rel:
            return m.group(0)
        if rel.startswith("./"):
            rel = rel[2:]
        if not docname or not docname.startswith("tutorials_contrib/"):
            return m.group(0)
        parts = pathlib.Path(docname).parent.parts
        if len(parts) < 2:
            return m.group(0)
        module = parts[1]
        for cand in (f"{module}/doc/{rel}",
                     f"{module}/{rel}",
                     rel):
            if (CONTRIB_ROOT / cand).is_file():
                return _emit_contrib_img(cand, alt)
        return m.group(0)
    text = re.sub(
        r'!\[(?P<alt>[^\]]*)\]\((?P<rel>[^)]+)\)',
        _img_xtree, text)

    # 12d. Force a blank line between consecutive `Label: ![](image)`
    #      lines so each pair becomes its own paragraph (otherwise the
    #      images flow inline). Skip `|`-prefixed table rows.
    text = re.sub(
        r"^(?P<line>(?!\|)[^\n]*!\[[^\]]*\]\([^)]+\)[^\n]*)\n"
        r"(?=(?!\|)[^\n]*!\[[^\]]*\]\([^)]+\))",
        r"\g<line>\n\n", text, flags=re.MULTILINE)

    # 12e. `![Figure N: caption](url)` → MyST `{figure}` directive so the
    #      caption renders visibly (plain image syntax keeps caption only
    #      in alt=). Used by hdf/* tutorials.
    text = re.sub(
        r"^(?P<indent>[ \t]*)!\[(?P<caption>Figure\s[^\]]+)\]\((?P<url>[^)]+)\)\s*$",
        lambda m: (f"{m.group('indent')}:::{{figure}} {m.group('url')}\n"
                   f"{m.group('indent')}{m.group('caption')}\n"
                   f"{m.group('indent')}:::"),
        text, flags=re.MULTILINE)

    # 13. Wrap the Original-author/Compatibility front-matter table
    #     in a `.opencv-meta-table` div so custom.css can style it.
    # 12b. Bare image filenames with no directory prefix (e.g. "psf.png") that
    #      Doxygen resolves via IMAGE_PATH but Sphinx cannot find as-is.
    #      Redirect to images/<name> when the file lives in the doc's own
    #      images/ sibling, otherwise fall back to the global index.
    def _bare_img_repl(m: re.Match) -> str:
        rel = m.group("rel")
        if docname:
            local = DOC_ROOT / pathlib.Path(docname).parent / "images" / rel
            if local.is_file():
                return f'{m.group("pre")}images/{rel})'
        hit = _IMAGE_INDEX.get(rel)
        if hit:
            return f'{m.group("pre")}/{hit})'
        return m.group(0)
    text = re.sub(
        r'(?P<pre>!\[[^\]]*\]\()(?P<rel>[A-Za-z0-9_.-]+\.[A-Za-z]{2,4})\)',
        _bare_img_repl, text)

    # 12c. Standalone ![Caption](path) → {figure} so alt text is a visible caption.
    text = re.sub(
        r"^([ \t]*)!\[(?P<alt>[^\]]+)\]\((?P<path>[^)\n]+)\)[ \t]*$",
        lambda m: (
            f"\n{m.group(1)}```{{figure}} {m.group('path')}\n"
            f"{m.group(1)}{m.group('alt').strip()}\n"
            f"{m.group(1)}```\n"
        ),
        text, flags=re.MULTILINE)

    # 12d. Doxygen ^ rowspan cell → merge into row above via <hr class="cv-rowdiv">.
    def _merge_caret_rows(src: str) -> str:
        lines = src.split("\n")
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if "^" in cells:
                    prev_idx = next(
                        (k for k in range(len(out) - 1, -1, -1)
                         if out[k].strip().startswith("|") and
                            not re.match(r"^\|[\s|:-]+\|$", out[k].strip())),
                        None)
                    if prev_idx is not None:
                        prev_cells = [c.strip() for c in
                                      out[prev_idx].strip().strip("|").split("|")]
                        merged = []
                        for j, cell in enumerate(cells):
                            pv = prev_cells[j] if j < len(prev_cells) else ""
                            if cell == "^":
                                merged.append(pv)
                            else:
                                sep = '<hr class="cv-rowdiv">'
                                merged.append(f"{pv}{sep}{cell}" if pv else cell)
                        out[prev_idx] = "| " + " | ".join(merged) + " |"
                        continue
            out.append(line)
        return "\n".join(out)
    text = _merge_caret_rows(text)

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

    # 13b. Auto-linkify bare URLs (Doxygen default; alternative is the
    #      linkify-it-py package). Skip code blocks/spans, existing
    #      markdown links, existing autolinks, and HTML attributes.
    #      Trailing sentence punctuation is left outside the autolink.
    _fence_open_re = re.compile(r"^[ \t]*(`{3,}|~{3,})")
    _inline_code_re = re.compile(r"`[^`\n]+`")
    _bare_url_re = re.compile(
        r"(?<!\]\()(?<!<)(?<!=\")(?<!=')"
        r"https?://[^\s<>\[\]()\"']+"
    )
    def _wrap_one_url(m: re.Match) -> str:
        u = m.group(0)
        trailing = ""
        while u and u[-1] in ".,;:!?":
            trailing = u[-1] + trailing
            u = u[:-1]
        return f"<{u}>{trailing}" if u else m.group(0)
    def _wrap_outside_inline(line: str) -> str:
        # Split on inline `code` so URLs inside backticks stay untouched.
        parts = _inline_code_re.split(line)
        codes = _inline_code_re.findall(line)
        result = []
        for i, p in enumerate(parts):
            result.append(_bare_url_re.sub(_wrap_one_url, p))
            if i < len(codes):
                result.append(codes[i])
        return "".join(result)
    _autolink_out, _in_fence = [], False
    for _line in text.split("\n"):
        if _fence_open_re.match(_line):
            _in_fence = not _in_fence
            _autolink_out.append(_line)
        elif _in_fence:
            _autolink_out.append(_line)
        else:
            _autolink_out.append(_wrap_outside_inline(_line))
    text = "\n".join(_autolink_out)

    # 14. Restore @verbatim stash (see step 0). Placeholder keys are private-
    #     use-area-safe strings so this is a literal replace.
    for _vk, _vv in _verbatim_stash.items():
        text = text.replace(_vk, _vv)

    return text


def _source_read(app, docname, source):
    # Translate any tutorial doc — the root index, everything under an enabled
    # main module, and (when staged) everything under an enabled contrib module.
    if not (docname.startswith("tutorials/")
            or docname.startswith("tutorials_contrib/")):
        return
    text = source[0]
    # On the master doc, append `- @subpage tutorial_contrib_root` so the
    # contrib site appears in the unified left sidebar without modifying
    # opencv/doc/tutorials/tutorials.markdown on disk.
    if (docname == "tutorials/tutorials"
            and CONTRIB_MODULES
            and "tutorial_contrib_root" in _ANCHOR_TO_DOC):
        text = text.rstrip() + "\n\n- @subpage tutorial_contrib_root\n"
    source[0] = _translate(text, docname)


def setup(app):
    app.connect("source-read", _source_read)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
