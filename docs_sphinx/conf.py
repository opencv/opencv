"""Sphinx wrapper for opencv/doc/ (single conf.py, config-dir/source-dir
separation so the legacy tree is never duplicated). Build via CMake:

    cmake --build <build> --target sphinx   # -> <build>/docs_sphinx/html/

opencv/doc/ stays untouched; Doxygen directives are translated to MyST in the
`source-read` hook. Enable tutorial modules by adding their folder names to
DOC_MODULES (state.py); only enabled modules compile, others drop from toctrees.
"""
from __future__ import annotations
import os as _os, pathlib, sys as _sys

# Config dir isn't on sys.path under config/source-dir separation; add it.
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from conf_helpers.state import (
    DOC_ROOT, CONTRIB_ROOT, SPHINX_INPUT_ROOT,
    DOC_MODULES, JS_DOC_MODULES, PY_DOC_MODULES, CONTRIB_MODULES, API_MODULES,
    DOXYGEN_BASE_URL, _doxygen_url, _PATCHED_XML_DIR, HAVE_BREATHE,
    USE_INDEX_LANDING,
)
import conf_helpers.build      # noqa: F401  bib staging, scans, API stubs, indexes.
import conf_helpers.patches    # noqa: F401  Sphinx C++ xref + warning patches.
from conf_helpers.translate import _source_read
from conf_helpers.postprocess import _inline_coll_graphs_on_finish

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

# -- Breathe (Doxygen XML -> Sphinx C++ domain) -----------------------------
if HAVE_BREATHE:
    extensions.append("breathe")
    breathe_projects = {"opencv": str(_PATCHED_XML_DIR)}
    breathe_default_project = "opencv"
    # Empty so class pages stay description-only (members are hand-rolled in
    # _write_class_stub); a global ("members",) would duplicate them and feed
    # macro-bearing decls to the C++ parser. Missing-XML fallback sets it itself.
    breathe_default_members = ()

source_suffix = {".md": "markdown", ".markdown": "markdown"}

# Swallow OpenCV's compatibility macros during C++ parsing, else signatures
# like `... getName() const CV_OVERRIDE` raise "Invalid C++ declaration".
cpp_id_attributes = [
    "CV_OVERRIDE", "CV_FINAL", "CV_NOEXCEPT",
    "CV_NORETURN", "CV_DEPRECATED", "CV_DEPRECATED_EXTERNAL",
    "CV_NODISCARD_STD", "CV_NODISCARD",
    "CV_EXPORTS", "CV_EXPORTS_W",
    "CV_WRAP",
    # Python-binding macros prefixing decls like `CV_PROP_RW Point2f pt`.
    "CV_PROP", "CV_PROP_RW", "CV_PROP_W",
    "CV_OUT", "CV_IN_OUT",
]
c_id_attributes = list(cpp_id_attributes)

# Master doc. By default the generated `index.markdown` landing page is the
# site root (USE_INDEX_LANDING); its toctree lists every cross-family root and
# its body is the OpenCV-modules link list. Setting the flag False falls back
# to the legacy layout where `tutorials/tutorials` is the root.
master_doc = "index" if USE_INDEX_LANDING else "tutorials/tutorials"

# Scope: master + enabled main modules + (optionally) enabled contrib modules.
include_patterns = (["index.markdown"] if USE_INDEX_LANDING else []) + [
                    "tutorials/tutorials.markdown", "faq.markdown",
                    "citelist.markdown", "intro.markdown"] + [
    f"tutorials/{m}/**" for m in DOC_MODULES
] + (["js_tutorials/js_tutorials.markdown"] if JS_DOC_MODULES else []) + [
    f"js_tutorials/{m}/**" for m in JS_DOC_MODULES
] + (["py_tutorials/py_tutorials.markdown"] if PY_DOC_MODULES else []) + [
    f"py_tutorials/{m}/**" for m in PY_DOC_MODULES
]
if CONTRIB_MODULES and (SPHINX_INPUT_ROOT / "tutorials_contrib").is_dir():
    include_patterns.append("tutorials_contrib/contrib_root.markdown")
    include_patterns += [f"tutorials_contrib/{m}/**" for m in CONTRIB_MODULES]
if API_MODULES:
    # Glob: the stub file set (generated later) is unknown here.
    include_patterns.append("api/**")
    # Orphan example pages; without this glob the class-page Examples links 404.
    include_patterns.append("examples/**")

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

# OpenCV's custom LaTeX macros (\vecthree, \cameramatrix, …) — ported from
# doc/mymath.js so MathJax resolves them the same way the Doxygen site does.
mathjax3_config = {
    "loader": {"load": ["[tex]/ams"]},
    "tex": {
        "packages": {"[+]": ["ams"]},
        "macros": {
            "matTT": [r"\[ \left|\begin{array}{ccc} #1 & #2 & #3\\ #4 & #5 & #6\\ #7 & #8 & #9 \end{array}\right| \]", 9],
            "fork": [r"\left\{ \begin{array}{l l} #1 & \mbox{#2}\\ #3 & \mbox{#4}\\ \end{array} \right.", 4],
            "forkthree": [r"\left\{ \begin{array}{l l} #1 & \mbox{#2}\\ #3 & \mbox{#4}\\ #5 & \mbox{#6}\\ \end{array} \right.", 6],
            "forkfour": [r"\left\{ \begin{array}{l l} #1 & \mbox{#2}\\ #3 & \mbox{#4}\\ #5 & \mbox{#6}\\ #7 & \mbox{#8}\\ \end{array} \right.", 8],
            "vecthree": [r"\begin{bmatrix} #1\\ #2\\ #3 \end{bmatrix}", 3],
            "vecthreethree": [r"\begin{bmatrix} #1 & #2 & #3\\ #4 & #5 & #6\\ #7 & #8 & #9 \end{bmatrix}", 9],
            "cameramatrix": [r"#1 = \begin{bmatrix} f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0 & 0 & 1 \end{bmatrix}", 1],
            "distcoeffs": [r"(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]]) \text{ of 4, 5, 8, 12 or 14 elements}"],
            "distcoeffsfisheye": [r"(k_1, k_2, k_3, k_4)"],
            "hdotsfor": [r"\dots", 1],
            "mathbbm": [r"\mathbb{#1}", 1],
            "bordermatrix": [r"\matrix{#1}", 1],
        },
    },
}

suppress_warnings = [
    "myst.header", "myst.xref_missing", "toc.not_included",
    "misc.highlighting_failure",
    "image.not_readable",
    # Same C++ symbol legitimately appears on >1 generated page (group + namespace).
    "cpp.duplicate_declaration",
]

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
    # Show all 7 nav links inline (no "More" dropdown).
    "header_links_before_dropdown": 7,
    # Doxygen-style top nav. Uses the `external_links` slot as the data hook, but
    # navbar-nav.html rewrites each DOXYGEN_BASE_URL target to a local relative
    # path, so these stay on-site (see html_context).
    "external_links": [
        {"url": _doxygen_url("index.html"),       "name": "Main Page"},
        {"url": _doxygen_url("pages.html"),       "name": "Related Pages"},
        {"url": _doxygen_url("namespaces.html"),  "name": "Namespaces"},
        {"url": _doxygen_url("annotated.html"),   "name": "Classes"},
        {"url": _doxygen_url("files.html"),       "name": "Files"},
        {"url": _doxygen_url("examples.html"),    "name": "Examples"},
        {"url": DOXYGEN_BASE_URL + "javadoc/",    "name": "Java Documentation"},
    ],
    # Doxygen search engine replaces the native one; render a single trigger in
    # navbar_end (navbar_persistent renders twice → duplicate element IDs).
    "navbar_persistent": [],
    "navbar_end": ["search-button-field", "theme-switcher", "navbar-icon-links"],
    "disable_search": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": True,
    "show_nav_level": 2,
    "navigation_depth": 4,
    # Every page shows the in-page "On this page" TOC, except the generated
    # landing page (index), where an empty list removes the secondary sidebar
    # entirely so the centered entry list isn't pushed off to the left.
    "secondary_sidebar_items": {"**": ["page-toc"], "index": []},
    "back_to_top_button": True,
    "show_version_warning_banner": False,
    "icon_links": [{"name": "GitHub",
                    "url": "https://github.com/opencv/opencv",
                    "icon": "fa-brands fa-github"}],
}

# Lets navbar-nav.html rewrite DOXYGEN_BASE_URL nav links to local relative paths.
html_context = {"doxygen_base_url": DOXYGEN_BASE_URL}

# Publish each contrib module via a build-dir symlink + html_extra_path (URLs
# /contrib_modules/<m>/..., nothing duplicated in srcdir). Skipped for ad-hoc
# sphinx-build inside a source tree (CMake-only path).
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


def setup(app):
    app.connect("source-read", _source_read)
    app.connect("build-finished", _inline_coll_graphs_on_finish)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
