# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

from __future__ import annotations
import os as _os, pathlib, sys as _sys

# Config dir isn't on sys.path under config/source-dir separation; add it.
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from conf_helpers.state import (
    DOC_ROOT, CONTRIB_ROOT, SPHINX_INPUT_ROOT,
    DOC_MODULES, JS_DOC_MODULES, PY_DOC_MODULES, CONTRIB_MODULES, API_MODULES,
    DOXYGEN_BASE_URL, _PATCHED_XML_DIR, HAVE_BREATHE,
    USE_INDEX_LANDING,
)
import conf_helpers.build      # noqa: F401  bib staging, scans, API stubs, indexes.
import conf_helpers.patches    # noqa: F401  Sphinx C++ xref + warning patches.
from conf_helpers.translate import _source_read
from conf_helpers.postprocess import _inline_coll_graphs_on_finish

# -- Project ----------------------------------------------------------------
project = "OpenCV"
author = "OpenCV Team"
release = "5.0"

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

master_doc = "index" if USE_INDEX_LANDING else "tutorials/tutorials"

# Scope: master + enabled main modules + (optionally) enabled contrib modules.
include_patterns = (["index.markdown"] if USE_INDEX_LANDING else []) + [
                    "tutorials/tutorials.markdown", "faq.markdown",
                    "citelist.markdown", "intro.markdown",
                    "related_pages.markdown", "namespace_list.markdown",
                    "class_list.markdown"] + [
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
    include_patterns.append("main_modules/**")
    include_patterns.append("extra_modules/**")
    # Orphan example pages; without this glob the class-page Examples links 404.
    include_patterns.append("examples/**")

exclude_patterns = [
    "**/Thumbs.db", "**/.DS_Store", "**/_old/**",
    "tutorials/core/how_to_use_OpenCV_parallel_for_/**",
    "tutorials/introduction/load_save_image/**",
    "tutorials/app/_old/**",
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
    "logo": {
        "text": f"OpenCV {release}",
        "image_light": "_static/opencv-logo.svg",
        "image_dark": "_static/opencv-logo.svg",
    },
    # Navbar layout: logo on the left, version switcher right beside it —
    # mirrors the legacy docs.opencv.org header where the version selector
    # sits inline with the wordmark. The switcher template reads the shared
    # /version.js (window.OPENCV_DOC_VERSIONS); no build-time list is generated.
    "navbar_start": ["navbar-logo", "opencv-version-switcher"],
    "header_links_before_dropdown": 6,
    "external_links": [
        {"docname": master_doc,                "name": "Main Page"},
        {"docname": "related_pages",           "name": "Related Pages"},
        {"docname": "namespace_list",          "name": "Namespaces"},
        {"docname": "class_list",              "name": "Classes"},
        {"docname": "examples/examples_root",  "name": "Examples"},
        {"url": DOXYGEN_BASE_URL + "javadoc/", "name": "Java documentation",
         "external": True},
    ],
    "navbar_persistent": [],
    "navbar_end": ["search-button-field", "theme-switcher", "navbar-icon-links"],
    "disable_search": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": True,
    # 1, not 2: the theme force-opens every <details> up to this level and
    # modules render at toctree-l1, so 2 would expand all of them; 1 opens only
    # the current module's branch. Render depth is unaffected (navigation_depth).
    "show_nav_level": 1,
    "navigation_depth": 4,
    "secondary_sidebar_items": {"**": ["page-toc"], "index": ["page-toc", "resources"]},
    "back_to_top_button": True,
    "show_version_warning_banner": False,
    "icon_links": [{"name": "GitHub",
                    "url": "https://github.com/opencv/opencv",
                    "icon": "fa-brands fa-github"}],
}

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
    conf_helpers.patches.register_global_sidebar(app)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
