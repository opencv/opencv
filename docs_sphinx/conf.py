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
import os as _os, pathlib, sys as _sys

# Sphinx is invoked with config-dir / source-dir separation and does NOT place
# the config directory on sys.path, so make the local conf_helpers package
# importable regardless of how sphinx-build is launched.
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from conf_helpers.state import (
    DOC_ROOT, CONTRIB_ROOT, SPHINX_INPUT_ROOT,
    DOC_MODULES, JS_DOC_MODULES, PY_DOC_MODULES, CONTRIB_MODULES, API_MODULES,
    DOXYGEN_BASE_URL, _doxygen_url, _PATCHED_XML_DIR, HAVE_BREATHE,
)
import conf_helpers.build      # noqa: F401  bib staging + anchor scans + API-stub
#                                            generation + image/snippet indexing.
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
# Availability detection lives in conf_helpers.state; here we just register the
# extension and point breathe at the patched XML tree.
if HAVE_BREATHE:
    extensions.append("breathe")
    breathe_projects = {"opencv": str(_PATCHED_XML_DIR)}
    breathe_default_project = "opencv"
    breathe_default_members = ("members",)

source_suffix = {".md": "markdown", ".markdown": "markdown"}

# Tell Sphinx's C/C++ domain parser to treat OpenCV's compatibility macros
# as identifier attributes (i.e. swallow them silently during parsing).
# Without this, signatures like
#     inline virtual const char *getName() const CV_OVERRIDE
# raise "Invalid C++ declaration: Expected end of definition" because the
# parser sees CV_OVERRIDE as an unknown token after `const`. These macros
# expand to `override` / `noexcept` / `final` / nothing in cvdef.h.
cpp_id_attributes = [
    "CV_OVERRIDE", "CV_FINAL", "CV_NOEXCEPT",
    "CV_NORETURN", "CV_DEPRECATED", "CV_DEPRECATED_EXTERNAL",
    "CV_NODISCARD_STD", "CV_NODISCARD",
    "CV_EXPORTS", "CV_EXPORTS_W",
    "CV_WRAP",
]
c_id_attributes = list(cpp_id_attributes)

# Root tutorial index (lists all modules via @subpage). Stays the master
# regardless of how many modules are in DOC_MODULES.
master_doc = "tutorials/tutorials"

# Source dir is the staged tree (or DOC_ROOT for legacy ad-hoc runs).
# Scope: master + enabled main modules + (optionally) enabled contrib modules.
include_patterns = ["tutorials/tutorials.markdown", "faq.markdown",
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
    # Stubs are generated below (in `_generate_api_stubs()`); the file set is
    # recursive over the Doxygen group hierarchy and unknown at this point,
    # so use a glob. The check happens at Sphinx source-enumeration time —
    # if no files exist, the pattern just matches nothing.
    include_patterns.append("api/**")

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

# Exposed to templates/navbar-nav.html so it can rewrite external_links
# whose URL starts with this base into depth-aware relative paths to the
# local Doxygen output, instead of redirecting users to docs.opencv.org.
html_context = {"doxygen_base_url": DOXYGEN_BASE_URL}

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


def setup(app):
    app.connect("source-read", _source_read)
    app.connect("build-finished", _inline_coll_graphs_on_finish)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
