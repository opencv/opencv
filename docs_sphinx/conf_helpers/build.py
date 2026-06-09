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

# Generated tutorial + TOC link (input tree only; source repo untouched).
_PREBUILT_TUTORIAL_MD = r"""Using OpenCV pre-built binaries in your own projects {#tutorial_using_prebuilt_binaries}
====================================================

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 5.0, C++17, Python >= 3.6 |

@tableofcontents

Goal
----

The objective of this tutorial is to show how to configure a local build environment to use
pre-built OpenCV 5.0 binaries that are already present on your device.

By the end of this guide you will know how to reference, link, and use an existing OpenCV
installation from your own C++ or Python application, without rebuilding the library from
source.

Detailed Description
--------------------

When OpenCV is installed via an installer, a system package manager (apt, Homebrew, vcpkg),
or built into a local workspace directory, it exports configuration scripts that let build
tools locate and link it automatically.

Because OpenCV 5.0 modernizes its build requirements, keep two things in mind:

-   **C++17 is required.** The library headers use modern language features, so your project
    must be compiled with the C++17 standard or higher.
-   **Modern CMake targets.** The legacy 1.x C API has been removed. Link via the
    `${OpenCV_LIBS}` variable from `find_package(OpenCV)`, or by naming the specific
    libraries on the compiler command line.

C++ Project Configuration
-------------------------

You can link against your pre-built binaries with CMake (recommended for cross-platform
stability) or by invoking g++ directly.

### Method 1: Using CMake (recommended)

#### 1. File layout

@code{.unparsed}
my_opencv_project/
├── CMakeLists.txt
└── main.cpp
@endcode

#### 2. CMakeLists.txt

@code{.cmake}
cmake_minimum_required(VERSION 3.22)
project(OpenCV5_Local_Project)

# OpenCV 5.0 requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate the pre-built OpenCV installation. If CMake cannot find it
# automatically, pass the path explicitly:
#   cmake -DOpenCV_DIR=/absolute/path/to/opencv/build/ ..
find_package(OpenCV 5.0 REQUIRED)

message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")

add_executable(opencv_test_app main.cpp)
target_link_libraries(opencv_test_app PRIVATE ${OpenCV_LIBS})
@endcode

#### 3. Build and run

@code{.bash}
cd path/to/my_opencv_project
mkdir build && cd build
cmake ..
cmake --build .
./opencv_test_app
@endcode

### Method 2: Direct compilation with g++

On Linux or macOS you can compile with a single command, without generating build files.

#### 1. Locate your paths

-   Standard global location: `/usr/local/include/opencv5` and `/usr/local/lib`
-   Custom build location: `/path/to/opencv/build/include` and `/path/to/opencv/build/lib`

#### 2. Using pkg-config

@code{.bash}
g++ -std=c++17 main.cpp -o opencv_test_app $(pkg-config --cflags --libs opencv5)
@endcode

#### 3. Explicit paths (custom or local builds)

@code{.bash}
g++ -std=c++17 main.cpp -o opencv_test_app \
    -I/usr/local/include/opencv5 \
    -L/usr/local/lib \
    -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
@endcode

@note In OpenCV 5.0 the legacy `calib3d` module was split into four modules: `geometry`,
`calib`, `stereo`, and `ptcloud`. Link the specific one(s) you need, e.g. `-lopencv_geometry`,
`-lopencv_calib`, `-lopencv_stereo`, or `-lopencv_ptcloud`, instead of `-lopencv_calib3d`.

### Source application

Save the following as `main.cpp`:

@code{.cpp}
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    cv::Mat image = cv::imread("lena.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }
    cv::Mat processed_image;
    cv::GaussianBlur(image, processed_image, cv::Size(7, 7), 1.5, 1.5);
    cv::imshow("Original", image);
    cv::imshow("Processed", processed_image);
    cv::waitKey(0);
    return 0;
}
@endcode

Python Project Configuration
----------------------------

If your device already has the pre-compiled Python bindings (`cv2`), you can use them directly.

#### 1. Verify the binding

@code{.bash}
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
@endcode

#### 2. Test script

Save the following as `app.py`:

@code{.python}
import cv2
import sys

img = cv2.imread('lena.jpg')
if img is None:
    sys.exit("Error: image file missing or path invalid.")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Color', img)
cv2.imshow('Grayscale', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
@endcode

Troubleshooting
---------------

-   **CMake: "Could not find a package configuration file ..."** &mdash; CMake cannot locate
    the install. Pass the path explicitly: `cmake -DOpenCV_DIR=/path/to/opencv/build/ ..`
-   **Compiler: "OpenCV 5.0 requires C++17"** &mdash; the toolchain fell back to an older
    standard. Add `-std=c++17` to the g++ command, or `set(CMAKE_CXX_STANDARD 17)` before
    `find_package` in CMake.
"""

if SPHINX_INPUT_ROOT != DOC_ROOT:
    try:
        _intro = SPHINX_INPUT_ROOT / "tutorials" / "introduction"
        if _intro.is_dir():
            _tdir = _intro / "using_prebuilt_binaries"
            _tdir.mkdir(parents=True, exist_ok=True)
            (_tdir / "using_prebuilt_binaries.markdown").write_text(
                _PREBUILT_TUTORIAL_MD, encoding="utf-8")
            _toc = _intro / "table_of_content_introduction.markdown"
            if _toc.is_file() and "tutorial_using_prebuilt_binaries" not in (
                    _toc_text := _toc.read_text(encoding="utf-8", errors="ignore")):
                _toc_text = _toc_text.replace(
                    "##### Usage basics\n",
                    "##### Usage basics\n-   @subpage tutorial_using_prebuilt_binaries - "
                    "Use an existing OpenCV 5 install in your own C++/Python project\n", 1)
                if _toc.is_symlink():
                    _toc.unlink()
                _toc.write_text(_toc_text, encoding="utf-8")
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

# Doxygen's IMAGE_PATH also spans opencv/samples (+ apps), so a tutorial can
# reference an image that lives only under samples — e.g. the Clojure tutorial's
# `![](images/lena.png)`, whose file is opencv/samples/java/clojure/.../lena.png.
# Those resolve to nothing in the tutorial-only index and render broken. Mirror
# Doxygen, but bounded: index+stage ONLY sample images that a tutorial actually
# references and that aren't already provided by a tutorial `images/` dir, so we
# don't copy the whole samples image set. Staged under sample_pics/ like the
# api_pics mechanism above.
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
