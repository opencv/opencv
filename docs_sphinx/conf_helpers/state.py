# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""Shared configuration & state for the OpenCV Sphinx wrapper."""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap

HERE = pathlib.Path(__file__).resolve().parent.parent
DOC_ROOT = (HERE.parent / "doc").resolve()
OPENCV_ROOT = HERE.parent.resolve()

# SCOPE — tutorial module folders; env override or auto-discover per tree
import os as _os

def _discover_doc_modules(subdir: str, toc_glob: str) -> list[str]:
    """Folder names under DOC_ROOT/<subdir> that carry a TOC page (sorted)."""
    root = DOC_ROOT / subdir
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and any(p.glob(toc_glob)))

def _module_list(env_var: str, subdir: str, toc_glob: str) -> list[str]:
    """Env-var override (comma list) when set & non-empty, else auto-discover."""
    val = _os.environ.get(env_var)
    if val:
        return [m.strip() for m in val.split(",") if m.strip()]
    return _discover_doc_modules(subdir, toc_glob)

DOC_MODULES = _module_list(
    "OPENCV_DOC_MODULES", "tutorials", "table_of_content_*.markdown")
JS_DOC_MODULES = _module_list(
    "OPENCV_JS_DOC_MODULES", "js_tutorials", "js_table_of_contents_*.markdown")
PY_DOC_MODULES = _module_list(
    "OPENCV_PY_DOC_MODULES", "py_tutorials", "py_table_of_contents_*.markdown")

# SCOPE — env OPENCV_CONTRIB_MODULES; empty/unset auto-discovers
CONTRIB_ROOT = pathlib.Path(
    _os.environ.get("OPENCV_CONTRIB_ROOT")
    or str(HERE.parent.parent / "opencv_contrib" / "modules")
).resolve()

def _discover_contrib_modules() -> list[str]:
    """Contrib module folders carrying a `tutorials/` subtree (CMake's gate)."""
    if not CONTRIB_ROOT.is_dir():
        return []
    return sorted(p.name for p in CONTRIB_ROOT.iterdir()
                  if p.is_dir() and (p / "tutorials").is_dir())

_contrib_env = _os.environ.get("OPENCV_CONTRIB_MODULES")
CONTRIB_MODULES = ([m.strip() for m in _contrib_env.split(",") if m.strip()]
                   if _contrib_env else _discover_contrib_modules())

# SCOPE — env OPENCV_API_MODULES (comma/semicolon); empty disables API pages
def _discover_api_modules() -> list[str]:
    """Main + contrib modules that declare an `@defgroup`, by module dir name.

    The `@defgroup` is not always in the umbrella `opencv2/<module>.hpp`: some
    modules declare it in a sub-header (datasets/dataset.hpp,
    quality/qualitybase.hpp, reg/map.hpp) or a differently-named top header
    (dnn_objdetect's core_detect.hpp, cannops' cann.hpp). So scan every header
    under each module's include/ tree, and use the module DIRECTORY name (which
    equals the top `@defgroup` / `group__<name>` by convention) as the id."""
    found: set[str] = set()
    _roots = [OPENCV_ROOT / "modules"]
    if CONTRIB_ROOT.is_dir():
        _roots.append(CONTRIB_ROOT)

    def _has_defgroup(hdr) -> bool:
        try:
            return "@defgroup" in hdr.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False

    for _root in _roots:
        if not _root.is_dir():
            continue
        for _mod in _root.iterdir():
            _inc = _mod / "include" / "opencv2"
            if not _inc.is_dir():
                continue
            # Umbrella / top-level headers first (cheap, the common case)…
            if any(_has_defgroup(h) for h in sorted(_inc.glob("*.hpp"))):
                found.add(_mod.name)
                continue
            # …else fall back to sub-directory headers.
            for _h in _inc.rglob("*.hpp"):
                if _has_defgroup(_h):
                    found.add(_mod.name)
                    break
    return sorted(found)


# Default = discovered full set; override (comma/semicolon), empty disables
API_MODULES = [
    m.strip()
    for m in re.split(r"[,;]", _os.environ.get("OPENCV_API_MODULES")
                      or ",".join(_discover_api_modules()))
    if m.strip()
]

# Sphinx srcdir; env OPENCV_SPHINX_INPUT_ROOT (`or` treats empty as unset)
SPHINX_INPUT_ROOT = pathlib.Path(
    _os.environ.get("OPENCV_SPHINX_INPUT_ROOT") or str(DOC_ROOT)
).resolve()

# sphinx_design availability
try:
    import sphinx_design as _sphinx_design  # noqa: F401
    HAVE_SPHINX_DESIGN = True
except ImportError:
    HAVE_SPHINX_DESIGN = False

# -- Breathe (Doxygen XML -> Sphinx C++ domain) -----------------------------
# env OPENCV_DOXYGEN_XML_DIR
_API_XML_DIR = pathlib.Path(
    _os.environ.get("OPENCV_DOXYGEN_XML_DIR")
    or str(HERE.parent.parent / "build_doc" / "doc" / "doxygen" / "xml")
).resolve()
# Patched namespace XML for breathe; see _patch_namespace_xml_for_breathe
_PATCHED_XML_DIR = _API_XML_DIR.parent / "xml_for_sphinx"


def _module_group_stem(m: str) -> str:
    # Doxygen names a group after its @defgroup, which can differ from the
    # module folder (folder `3d` -> `@defgroup _3d`). Return that real stem.
    if (_API_XML_DIR / f"group__{m.replace('_', '__')}.xml").is_file():
        return m
    for _root in (OPENCV_ROOT / "modules", CONTRIB_ROOT / "modules", CONTRIB_ROOT):
        _hdr = _root / m / "include" / "opencv2" / f"{m}.hpp"
        if _hdr.is_file():
            _mm = re.search(r"[@\\]defgroup\s+(\S+)",
                            _hdr.read_text(encoding="utf-8", errors="ignore"))
            return _mm.group(1) if _mm else m
    return m

# -- Python enum/constant signatures ----------------------------------------
# C++ enumerator FQN -> cv2.* name; env OPENCV_PYTHON_SIGNATURES_FILE
_PY_SIGNATURES: dict = {}
import json as _json
for _pysigs_candidate in (
    _API_XML_DIR.parents[2] / "modules" / "python_bindings_generator"
        / "pyopencv_signatures.json",
    _os.environ.get("OPENCV_PYTHON_SIGNATURES_FILE") or None,
):
    if not _pysigs_candidate:
        continue
    _pysigs_path = pathlib.Path(str(_pysigs_candidate))
    if _pysigs_path.is_file():
        try:
            _PY_SIGNATURES = _json.loads(_pysigs_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            _PY_SIGNATURES = {}
        break


def _python_enum_name(enum_qualified: str, value_name: str,
                      strong: bool) -> str | None:
    """Return the cv2.* Python name for one C++ enumerator, or None."""
    if not _PY_SIGNATURES:
        return None
    if strong:
        scope = enum_qualified
    elif "::" in enum_qualified:
        scope = enum_qualified.rsplit("::", 1)[0]
    else:
        scope = ""
    cpp_key = f"{scope}::{value_name}" if scope else value_name
    entries = _PY_SIGNATURES.get(cpp_key)
    if entries:
        return entries[0].get("name")
    return None


# -- Breathe availability ----------------------------------------------------
# Absence empties API_MODULES
HAVE_BREATHE = False
if API_MODULES:
    try:
        import breathe  # noqa: F401
        HAVE_BREATHE = True
    except ImportError:
        API_MODULES = []

# -- Doxygen integration -----------------------------------------------------
DOXYGEN_BASE_URL = (
    _os.environ.get("OPENCV_DOXYGEN_BASE_URL", "https://docs.opencv.org/5.x/")
    .rstrip("/") + "/")
# First existing wins; env OPENCV_DOXYGEN_TAGFILE overrides
_TAG_CANDIDATES = (
    # Track cmake's actual build dir via the XML dir.
    _API_XML_DIR.parent / "html" / "opencv.tag",
    _API_XML_DIR.parent / "opencv.tag",
    HERE.parent / "build" / "doc" / "doxygen" / "html" / "opencv.tag",
    HERE.parent.parent / "build" / "doc" / "doxygen" / "html" / "opencv.tag",
    # extra build-dir layouts (vanilla, contrib, nested CI)
    HERE.parent.parent / "build" / "doc" / "opencv.tag",
    HERE.parent.parent / "build_contrib" / "doc" / "doxygen" / "html" / "opencv.tag",
    HERE.parent.parent / "build_contrib" / "doc" / "opencv.tag",
    HERE.parent.parent / "build" / "build_contrib" / "build_contrib"
        / "doc" / "doxygen" / "html" / "opencv.tag",
)
_TAG_FILE = pathlib.Path(_os.environ.get(
    "OPENCV_DOXYGEN_TAGFILE",
    str(next((p for p in _TAG_CANDIDATES if p.is_file()), _TAG_CANDIDATES[0])),
))

# anchor -> doxygen URL filename
_TAG_FILENAMES: dict[str, str] = {}
# anchor -> title
_TAG_TITLES: dict[str, str] = {}
# Page-only subset of _TAG_TITLES (kind="page" compounds only — NOT groups/files),
# i.e. Doxygen's "Related Pages" set: \page docs like intro, faq, cuda_intro, …
_DOC_PAGE_TITLES: dict[str, str] = {}
# (compound-stem, member-name, normalized-args) -> Doxygen HTML anchor.
# Bridges our XML-driven members to the HTML anchors that name the call/caller
# graph SVGs (XML memberdef ids and HTML anchors live in disjoint hash spaces).
_CALL_GRAPH_ANCHORS: dict[tuple[str, str, str], str] = {}
# Doxygen member anchor -> Sphinx anchor (lowercased member name). Lets diagram
# cross-links jump to the exact member, as the original Doxygen pages did.
_DOXY_ANCHOR_TO_MEMBER: dict[str, str] = {}


def _norm_args(arglist: str) -> str:
    """Normalize a C++ arg-list so an XML `argsstring` matches a tag `arglist`."""
    import html as _html
    return re.sub(r"\s+", "", _html.unescape(arglist or ""))
# cv-namespace short-name -> doxygen URL; used by step 7c
_CV_SYMBOL_URL: dict[str, str] = {}
# include-path -> Doxygen HTML file URL; linkifies enum `#include` lines.
_FILE_URL: dict[str, str] = {}
if _TAG_FILE.is_file():
    try:
        import xml.etree.ElementTree as _ET
        _tag_root = _ET.parse(str(_TAG_FILE)).getroot()
        for _c in _tag_root.iter("compound"):
            _kind = _c.get("kind")
            # Call/caller-graph anchors: every compound's function members,
            # keyed by the page they're documented on (the SVG filename prefix).
            for _fm in _c.findall("member"):
                if _fm.get("kind") != "function":
                    continue
                _fn = _fm.findtext("name")
                _faf = _fm.findtext("anchorfile") or ""
                _fan = _fm.findtext("anchor") or ""
                if not (_fn and _faf and _fan):
                    continue
                _fstem = pathlib.Path(_faf).stem
                _CALL_GRAPH_ANCHORS.setdefault(
                    (_fstem, _fn, _norm_args(_fm.findtext("arglist") or "")), _fan)
                # Doxygen anchor -> Sphinx member anchor (lowercased name).
                _DOXY_ANCHOR_TO_MEMBER.setdefault(_fan, _fn.lower())
            if _kind == "page":
                _n, _f = _c.findtext("name"), _c.findtext("filename")
                _t = _c.findtext("title")
                if _n and _f:
                    _TAG_FILENAMES[_n] = _f if _f.endswith(".html") else _f + ".html"
                if _n and _t:
                    _TAG_TITLES[_n] = _t
                    _DOC_PAGE_TITLES[_n] = _t
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
                # Skip G-API/detail internals; a bare cv::Point isn't gapi::own::Point.
                if _full.startswith("cv::") and not any(
                        _x in _full for _x in ("::own::", "::wip::", "::detail::")):
                    _short = _full.split("::")[-1]
                    _af = _c.findtext("filename")
                    # Prefer canonical cv::<short> over nested homonyms.
                    if _short and _af and (
                            _short not in _CV_SYMBOL_URL or _full == f"cv::{_short}"):
                        _CV_SYMBOL_URL[_short] = DOXYGEN_BASE_URL + _af
            elif _kind == "group":
                # module pages are kind="group"
                _n = _c.findtext("name")
                _f = _c.findtext("filename")
                _t = _c.findtext("title")
                if _n and _f:
                    _TAG_FILENAMES[_n] = _f if _f.endswith(".html") else _f + ".html"
                if _n and _t:
                    _TAG_TITLES[_n] = _t
            elif _kind == "file":
                # Header file -> Doxygen page; key by include path.
                _n = _c.findtext("name") or ""
                _p = _c.findtext("path") or ""
                _f = _c.findtext("filename") or ""
                if _n and _f and not _p.startswith("/"):
                    _key = (_p + _n) if _p.endswith("/") or not _p else f"{_p}/{_n}"
                    _FILE_URL.setdefault(
                        _key, _f if _f.endswith(".html") else _f + ".html")
            else:
                # CV_* macros (kind="define") re-exported as cv.CV_*
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

# Canonical public header for each optional module that may be absent from a
# build's Doxygen group XML. Ensure a `_FILE_URL` entry exists so the umbrella
# `#include` on the module's placeholder page resolves (and a file-reference
# page is generated for it). `setdefault` → a real tag entry always wins.
_FALLBACK_MODULE_HEADERS: dict[str, str] = {
    "datasets":      "opencv2/datasets/dataset.hpp",
    "dnn_objdetect": "opencv2/core_detect.hpp",
    "quality":       "opencv2/quality.hpp",
    "reg":           "opencv2/reg/map.hpp",
    "cannops":       "opencv2/cann.hpp",
}
# Every header path referenced by a fallback page (umbrella + per-function
# includes), ensured present in `_FILE_URL` so each `#include` resolves to a
# generated file-reference page. `setdefault` → real tag entries always win.
_FALLBACK_FILE_HEADERS: list[str] = [
    "opencv2/datasets/dataset.hpp", "opencv2/datasets/util.hpp",
    "opencv2/core_detect.hpp",
    "opencv2/quality.hpp", "opencv2/quality/qualitybase.hpp",
    "opencv2/reg/map.hpp", "opencv2/reg/mapper.hpp",
    "opencv2/cann.hpp", "opencv2/cann_interface.hpp",
]
for _inc in list(_FALLBACK_MODULE_HEADERS.values()) + _FALLBACK_FILE_HEADERS:
    # Doxygen file-page stem: `.`->`_8`, `_`->`__` on the bare filename.
    _fn = _inc.rsplit("/", 1)[-1].replace("_", "__").replace(".", "_8")
    _FILE_URL.setdefault(_inc, f"{_fn}.html")

def _doxygen_url(page: str) -> str:
    return DOXYGEN_BASE_URL + _TAG_FILENAMES.get(page, page)


# -- Live (docs.opencv.org) tagfile for API stub URL construction ----------
# CREATE_SUBDIRS=NO flat names 404 live; live tag has subdir prefixes
_LIVE_TAG_FILE = pathlib.Path(_os.environ.get(
    "OPENCV_DOXYGEN_LIVE_TAGFILE",
    str(HERE.parent.parent / "build" / "doc" / "doxygen" / "opencv-live.tag"),
))
if not _LIVE_TAG_FILE.is_file():
    for _alt in (
        HERE.parent.parent / "build" / "build_contrib" / "build_contrib"
            / "doc" / "doxygen" / "opencv-live.tag",
        HERE.parent.parent / "build_contrib" / "doc" / "doxygen" / "opencv-live.tag",
    ):
        if _alt.is_file():
            _LIVE_TAG_FILE = _alt
            break

_LIVE_GROUP_URL: dict[str, str] = {}    # 'group__core__basic' -> live URL
_LIVE_CLASS_URL: dict[str, str] = {}    # 'Matx' -> live URL
_LIVE_TYPEDEF_URL: dict[str, str] = {}  # 'uchar' -> live URL (group anchor)
if _LIVE_TAG_FILE.is_file():
    try:
        import xml.etree.ElementTree as _ET
        for _c in _ET.parse(str(_LIVE_TAG_FILE)).getroot().iter("compound"):
            _kind = _c.get("kind")
            _n = _c.findtext("name") or ""
            _f = _c.findtext("filename") or ""
            if not (_n and _f):
                continue
            _fn = _f if _f.endswith(".html") else _f + ".html"
            if _kind == "group":
                # key by filename basename
                _basename = pathlib.PurePosixPath(_fn).name[:-5]  # strip .html
                _LIVE_GROUP_URL[_basename] = DOXYGEN_BASE_URL + _fn
            elif _kind == "class":
                _short = _n.split("::")[-1]
                _LIVE_CLASS_URL.setdefault(_short, DOXYGEN_BASE_URL + _fn)
            # typedef members -> live anchor URLs
            for _mem in _c.findall("member"):
                if _mem.get("kind") != "typedef":
                    continue
                _mn = (_mem.findtext("name") or "").strip()
                _maf = (_mem.findtext("anchorfile") or "").strip()
                _man = (_mem.findtext("anchor") or "").strip()
                if _mn and _maf and _man:
                    _LIVE_TYPEDEF_URL.setdefault(
                        _mn, f"{DOXYGEN_BASE_URL}{_maf}#{_man}")
    except Exception:
        pass


# -- Local-link variants of the maps above (step 8g) ------------------------
_LOCAL_SRC_TAG = _TAG_FILE if _TAG_FILE.is_file() else _LIVE_TAG_FILE
_LOCAL_CLASS_URL: dict[str, str] = {
    # _Tp template-parameter placeholder stub
    "_Tp": "class_Tp.html",
}
_LOCAL_TYPEDEF_URL: dict[str, str] = {}  # 'uchar' -> 'core_hal_interface.html#_CPPv45uchar'
# Doxygen template-param placeholder pages -> (display name, Sphinx page).
# Sphinx mirrors these as stubs (stubs._write_placeholder_stubs) so diagram
# cross-links resolve instead of 404ing.
_PLACEHOLDER_STUBS: dict[str, tuple[str, str]] = {
    "class__Tp.html":        ("_Tp",        "class_Tp.html"),
    "classfloat__type.html": ("float_type", "classfloat_type.html"),
}
# Doxygen compound filename -> Sphinx page filename (class/struct compounds).
_LOCAL_PAGE_BY_DOXY_FILE: dict[str, str] = {
    _doxy: _page for _doxy, (_disp, _page) in _PLACEHOLDER_STUBS.items()}
_cls_cands: dict[str, set] = {}  # short name -> set of full names (collision check)
if _LOCAL_SRC_TAG.is_file():
    try:
        import xml.etree.ElementTree as _ET
        for _c in _ET.parse(str(_LOCAL_SRC_TAG)).getroot().iter("compound"):
            if _c.get("kind") in ("class", "struct"):
                _n = _c.findtext("name") or ""
                _f = _c.findtext("filename") or ""
                if _n and _f:
                    _short = _n.split("::")[-1]
                    _fn = _f if _f.endswith(".html") else _f + ".html"
                    _doxy_base = pathlib.PurePosixPath(_fn).name
                    # Skip G-API/detail internals as bare `cv::` targets.
                    if not any(_x in _n for _x in ("::own::", "::wip::", "::detail::")):
                        _cls_cands.setdefault(_short, set()).add(_n)
                        # Prefer canonical cv::<short> over nested homonyms.
                        if _short not in _LOCAL_CLASS_URL or _n == f"cv::{_short}":
                            _LOCAL_CLASS_URL[_short] = _doxy_base
                    # Sphinx mirrors Doxygen's filename except the few remapped.
                    _LOCAL_PAGE_BY_DOXY_FILE.setdefault(
                        _doxy_base, _LOCAL_CLASS_URL.get(_short, _doxy_base))
            for _mem in _c.findall("member"):
                # variable only from namespaces; class-member vars poison the map
                _mk = _mem.get("kind")
                if _mk == "typedef":
                    pass
                elif _mk == "enumeration":
                    pass   # enum types like cv::DataLayout — linkable
                elif _mk == "variable" and _c.get("kind") == "namespace":
                    pass
                else:
                    continue
                _mn = (_mem.findtext("name") or "").strip()
                _maf = (_mem.findtext("anchorfile") or "").strip()
                if not (_mn and _maf):
                    continue
                if _mn in _LOCAL_TYPEDEF_URL:
                    continue   # first-occurrence wins
                _bn = pathlib.PurePosixPath(_maf).name
                if _bn.startswith("group__"):
                    # group__core__hal__interface.html -> core_hal_interface.html
                    _local_page = (_bn[len("group__"):]
                                   .replace(".html", "")
                                   .replace("__", "_")
                                   + ".html")
                elif _bn.startswith("namespacecv"):
                    _local_page = "core_basic.html"
                else:
                    _local_page = _bn
                # HAL typedefs are global C; else cv::-scoped (cpp-domain v4 anchor)
                if "hal_interface" in _local_page:
                    _anchor = f"_CPPv4{len(_mn)}{_mn}"
                else:
                    _anchor = f"_CPPv4N2cv{len(_mn)}{_mn}E"
                _LOCAL_TYPEDEF_URL[_mn] = f"{_local_page}#{_anchor}"
    except Exception:
        pass
    # Drop ambiguous short names (>1 class, no canonical cv::<short>) so a bare
    # `cv::Point` stays plain text instead of linking to a wrong homonym.
    for _s, _cands in _cls_cands.items():
        if len(_cands) > 1 and f"cv::{_s}" not in _cands:
            _LOCAL_CLASS_URL.pop(_s, None)


def _doxy_page_to_local(basename: str) -> str:
    """Map a Doxygen compound page filename to the Sphinx page for the same
    symbol. Pure name transform — the caller verifies the file exists.
        group__core__utils.html -> core_utils.html
        namespacecv*.html       -> core_basic.html
        class*/struct*/union*   -> unchanged (or remapped, e.g. the _Tp stub)
    """
    if basename.startswith("group__"):
        return (basename[len("group__"):].replace(".html", "")
                .replace("__", "_") + ".html")
    if basename.startswith("namespace"):
        return "core_basic.html"
    return _LOCAL_PAGE_BY_DOXY_FILE.get(basename, basename)


# -- Class template-parameter display map (step 8e) -------------------------
# class short name -> template-param list e.g. '< _Tp, cn >'
_CLASS_TEMPLATE_DISPLAY: dict[str, str] = {}
if _API_XML_DIR.is_dir():
    try:
        import xml.etree.ElementTree as _ET
        for _xml in _API_XML_DIR.glob("classcv_1_1*.xml"):
            try:
                _cd = _ET.parse(_xml).getroot().find("compounddef")
            except _ET.ParseError:
                continue
            if _cd is None:
                continue
            _tpl = _cd.find("templateparamlist")
            if _tpl is None:
                continue
            _names = []
            for _p in _tpl.findall("param"):
                _decl = (_p.findtext("declname")
                         or _p.findtext("defname") or "").strip()
                _type = (_p.findtext("type") or "").strip()
                if _decl:
                    _names.append(_decl)
                elif _type in ("typename", "class"):
                    _names.append("_Tp")
                elif _type:
                    _names.append(_type)
            if _names:
                _name = (_cd.findtext("compoundname") or "").split("::")[-1]
                _CLASS_TEMPLATE_DISPLAY[_name] = f"< {', '.join(_names)} >"
    except Exception:
        pass


# Punctuation -> alpha token so operator overloads get distinct slugs
_FUNC_SLUG_PUNCT = {
    "=": "eq", "!": "ne", "<": "lt", ">": "gt", "+": "plus", "-": "minus",
    "*": "mul", "/": "div", "&": "amp", "|": "or", "%": "mod", "^": "xor",
    "~": "tilde", "[": "lbr", "]": "rbr",
}


def _func_slug(name: str) -> str:
    """In-page anchor slug for a function name; shared by stub gen and translator."""
    parts = []
    for ch in name.lower():
        if ch.isalnum() or ch == "_":
            parts.append(ch)
        elif ch in _FUNC_SLUG_PUNCT:
            parts.append("-" + _FUNC_SLUG_PUNCT[ch])
        else:
            parts.append("-")
    s = re.sub(r"-+", "-", "".join(parts)).strip("-")
    return f"cv-{s}" if s else "cv"


# ---- Citation numbering --------------------------------------------------
# @cite KEY -> [N], N = bibtex-plain sort position
def _bib_parse(text: str) -> list[dict]:
    """Walk a BibTeX file into a list of {_type, _key, field: value, ...}.

    Tolerant of a missing closing `}` (as bibtex/Doxygen are): a new
    `@type{key,` at the start of a line implicitly closes the previous entry,
    so one malformed record can't swallow the rest of the file. (e.g.
    ximgproc.bib's `akinlar201782` has no closing brace upstream.)"""
    out: list[dict] = []
    n = len(text)
    header = re.compile(r"@(\w+)\s*\{\s*([^\s,]+)\s*,")
    # Every entry header that begins a line is a hard boundary.
    starts = [m.start() for m in
              re.finditer(r"(?m)^[^\S\n]*@\w+\s*\{\s*[^\s,]+\s*,", text)]
    for idx, s in enumerate(starts):
        m = header.match(text, s)
        if not m:
            continue
        kind, key = m.group(1), m.group(2)
        body_start = m.end()
        limit = starts[idx + 1] if idx + 1 < len(starts) else n
        # Matching close within this entry's span; if unbalanced (missing `}`),
        # fall back to the next entry's boundary, trimming a trailing brace.
        depth, i, end = 1, body_start, None
        while i < limit:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        body = (text[body_start:end] if end is not None
                else text[body_start:limit].rstrip().rstrip("}"))
        out.append({"_type": kind.lower(), "_key": key, **_bib_fields(body)})
    return out

def _bib_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    n, j = len(body), 0
    while j < n:
        while j < n and body[j] in " \t\n\r,":
            j += 1
        if j >= n:
            break
        ns = j
        while j < n and (body[j].isalnum() or body[j] == "_"):
            j += 1
        name = body[ns:j].lower()
        if not name:
            break
        while j < n and body[j] in " \t\n\r":
            j += 1
        if j >= n or body[j] != "=":
            break
        j += 1
        while j < n and body[j] in " \t\n\r":
            j += 1
        if j >= n:
            break
        if body[j] == "{":
            j += 1
            depth, vs = 1, j
            while j < n and depth > 0:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            value = body[vs:j - 1]
        elif body[j] == '"':
            j += 1
            vs = j
            while j < n and body[j] != '"':
                if body[j] == "\\" and j + 1 < n:
                    j += 1
                j += 1
            value = body[vs:j]
            if j < n:
                j += 1
        else:
            vs = j
            while j < n and body[j] not in ",\n":
                j += 1
            value = body[vs:j].strip()
        fields[name] = value
    return fields

# LaTeX accent + special-char cleanup
_LATEX_ACCENT_RE = re.compile(r"\\([\"'`^~.])\s*\{?\s*([A-Za-z])\s*\}?")
_LATEX_ACCENT_MAP = {
    ('"', 'a'): 'ä', ('"', 'e'): 'ë', ('"', 'i'): 'ï', ('"', 'o'): 'ö',
    ('"', 'u'): 'ü', ('"', 'A'): 'Ä', ('"', 'O'): 'Ö', ('"', 'U'): 'Ü',
    ("'", 'a'): 'á', ("'", 'e'): 'é', ("'", 'i'): 'í', ("'", 'o'): 'ó',
    ("'", 'u'): 'ú', ("'", 'c'): 'ć', ("'", 'n'): 'ń', ("'", 'A'): 'Á',
    ("'", 'E'): 'É', ("'", 'I'): 'Í', ("'", 'O'): 'Ó', ("'", 'U'): 'Ú',
    ("`", 'a'): 'à', ("`", 'e'): 'è', ("`", 'i'): 'ì', ("`", 'o'): 'ò',
    ("`", 'u'): 'ù',
    ('^', 'a'): 'â', ('^', 'e'): 'ê', ('^', 'i'): 'î', ('^', 'o'): 'ô',
    ('^', 'u'): 'û',
    ('~', 'a'): 'ã', ('~', 'n'): 'ñ', ('~', 'o'): 'õ',
    ('.', 'c'): 'ċ', ('.', 'e'): 'ė',
}
_LATEX_SPECIAL = {
    r"\&": "&", r"\%": "%", r"\#": "#", r"\$": "$",
    r"\_": "_", r"\{": "{", r"\}": "}",
    r"\textendash": "–", r"\textemdash": "—",
    r"\ldots": "…", r"\dots": "…",
    r"\o": "ø", r"\O": "Ø", r"\ss": "ß",
    r"\aa": "å", r"\AA": "Å", r"\ae": "æ", r"\AE": "Æ",
    "---": "—", "--": "–",
}

def _bib_clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    s = _LATEX_ACCENT_RE.sub(
        lambda m: _LATEX_ACCENT_MAP.get((m.group(1), m.group(2)), m.group(2)), s)
    for k, v in _LATEX_SPECIAL.items():
        s = s.replace(k, v)
    return s.replace("{", "").replace("}", "").strip()

def _bib_join_authors(field: str) -> str:
    """Render an author/editor list bibtex-plain style."""
    if not field:
        return ""
    parts = re.split(r"\s+and\s+", field)
    out: list[str] = []
    for p in parts:
        p = _bib_clean(p)
        if "," in p:  # "Last, First" → "First Last"
            last, first = p.split(",", 1)
            p = f"{first.strip()} {last.strip()}"
        out.append(p)
    if len(out) <= 1:
        return out[0] if out else ""
    if len(out) == 2:
        return f"{out[0]} and {out[1]}"
    return ", ".join(out[:-1]) + f", and {out[-1]}"

def _bib_render_entry(e: dict, num: int | None) -> str:
    key = e["_key"]
    bracket = f"[{num}]" if num is not None else f"[{key}]"
    authors = _bib_join_authors(e.get("author") or e.get("editor") or "")
    title  = _bib_clean(e.get("title", ""))
    year   = _bib_clean(e.get("year", ""))
    month  = _bib_clean(e.get("month", ""))
    pages  = _bib_clean(e.get("pages", ""))
    volume = _bib_clean(e.get("volume", ""))
    number = _bib_clean(e.get("number", ""))
    doi    = _bib_clean(e.get("doi", ""))
    url    = _bib_clean(e.get("url", ""))
    journal   = _bib_clean(e.get("journal", ""))
    booktitle = _bib_clean(e.get("booktitle", ""))
    publisher = _bib_clean(e.get("publisher") or e.get("howpublished")
                           or e.get("institution") or "")

    # title links to url, else doi.org
    title_url = url or (f"https://doi.org/{doi}" if doi else "")
    title_md = f"[{title}]({title_url})" if (title and title_url) else title

    date = (f"{month} {year}".strip()) if (month or year) else ""

    bits: list[str] = []
    if authors:
        bits.append(authors)
    if title_md:
        bits.append(title_md)

    # venue formatting by entry kind
    kind = e.get("_type", "")
    if kind == "article" and journal:
        seg = f"*{journal}*"
        if volume:
            seg += f", {volume}" + (f"({number})" if number else "")
            if pages:
                seg += f":{pages}"
        elif pages:
            seg += f", {pages}"
        if date:
            seg += f", {date}"
        bits.append(seg)
    elif kind in ("inproceedings", "incollection") and booktitle:
        seg = f"In *{booktitle}*"
        if pages:
            seg += f", pages {pages}"
        if publisher:
            seg += f". {publisher}"
        if date:
            seg += f", {date}"
        bits.append(seg)
    else:
        # @book / @misc / fallback
        tail = []
        if publisher:
            tail.append(publisher)
        if booktitle and not publisher:
            tail.append(f"*{booktitle}*")
        if date:
            tail.append(date)
        if tail:
            bits.append(", ".join(tail))

    body = ". ".join(bits)
    if body and not body.endswith("."):
        body += "."

    # keep CITEREF_<Key> case so cached Doxygen links resolve
    return f'<a id="CITEREF_{key}"></a>\n\n**{bracket}** {body}'

def _bib_render_all(entries: list[dict], numbering: dict[str, int]) -> str:
    out = ["Bibliography {#citelist}", "============", ""]
    for e in entries:
        out.append(_bib_render_entry(e, numbering.get(e["_key"])))
        out.append("")
    return "\n".join(out)


def _bib_sort_key(e: dict) -> tuple:
    """bibtex `plain` sort: author last name, year, title (matches live site)."""
    authors = e.get("author") or e.get("editor") or "zzz"
    first = re.split(r"\s+and\s+", authors)[0]
    first = _bib_clean(first)
    if "," in first:                       # "Last, First"
        last = first.split(",", 1)[0].strip()
    else:                                  # "First Middle Last"
        toks = first.split()
        last = toks[-1] if toks else "zzz"
    return (last.lower(),
            _bib_clean(e.get("year", "")),
            _bib_clean(e.get("title", "")).lower())

# opencv.bib + per-module + contrib bibs (contrib affects sort order)
_BIB_FILES: list[pathlib.Path] = []
if (DOC_ROOT / "opencv.bib").is_file():
    _BIB_FILES.append(DOC_ROOT / "opencv.bib")
_BIB_FILES += sorted((OPENCV_ROOT / "modules").glob("*/doc/*.bib"))
if CONTRIB_ROOT.is_dir():
    _BIB_FILES += sorted(CONTRIB_ROOT.glob("*/doc/*.bib"))

_CITE_NUMBER: dict[str, int] = {}
# reused by citelist generator for the same sort order
_BIB_ENTRIES_SORTED: list[dict] = []
_seen_keys: set[str] = set()  # dedupe; first wins
_all_entries: list[dict] = []
for _bf in _BIB_FILES:
    try:
        _txt = _bf.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    for _e in _bib_parse(_txt):
        if _e["_key"] in _seen_keys:
            continue
        _seen_keys.add(_e["_key"])
        _all_entries.append(_e)
_BIB_ENTRIES_SORTED = sorted(_all_entries, key=_bib_sort_key)
for _i, _e in enumerate(_BIB_ENTRIES_SORTED, 1):
    _CITE_NUMBER[_e["_key"]] = _i

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
    """Follow `_REDIRECT_MAP` transitively (cycle-safe)."""
    seen: set[str] = set()
    while anchor in _REDIRECT_MAP and anchor not in seen:
        seen.add(anchor)
        anchor = _REDIRECT_MAP[anchor]
    return anchor

# Anchor maps
_ANCHOR_TO_DOC: dict[str, str] = {}            # anchor -> docname (internal)
_ANCHOR_TO_EXTERNAL: dict[str, tuple[str, str]] = {}  # anchor -> (title, url)
_ANCHOR_TO_TITLE: dict[str, str] = {}          # anchor -> first-heading title
_LOCAL_MEMBER_IDS: set[str] = set()            # member detail targets we emit
# anchors reachable via @subpage/@ref; rest are orphans
_REFERENCED_ANCHORS: set[str] = set()

_HEAD_RE = re.compile(
    r"^(?P<title1>[^\n]+?)\s*\{#(?P<anchor1>[\w-]+)\}\s*\n[=\-]{3,}\s*$"
    r"|"
    r"^#+\s+(?P<title2>[^\n]+?)\s*\{#(?P<anchor2>[\w-]+)\}\s*$",
    re.MULTILINE)

def _scan_internal(path: pathlib.Path, base: pathlib.Path | None = None) -> None:
    """Add every {#anchor} and `@anchor NAME` in `path` to _ANCHOR_TO_DOC."""
    base = base or SPHINX_INPUT_ROOT
    _SUFFIXES = (".markdown", ".md")
    if path.is_file():
        files = [path] if path.suffix in _SUFFIXES else []
    elif path.is_dir():
        # skip _old/** (not compiled)
        files = [p for s in _SUFFIXES for p in path.rglob(f"*{s}")
                 if "_old" not in p.parts]
    else:
        files = []
    for md in files:
        try:
            body = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # unresolved path so staged symlinks give staging-root docnames
        try:
            rel = md.relative_to(base).with_suffix("").as_posix()
        except ValueError:
            # file outside base; fall back to DOC_ROOT-relative
            rel = md.relative_to(DOC_ROOT).with_suffix("").as_posix()
        for m in re.finditer(r"\{#([\w-]+)\}", body):
            _ANCHOR_TO_DOC[m.group(1)] = rel
        for m in re.finditer(r"^@anchor\s+([\w-]+)\s*$", body, re.MULTILINE):
            _ANCHOR_TO_DOC[m.group(1)] = rel
        # anchors this file links to (orphan detection)
        for m in re.finditer(r"@(?:subpage|ref)\s+([\w-]+)", body):
            _REFERENCED_ANCHORS.add(m.group(1))
        # first heading title
        tm = _HEAD_RE.search(body[:4000])
        if tm:
            anchor = tm.group("anchor1") or tm.group("anchor2")
            title = (tm.group("title1") or tm.group("title2") or "").strip()
            if anchor and title:
                _ANCHOR_TO_TITLE[anchor] = title

def _scan_external(toc_file: pathlib.Path) -> None:
    """Add a module TOC file's top (title, anchor) to _ANCHOR_TO_EXTERNAL."""
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

# shared dict objects populated by the build module
_IMAGE_INDEX: dict[str, str] = {}
_SNIPPET_INDEX: dict[str, pathlib.Path] = {}

_ALL_CLASSES: dict[str, dict] = {}      # refid -> {qualified, kind, brief, docname}
_ALL_NAMESPACES: dict[str, dict] = {}   # name  -> {refid, brief, docname}

_TOGGLE_LABELS = {"cpp": "C++", "java": "Java", "python": "Python"}


# Mirrors Doxygen EXAMPLE_PATH; OPENCV_ROOT first (order matters)
_SNIPPET_BASES = [
    OPENCV_ROOT,
    OPENCV_ROOT / "samples",
    OPENCV_ROOT / "apps",
] + [CONTRIB_ROOT / _m / "samples" for _m in CONTRIB_MODULES]

# Remap Doxygen language names Pygments doesn't know
_LANG_ALIASES = {
    "none": "text",
    "unparsed": "text",
    "guess": "text",
    "gradle": "groovy",
    "run": "bash",
    # `m` = Objective-C in the iOS tutorials (.m sources); Pygments uses `objc`.
    "m": "objc",
    # No Pygments lexer for these fence tags used by ios/app/face tutorials —
    # render as plain text instead of warning "lexer name is not known".
    "csv": "text",
    "plaintext": "text",
}

USE_INDEX_LANDING = True

__all__ = [
    "USE_INDEX_LANDING",
    "HERE", "DOC_ROOT", "OPENCV_ROOT",
    "DOC_MODULES", "JS_DOC_MODULES", "PY_DOC_MODULES",
    "CONTRIB_MODULES", "CONTRIB_ROOT", "SPHINX_INPUT_ROOT", "API_MODULES",
    "_API_XML_DIR", "_PATCHED_XML_DIR", "_module_group_stem",
    "_PY_SIGNATURES", "_python_enum_name",
    "HAVE_SPHINX_DESIGN", "HAVE_BREATHE",
    "DOXYGEN_BASE_URL", "_doxygen_url",
    "_TAG_FILE", "_TAG_FILENAMES", "_TAG_TITLES", "_DOC_PAGE_TITLES",
    "_CV_SYMBOL_URL", "_FILE_URL", "_FALLBACK_MODULE_HEADERS",
    "_CALL_GRAPH_ANCHORS", "_DOXY_ANCHOR_TO_MEMBER", "_norm_args",
    "_LIVE_GROUP_URL", "_LIVE_CLASS_URL", "_LIVE_TYPEDEF_URL",
    "_LOCAL_CLASS_URL", "_LOCAL_TYPEDEF_URL", "_CLASS_TEMPLATE_DISPLAY",
    "_LOCAL_PAGE_BY_DOXY_FILE", "_PLACEHOLDER_STUBS", "_doxy_page_to_local",
    "_func_slug",
    "_CITE_NUMBER", "_BIB_ENTRIES_SORTED", "_bib_render_all",
    "_REDIRECT_MAP", "_resolve_redirect",
    "_ANCHOR_TO_DOC", "_ANCHOR_TO_EXTERNAL", "_ANCHOR_TO_TITLE", "_LOCAL_MEMBER_IDS",
    "_REFERENCED_ANCHORS", "_HEAD_RE",
    "_scan_internal", "_scan_external",
    "_IMAGE_INDEX", "_SNIPPET_INDEX", "_SNIPPET_BASES",
    "_ALL_CLASSES", "_ALL_NAMESPACES",
    "_TOGGLE_LABELS", "_LANG_ALIASES",
]
