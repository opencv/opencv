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

# ---------------------------------------------------------------------------
# SCOPE — contrib tree.  Folder names under opencv_contrib/modules/.
# Override via env var to avoid editing this file:
#     OPENCV_CONTRIB_MODULES=ml,bgsegm cmake --build <build> --target sphinx
# Empty list = main-only build (legacy behavior, no contrib site).
# ---------------------------------------------------------------------------
CONTRIB_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_CONTRIB_MODULES") or "ml,bgsegm,bioinspired,cannops,ccalib,cnn_3dobj,cvv,dnn_objdetect,dnn_superres,gapi,hdf,julia,line_descriptor,phase_unwrapping,structured_light,viz,tracking").split(",")
    if m.strip()
]
CONTRIB_ROOT = pathlib.Path(
    _os.environ.get("OPENCV_CONTRIB_ROOT")
    or str(HERE.parent.parent / "opencv_contrib" / "modules")
).resolve()

# ---------------------------------------------------------------------------
# SCOPE — API reference. Module folder names under opencv/modules/. Each
# entry's umbrella header (modules/<m>/include/opencv2/<m>.hpp) must declare
# `@defgroup <m> …` at the top — that's the breathe target. Override:
#     OPENCV_API_MODULES=core,imgproc cmake --build <build> --target sphinx
# Empty = legacy behavior (no API pages in Sphinx; navbar's external_links
# still routes users to the Doxygen-rendered group__*.html).
# ---------------------------------------------------------------------------
API_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_API_MODULES") or "core").split(",")
    if m.strip()
]

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
extensions = ["myst_parser"]
for _ext in ("sphinx_design", "sphinx_copybutton"):
    try:
        __import__(_ext)
        extensions.append(_ext)
    except ImportError:
        pass
HAVE_SPHINX_DESIGN = "sphinx_design" in extensions

# -- Breathe (Doxygen XML -> Sphinx C++ domain) -----------------------------
# Gated on API_MODULES being non-empty AND breathe being importable. A
# stripped env (no breathe) or empty API_MODULES degrades to tutorial-only,
# matching today's pre-API behavior. XML dir is forwarded by
# docs_sphinx/CMakeLists.txt via OPENCV_DOXYGEN_XML_DIR; fallback path
# matches the canonical build_doc layout for ad-hoc sphinx-build runs.
_API_XML_DIR = pathlib.Path(
    _os.environ.get("OPENCV_DOXYGEN_XML_DIR")
    or str(HERE.parent.parent / "build_doc" / "doc" / "doxygen" / "xml")
).resolve()
# Patched XML dir (see `_patch_namespace_xml_for_breathe` below). breathe is
# pointed at this rather than the raw Doxygen output so that functions defined
# inside `@addtogroup` regions (which Doxygen lists only as `<member refid>`
# in namespace XML, not full `<memberdef>`) become findable. The dir is built
# at sphinx-build time, mirrors the original XML via symlinks, and only the
# affected namespace XMLs are rewritten in place.
_PATCHED_XML_DIR = _API_XML_DIR.parent / "xml_for_sphinx"
if API_MODULES:
    try:
        import breathe  # noqa: F401
        extensions.append("breathe")
        breathe_projects = {"opencv": str(_PATCHED_XML_DIR)}
        breathe_default_project = "opencv"
        breathe_default_members = ("members",)
    except ImportError:
        API_MODULES = []

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
            elif _kind == "group":
                # Doxygen module pages (core, imgproc, dnn, …) live as
                # `kind="group"` compounds in the tagfile. Without capturing
                # them here, inline `@ref core` and bullet-list refs to module
                # roots in intro.markdown don't resolve to anything.
                _n = _c.findtext("name")
                _f = _c.findtext("filename")
                _t = _c.findtext("title")
                if _n and _f:
                    _TAG_FILENAMES[_n] = _f if _f.endswith(".html") else _f + ".html"
                if _n and _t:
                    _TAG_TITLES[_n] = _t
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
# doesn't need a network fetch.  The same parsed entries also feed the
# Sphinx-side bibliography page generated further down, so the `[N]`
# emitted at @cite-resolution time always matches the `[N]` rendered on
# the citelist page.
def _bib_parse(text: str) -> list[dict]:
    """Walk a BibTeX file into a list of {_type, _key, field: value, ...}.
    Brace-balanced; handles `{...}` and `"..."` value forms. Concatenation
    (`val # "more"`) is not supported — opencv.bib doesn't use it."""
    out: list[dict] = []
    n, i = len(text), 0
    while i < n:
        m = re.search(r"@(\w+)\s*\{\s*([^\s,]+)\s*,", text[i:])
        if not m:
            break
        kind, key = m.group(1), m.group(2)
        i += m.end()
        depth, body_start = 1, i
        while i < n and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth != 0:
            break  # malformed entry; stop rather than misparse the rest
        out.append({"_type": kind.lower(), "_key": key,
                    **_bib_fields(text[body_start:i - 1])})
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

# LaTeX accent + special-char cleanup. Just enough to render opencv.bib
# readably; not a full parser.
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
    """Render an author/editor list bibtex-plain style:
        1 author  -> "A"
        2 authors -> "A and B"
        3+        -> "A, B, and C"  (Oxford comma + 'and')"""
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

    # bibtex `plain` style wraps the title in the URL hyperlink (or DOI URL
    # when no `url` field is set). DOI without a URL field is rendered as a
    # https://doi.org/... link on the title.
    title_url = url or (f"https://doi.org/{doi}" if doi else "")
    title_md = f"[{title}]({title_url})" if (title and title_url) else title

    # date = month + year ("nov 2012") — bibtex's plain `byear` formatter.
    date = (f"{month} {year}".strip()) if (month or year) else ""

    bits: list[str] = []
    if authors:
        bits.append(authors)
    if title_md:
        bits.append(title_md)

    # Venue formatting differs by entry kind (Doxygen/bibtex plain style):
    #   @article       -> "*Journal*, V(N):pages, date."
    #   @inproceedings -> "In *Booktitle*, pages X-Y. Publisher, date."
    #   @incollection  -> "In *Booktitle*, pages X-Y. Publisher, date."
    #   @book/@misc    -> "Publisher, date."   (or just date)
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
        # @book / @misc / @techreport / fallback
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

    # Raw HTML anchor preserves the original CITEREF_<Key> case (matches
    # Doxygen's URL convention exactly, so cached links to
    # `citelist.html#CITEREF_<Key>` keep resolving on the new site too).
    return f'<a id="CITEREF_{key}"></a>\n\n**{bracket}** {body}'

def _bib_render_all(entries: list[dict], numbering: dict[str, int]) -> str:
    out = ["Bibliography {#citelist}", "============", ""]
    for e in entries:
        out.append(_bib_render_entry(e, numbering.get(e["_key"])))
        out.append("")
    return "\n".join(out)


def _bib_sort_key(e: dict) -> tuple:
    """bibtex `plain` style: sort by first author's last name, then year,
    then title.  Without this, our citelist numbering doesn't match
    docs.opencv.org/5.x — Doxygen runs bibtex with LATEX_BIB_STYLE=plain
    (set in doc/Doxyfile.in), which sorts by author, NOT by bib key."""
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

# Discover every .bib file Doxygen would feed bibtex (see opencv/doc/
# CMakeLists.txt: paths_bib accumulates `${m}.bib` for each module in
# OPENCV_DOC_LIST plus the main opencv.bib).  Reading them all here is
# what makes `[1] Achanta…` appear first — that entry lives in
# opencv_contrib/modules/ximgproc/doc/ximgproc.bib, not in doc/opencv.bib.
_BIB_FILES: list[pathlib.Path] = []
if (DOC_ROOT / "opencv.bib").is_file():
    _BIB_FILES.append(DOC_ROOT / "opencv.bib")
_BIB_FILES += sorted((OPENCV_ROOT / "modules").glob("*/doc/*.bib"))
if CONTRIB_ROOT.is_dir():
    _BIB_FILES += sorted(CONTRIB_ROOT.glob("*/doc/*.bib"))

_CITE_NUMBER: dict[str, int] = {}
# Parsed entries kept in module scope so the citelist generator (below) reuses
# the same sort order that fed `_CITE_NUMBER`. Numbering stays consistent
# without re-parsing or re-sorting in two places.
_BIB_ENTRIES_SORTED: list[dict] = []
_seen_keys: set[str] = set()  # dedupe across bibs; first occurrence wins
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

# Exposed to templates/navbar-nav.html so it can rewrite external_links
# whose URL starts with this base into depth-aware relative paths to the
# local Doxygen output, instead of redirecting users to docs.opencv.org.
html_context = {"doxygen_base_url": DOXYGEN_BASE_URL}

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

def _scan_internal(path: pathlib.Path, base: pathlib.Path | None = None) -> None:
    """Add every {#anchor} and standalone `@anchor NAME` in `path` (file
    or dir) to _ANCHOR_TO_DOC. Picks up both `.markdown` (the bulk of the
    tree) and `.md` (the form used by ports like dnn/dnn_pytorch_tf_*).
    Docname is computed relative to `base` (default SPHINX_INPUT_ROOT) so
    the same scanner serves both main and contrib trees."""
    base = base or SPHINX_INPUT_ROOT
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
        # Use the unresolved path so symlinks in the staged input tree
        # produce docnames relative to the staging root, not to their
        # real source location (opencv/doc/ or opencv_contrib/modules/).
        try:
            rel = md.relative_to(base).with_suffix("").as_posix()
        except ValueError:
            # File lives outside `base` (e.g. js_tutorials/py_tutorials
            # scanned from DOC_ROOT while base=SPHINX_INPUT_ROOT). Fall
            # back to DOC_ROOT-relative naming.
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

# Internal scan: master + every enabled main and contrib module subtree.
# Walk the staged tree so docnames stay relative to SPHINX_INPUT_ROOT (Sphinx
# srcdir), regardless of where the actual source files live on disk.
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
_contrib_root_md = SPHINX_INPUT_ROOT / "tutorials_contrib" / "contrib_root.markdown"
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
# Generate the API stub tree from Doxygen XML, then scan it. Stub layout
# mirrors Doxygen's group hierarchy: parent groups (those with <innergroup>
# children in their XML) become navigation index pages with `@subpage`
# toctrees; leaf groups get a single `{doxygengroup} <name>` directive.
# This matches what docs.opencv.org/.../group__core.html does — Doxygen
# also separates navigation from content; only one breathe directive
# (`{doxygengroup} core :inner:`) flattens the whole hierarchy onto one
# page, which is why we don't use it.
def _itertext(el) -> str:
    """Flatten an XML element's inner text. None-safe."""
    return "".join(el.itertext()).strip() if el is not None else ""


# memberdef@kind → display section title. Mirrors Doxygen's group-page order.
_MEMBERDEF_SECTIONS = (
    ("typedef",  "Typedefs"),
    ("enum",     "Enumerations"),
    ("function", "Functions"),
    ("variable", "Variables"),
    ("define",   "Macros"),
)


def _read_class_brief(refid: str, xml_dir: pathlib.Path,
                      _cache: dict = {}) -> str:
    """Read brief description from a class/struct's compound XML. Cached."""
    if refid in _cache:
        return _cache[refid]
    import xml.etree.ElementTree as _ET
    xml_path = xml_dir / f"{refid}.xml"
    brief = ""
    if xml_path.is_file():
        try:
            ccd = _ET.parse(xml_path).getroot().find("compounddef")
            if ccd is not None:
                brief = _itertext(ccd.find("briefdescription"))
        except _ET.ParseError:
            pass
    _cache[refid] = brief
    return brief


def _build_api_hierarchy(refid: str, xml_dir: pathlib.Path,
                         _seen: set | None = None) -> dict | None:
    """Walk a group XML's <innergroup> children recursively.
    Returns {name, title, detailed, innerclasses, sections, children} or None.
    `_seen` guards against the rare case of cycles in the group graph."""
    import xml.etree.ElementTree as _ET
    _seen = _seen if _seen is not None else set()
    if refid in _seen:
        return None
    _seen.add(refid)
    xml_path = xml_dir / f"{refid}.xml"
    if not xml_path.is_file():
        return None
    try:
        root = _ET.parse(xml_path).getroot()
    except _ET.ParseError:
        return None
    cd = root.find("compounddef")
    if cd is None:
        return None
    name = (cd.findtext("compoundname") or "").strip()
    title = (cd.findtext("title") or name).strip()
    # Detailed description (used on parent index pages; breathe handles it
    # on leaf pages, so we extract it for context-display only).
    detailed_el = cd.find("detaileddescription")
    detailed = ""
    if detailed_el is not None:
        paras = [_itertext(p) for p in detailed_el.findall("para")]
        detailed = "\n\n".join(p for p in paras if p)
    # Inner classes (public only). One read per class's XML for its brief.
    # `qualified` is what `{doxygenclass}` needs (e.g. cv::ocl::Context); the
    # innerclass element's text already carries that, but normalize spaces.
    innerclasses = []
    for ic in cd.findall("innerclass"):
        if ic.get("prot") != "public":
            continue
        ic_refid = ic.get("refid", "")
        qualified = " ".join((ic.text or "").split())
        innerclasses.append({
            "refid": ic_refid,
            "name": qualified,
            "qualified": qualified,
            "kind": "struct" if ic_refid.startswith("struct") else "class",
            "brief": _read_class_brief(ic_refid, xml_dir),
        })
    # Section members (typedefs, enums, functions, variables, macros).
    # `qualified` and `param_types` exist so we can emit per-member breathe
    # directives (doxygenenum / doxygenfunction / …) instead of one big
    # doxygengroup; the latter inlines every <innerclass> on the group page,
    # which is the opposite of Doxygen's group-page layout.
    sections: dict[str, list[dict]] = {}
    for sd in cd.findall("sectiondef"):
        for md in sd.findall("memberdef"):
            kind = md.get("kind", "")
            section_title = dict(_MEMBERDEF_SECTIONS).get(kind)
            if not section_title:
                continue
            qualified = (md.findtext("qualifiedname") or "").strip()
            if not qualified:
                qualified = (md.findtext("name") or "").strip()
            # Param types: <type> text plus any <array> suffix (Doxygen
            # stores `int foo[3]` as <type>int</type>...<array>[3]</array>;
            # without merging them our breathe disambiguator omits the `[3]`
            # and breathe reports "Unable to resolve function with
            # arguments (…, const double)" even though the function exists).
            def _param_type(p) -> str:
                t = _itertext(p.find("type"))
                arr = (p.findtext("array") or "").strip()
                return (t + arr) if arr else t
            param_types = [_param_type(p) for p in md.findall("param")]
            # Enum values + scoped-vs-unscoped flag (for Doxygen-style
            # synopsis rendering — one code block per enum with all values
            # inside `{ }`, instead of breathe's discrete signature blocks).
            enum_values = []
            is_strong = md.get("strong", "no") == "yes"
            if kind == "enum":
                for ev in md.findall("enumvalue"):
                    enum_values.append({
                        "name":        (ev.findtext("name") or "").strip(),
                        "initializer": (ev.findtext("initializer") or "").strip(),
                    })
            sections.setdefault(section_title, []).append({
                "id":          md.get("id", ""),
                "kind":        kind,
                "name":        (md.findtext("name") or "").strip(),
                "qualified":   qualified,
                "type":        _itertext(md.find("type")),
                "args":        (md.findtext("argsstring") or "").strip(),
                "param_types": param_types,
                "brief":       _itertext(md.find("briefdescription")),
                "enum_values": enum_values,
                "strong":      is_strong,
            })
    # Recurse into subgroups.
    children = []
    for ig in cd.findall("innergroup"):
        child = _build_api_hierarchy(ig.get("refid"), xml_dir, _seen)
        if child is not None:
            children.append(child)
    return {"name": name, "title": title, "detailed": detailed,
            "innerclasses": innerclasses, "sections": sections,
            "children": children}


def _md_escape_cell(text: str) -> str:
    """Make `text` safe for a single Markdown table cell."""
    # Newlines collapse to spaces, pipes escape, angle brackets stay.
    return (text or "").replace("\n", " ").replace("\r", " ") \
                       .replace("|", "\\|").strip()


# Per-member breathe directive selector. The full doxygengroup directive
# recursively inlines every <innerclass> + <innernamespace>, which is the
# *opposite* of how Doxygen's group page lays out (classes are links to
# separate pages there). Emitting one directive per member keeps each
# member's detail block scoped to itself and lets us push classes to their
# own per-class pages — see _write_class_stub.
_MEMBER_DIRECTIVE = {
    "enum":     "doxygenenum",
    "function": "doxygenfunction",
    "typedef":  "doxygentypedef",
    "variable": "doxygenvariable",
    "define":   "doxygendefine",
}
# Section header for each member kind's detail block. Mirrors what Doxygen
# emits on a group page (e.g. group__core__opencl.html has separate
# "Enumeration Type Documentation" and "Function Documentation" sections,
# not one collapsed "Detailed Description").
_MEMBER_DETAIL_SECTION = {
    "Typedefs":     "Typedef Documentation",
    "Enumerations": "Enumeration Type Documentation",
    "Functions":    "Function Documentation",
    "Variables":    "Variable Documentation",
    "Macros":       "Macro Definition Documentation",
}


def _enum_synopsis_lines(m: dict) -> list[str]:
    """Render an enum as a Doxygen-style code synopsis: one `enum {…}` block
    listing every value with its initializer. Used in place of breathe's
    `{doxygenenum}` directive, which emits a discrete signature box per
    enumerator (one box for the enum + one per value) — that's the layout
    in the user's "before" screenshot. Doxygen's group page renders the
    enum as a single code-style box; this helper reproduces that.

    Value-name qualification follows Doxygen:
      * Scoped (`enum class`) → values prefixed with the enum's own
        qualified name.
      * Unscoped → values prefixed with the enum's *parent* scope
        (namespace or enclosing class), so they look like the C++ name
        you'd actually write in code."""
    qualified = m.get("qualified") or m["name"]
    is_strong = bool(m.get("strong"))
    keyword = "enum class" if is_strong else "enum"
    if is_strong:
        prefix = qualified + "::"
    elif "::" in qualified:
        prefix = qualified.rsplit("::", 1)[0] + "::"
    else:
        prefix = ""
    out = [f"{keyword} {qualified} {{"]
    vals = m.get("enum_values", []) or []
    for i, v in enumerate(vals):
        comma = "," if i < len(vals) - 1 else ""
        init = (" " + v["initializer"]) if v["initializer"] else ""
        out.append(f"    {prefix}{v['name']}{init}{comma}")
    out.append("}")
    return out


def _function_signature(member: dict) -> str:
    """Disambiguator used after a qualified function name in `{doxygenfunction}`.
    Breathe expects `name(type1, type2, …)` with parameter names dropped (it
    matches against Doxygen's `<param><type>` text, and on the type-mangled
    signature — parameter names *and* default values are irrelevant to the
    match). Empty-arg functions get `()` — required for breathe to match
    correctly even for non-overloads.

    Trailing `const` is appended for const member functions: breathe matches
    the cv-qualifier as part of the declaration, so a bare `(types)` arg list
    fails to resolve a `const` method. Doxygen stores `int channels(int i=-1)
    const`; `{doxygenfunction} cv::_InputArray::channels(int)` (no const)
    parses to a non-const AST and reports "Unable to resolve function … with
    arguments (int)". Appending ` const` makes the directive arg-list match
    the stored declaration. Group-page members carry no `const` key, so free
    functions are unaffected."""
    types = ", ".join((t or "").strip() for t in member.get("param_types", []))
    sig = f"({types})"
    if member.get("const"):
        sig += " const"
    return sig


def _class_page_name(refid: str) -> str:
    """Filename (without extension) for the per-class page. We use the Doxygen
    refid verbatim so MyST cross-refs and internal links from breathe stay
    stable (breathe's class anchors are the C++-mangled `_CPPv4N…` form, not
    the refid — so there's no collision with the page name)."""
    return refid


def _write_api_stub(node: dict, out_dir: pathlib.Path,
                    classes_seen: dict) -> None:
    """Write one .md per group node. Recurses into children.

    Parent groups (have <innergroup> children) → navigation index pages with
    @subpage toctrees. Leaf groups → Doxygen-style summary tables (Classes /
    Typedefs / Enumerations / Functions / Variables / Macros) at top, then
    a per-member detail block per kind (one breathe directive per member, not
    the recursive `{doxygengroup}` which inlines every nested class). Inner
    classes get their own pages — emitted later by `_generate_api_stubs` from
    `classes_seen`, which this fn populates."""
    name = node["name"]
    title = node["title"]
    out = out_dir / f"{name}.md"

    if node["children"]:
        # Navigation index page — list children as @subpage entries; the
        # existing _subpage_list_to_toctree rule converts them to a real
        # toctree at translate time.
        lines = [f"# {title} {{#api_{name}}}", ""]
        if node["detailed"]:
            lines += [node["detailed"], ""]
        lines += ["## Topics", ""]
        for child in node["children"]:
            lines.append(f"- @subpage api_{child['name']}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for child in node["children"]:
            _write_api_stub(child, out_dir, classes_seen)
        return

    # ---- Leaf page ----------------------------------------------------------
    lines = [f"# {title} {{#api_{name}}}", ""]

    # Classes summary table — link to the per-class page that
    # `_generate_api_stubs` emits (one .md per refid, deduped across groups).
    if node["innerclasses"]:
        lines += ["## Classes", "",
                  "| Name | Description |", "|---|---|"]
        for c in node["innerclasses"]:
            classes_seen.setdefault(c["refid"], c)
            page = _class_page_name(c["refid"])
            link = f"[`{c['kind']} {c['name']}`]({page}.md)"
            lines.append(f"| {link} | {_md_escape_cell(c['brief'])} |")
        lines.append("")

    # Build a fast lookup of class qualified names known so far — used to
    # detect when a group's "Functions"/"Variables" sectiondef is actually
    # listing a class member (Doxygen groups span class boundaries). Such
    # members get rendered on the class page, not as standalone breathe
    # directives.
    class_qualifieds = {c.get("qualified") for c in classes_seen.values()
                        if c.get("qualified")}

    def _is_class_member(m: dict) -> bool:
        q = m.get("qualified") or ""
        if "::" not in q:
            return False
        parent = q.rsplit("::", 1)[0]
        return parent in class_qualifieds

    def _is_template_spec(m: dict) -> bool:
        # Explicit template specializations carry `<…>` in the name (Doxygen
        # stores `cv::saturate_cast< unsigned >`). breathe's C++ parser
        # rejects this as a function-name argument, so we can't emit a
        # `doxygenfunction` directive for them — the summary table still
        # lists them; only the per-member detail block is skipped.
        return "<" in (m.get("name") or "")


    # Section summary tables in Doxygen's order. For class-member items the
    # in-page anchor breathe would have emitted doesn't exist (we skip the
    # per-member directive below) — point the link at the parent class page
    # instead so the table stays clickable.
    def _member_anchor_link(m: dict, label: str) -> str:
        if _is_class_member(m):
            q = m["qualified"]
            parent_qualified = q.rsplit("::", 1)[0]
            for c in classes_seen.values():
                if c.get("qualified") == parent_qualified:
                    return f"[`{label}`]({_class_page_name(c['refid'])}.md)"
        return f"[`{label}`](#{m['id']})"

    for _, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        if section_title == "Functions":
            lines += ["| Return | Name | Description |", "|---|---|---|"]
            for m in items:
                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                label = f"{m['name']}{_md_escape_cell(m['args'])}"
                sig_link = _member_anchor_link(m, label)
                lines.append(
                    f"| `{ret}` | {sig_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title in ("Typedefs", "Variables"):
            lines += ["| Type | Name | Description |", "|---|---|---|"]
            for m in items:
                t = _md_escape_cell(m["type"]) or "&nbsp;"
                name_link = _member_anchor_link(m, m["name"])
                lines.append(f"| `{t}` | {name_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Enumerations":
            # Code-style synopsis (Doxygen layout) instead of name/desc table.
            # Both summary and detail-block representations would duplicate
            # the same content — we only emit the synopsis here, and skip
            # enums in the detail-block loop below.
            for m in items:
                if m["brief"]:
                    lines.append(_md_escape_cell(m["brief"]))
                    lines.append("")
                lines.append("```cpp")
                lines.extend(_enum_synopsis_lines(m))
                lines.append("```")
                lines.append("")
            continue   # already appended trailing blank
        else:  # Macros
            lines += ["| Name | Description |", "|---|---|"]
            for m in items:
                name_link = _member_anchor_link(m, m["name"])
                lines.append(f"| {name_link} | {_md_escape_cell(m['brief'])} |")
        lines.append("")

    # Per-member detail blocks (Enumeration Type Documentation,
    # Function Documentation, …). One breathe directive per item — except
    # for enums, which are rendered as a single code-style synopsis (one
    # `enum {…}` block listing all values) to match Doxygen's group-page
    # layout. breathe's `{doxygenenum}` instead emits a discrete signature
    # box per enumerator, which looks fragmented and doesn't match the
    # reference rendering. Class members and template specializations are
    # skipped — see `_is_*`. Macros are deduped by name: Doxygen can emit
    # multiple <memberdef>s for an arity-overloaded macro, but breathe
    # renders the same C declaration each time and docutils complains
    # about duplicate IDs.
    seen_define_names: set[str] = set()
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        directive = _MEMBER_DIRECTIVE.get(kind_key)
        if not directive:
            continue
        # `rendered` collects entries with one of two shapes:
        #   ("breathe", spec, directive_name)  — emitted as a breathe block
        #   ("synopsis", brief, code_lines)    — emitted as a fenced block
        rendered = []
        for m in items:
            # Class members render on the class page; skip on the group page.
            if kind_key in ("function", "variable") and _is_class_member(m):
                continue
            # Explicit template specializations: breathe can't parse the
            # `<T>` in the function name, so we leave them out of the
            # detail section (table above still lists them).
            if _is_template_spec(m):
                continue
            if m["kind"] == "enum":
                # Enums are rendered as synopses in the "Enumerations"
                # summary section above; no separate detail block needed.
                continue
            qualified = m["qualified"] or m["name"]
            if m["kind"] == "function":
                # Always pass the param-types disambiguator. breathe sees
                # multiple matches for common names (e.g. cv::cos lives in
                # both core_quaternion and core_utils_softfloat) — without
                # a signature it can't pick. The XML patcher above
                # guarantees the matching <memberdef> is reachable for
                # breathe's lookup.
                spec = qualified + _function_signature(m)
            elif m["kind"] == "define":
                # Preprocessor macros aren't namespaced. Dedupe by name —
                # arity-overloaded macros (e.g. CV_LOG_VERBOSE) appear as
                # multiple memberdefs but render to the same C declaration.
                if m["name"] in seen_define_names:
                    continue
                seen_define_names.add(m["name"])
                spec = m["name"]
            else:
                spec = qualified
            rendered.append(("breathe", spec, directive))
        if not rendered:
            continue
        lines.append(f"## {_MEMBER_DETAIL_SECTION[section_title]}")
        lines.append("")
        for entry in rendered:
            if entry[0] == "synopsis":
                _, brief, code_lines = entry
                if brief:
                    lines.append(brief)
                    lines.append("")
                lines.append("```cpp")
                lines.extend(code_lines)
                lines.append("```")
                lines.append("")
            else:
                _, spec, dname = entry
                lines += [
                    f"```{{{dname}}} {spec}",
                    ":project: opencv",
                    "```",
                    "",
                ]

    # Hidden toctree for per-class pages — needed so Sphinx knows these
    # files exist and so the left sidebar lists them under this group.
    if node["innerclasses"]:
        lines += ["```{toctree}", ":hidden:", ":maxdepth: 1", ""]
        for c in node["innerclasses"]:
            lines.append(_class_page_name(c["refid"]))
        lines += ["```", ""]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Class-XML sectiondef kind → (summary heading, detail heading).
# Order in this mapping is the order Doxygen uses on a class page.
_CLASS_SUMMARY_SECTIONS = [
    ("public-type",             "Public Types"),
    ("public-func",             "Public Member Functions"),
    ("public-static-func",      "Static Public Member Functions"),
    ("public-attrib",           "Public Attributes"),
    ("public-static-attrib",    "Static Public Attributes"),
    ("protected-type",          "Protected Types"),
    ("protected-func",          "Protected Member Functions"),
    ("protected-static-func",   "Static Protected Member Functions"),
    ("protected-attrib",        "Protected Attributes"),
    ("protected-static-attrib", "Static Protected Attributes"),
    ("friend",                  "Friends"),
]


def _read_class_data(refid: str, xml_dir: pathlib.Path) -> dict | None:
    """Walk a class/struct compound XML and return everything the per-class
    page needs: brief + detailed for the class itself, and a list of
    members grouped by sectiondef kind. Returns None if the XML file is
    missing or unparseable — callers should fall back to a bare
    `{doxygenclass}` directive in that case.

    Each member dict carries the same fields `_build_api_hierarchy`
    captures, plus the `protection`, `static`, `const`, `virtual`,
    `explicit` flags from memberdef attributes (needed to render
    Doxygen-style annotations in the summary table)."""
    import xml.etree.ElementTree as _ET
    xml_path = xml_dir / f"{refid}.xml"
    if not xml_path.is_file():
        return None
    try:
        root = _ET.parse(xml_path).getroot()
    except _ET.ParseError:
        return None
    cd = root.find("compounddef")
    if cd is None:
        return None

    def _param_type(p) -> str:
        t = _itertext(p.find("type"))
        arr = (p.findtext("array") or "").strip()
        return (t + arr) if arr else t

    sections: dict[str, list[dict]] = {}
    for sd in cd.findall("sectiondef"):
        skind = sd.get("kind", "")
        items: list[dict] = []
        for md in sd.findall("memberdef"):
            mkind = md.get("kind", "")
            qualified = (md.findtext("qualifiedname") or "").strip()
            name = (md.findtext("name") or "").strip()
            enum_values = []
            if mkind == "enum":
                for ev in md.findall("enumvalue"):
                    enum_values.append({
                        "name":        (ev.findtext("name") or "").strip(),
                        "initializer": (ev.findtext("initializer") or "").strip(),
                    })
            items.append({
                "id":          md.get("id", ""),
                "kind":        mkind,
                "name":        name,
                "qualified":   qualified or name,
                "type":        _itertext(md.find("type")),
                "args":        (md.findtext("argsstring") or "").strip(),
                "param_types": [_param_type(p) for p in md.findall("param")],
                "brief":       _itertext(md.find("briefdescription")),
                "static":      md.get("static") == "yes",
                "virt":        md.get("virt", "non-virtual"),
                "const":       md.get("const") == "yes",
                "explicit":    md.get("explicit") == "yes",
                "enum_values": enum_values,
                "strong":      md.get("strong", "no") == "yes",
            })
        if items:
            sections[skind] = items

    detailed_el = cd.find("detaileddescription")
    detailed_paras = []
    if detailed_el is not None:
        for p in detailed_el.findall("para"):
            txt = _itertext(p)
            if txt:
                detailed_paras.append(txt)

    return {
        "name":     (cd.findtext("compoundname") or "").strip(),
        "brief":    _itertext(cd.find("briefdescription")),
        "detailed": "\n\n".join(detailed_paras),
        "sections": sections,
    }


def _find_collaboration_svg(refid: str, html_root: pathlib.Path) -> pathlib.Path | None:
    """Locate the Doxygen-generated collaboration-diagram SVG for a class.

    Our XML pipeline (CMakeLists.txt's `Doxyfile-xml`) sets
    `COLLABORATION_GRAPH = NO` and friends — graph elements in XML would be
    forwarded by breathe as `graphviz` docutils nodes, which need an
    extension we don't load. So the diagram never reaches the XML.

    The *legacy* Doxygen HTML build (the `doxygen` target, separate from
    `sphinx-xml`) still renders it as `<refid>__coll__graph.svg`, written
    into a content-addressed subdir because the legacy Doxyfile keeps
    `CREATE_SUBDIRS=YES`. The HTML tree sits next to the XML tree
    (`…/doxygen/html` ⟷ `…/doxygen/xml`). We read that asset read-only —
    nothing in the Doxygen output is modified. Returns None when the legacy
    HTML build hasn't run (graphs simply stay absent, no crash)."""
    if not html_root.is_dir():
        return None
    matches = sorted(html_root.rglob(f"{refid}__coll__graph.svg"))
    return matches[0] if matches else None


def _svg_make_transparent(text: str) -> str:
    """Light-mode variant: only the full-canvas backdrop is made transparent
    so the white page shows through (native Doxygen look). Graphviz paints the
    canvas as a single `fill="white" stroke="transparent"` polygon."""
    return text.replace('fill="white" stroke="transparent"',
                        'fill="none" stroke="transparent"', 1)


def _svg_dark_variant(text: str) -> str:
    """Dark-mode variant: recolour the (light) Doxygen SVG into a dark diagram
    matching docs.opencv.org — transparent canvas (page slate shows through),
    dark node fills, light borders/text, lightened connector arrows. We recolour
    the SVG itself (rather than a CSS `filter: invert`, which turns the large
    white node boxes solid black) so the result blends with the dark page.

    Order matters: blank the backdrop first, *then* repaint the remaining white
    node fills, so the two `fill="white"` cases don't collide."""
    text = _svg_make_transparent(text)              # backdrop → transparent
    text = text.replace('fill="white"', 'fill="#1c2128"')   # node box fills → dark slate
    text = text.replace('fill="#bfbfbf"', 'fill="#373e47"')  # header bar → darker grey
    text = text.replace('stroke="black"', 'stroke="#c9d1d9"')  # borders → light
    text = text.replace('stroke="#404040"', 'stroke="#768390"')  # arrows → lighter grey
    # Graphviz <text> has no fill attribute (defaults to black); inject a light
    # fill so labels are readable on the dark canvas.
    text = text.replace('<text ', '<text fill="#adbac7" ')
    return text


def _write_class_stub(cls: dict, out_dir: pathlib.Path,
                      xml_dir: pathlib.Path) -> None:
    """One .md per inner class. Mirrors Doxygen's class-page layout:
      * Brief + detailed description for the class itself
      * Summary tables, one per sectiondef kind (Public Member Functions,
        Static Public Member Functions, Protected Attributes, etc.)
      * Detail blocks per member, grouped Doxygen-style into Constructor &
        Destructor Documentation / Member Function Documentation /
        Member Data Documentation / etc.

    Detail blocks are per-member breathe directives (`{doxygenfunction}` /
    `{doxygenvariable}` / `{doxygentypedef}`), not the recursive
    `{doxygenclass} :members:` — the latter is breathe's discrete
    one-signature-per-method layout that the user wanted replaced.

    Falls back to a bare `{doxygenclass}` / `{doxygenstruct}` if the class
    XML can't be read (e.g. XML wasn't regenerated)."""
    page = _class_page_name(cls["refid"])
    out = out_dir / f"{page}.md"
    qualified = cls["qualified"] or cls["name"]
    kind_label = cls["kind"].title()  # "Class" / "Struct"
    title = f"{kind_label} {qualified}"
    # Note: no `{#refid}` anchor in the heading — duplicates the
    # docname-derived target. `_generate_api_stubs` seeds the
    # refid→docname mapping into `_ANCHOR_TO_DOC` for `@ref` resolution.
    lines = [f"# {title}", ""]

    # Collaboration diagram — surface the SVG the legacy Doxygen HTML build
    # already rendered (the XML pipeline disables graphs; see
    # `_find_collaboration_svg`). Copy it next to the stub so Sphinx's image
    # collector publishes it to `_images/`, then reference it relative to the
    # api/ doc. The `images/`/`js_assets/` rewrite in `_translate` doesn't
    # touch this path (no such dir segment), and `_img_xtree` leaves
    # non-contrib image refs unchanged. Absent SVG → section silently omitted.
    _svg = _find_collaboration_svg(cls["refid"], xml_dir.parent / "html")
    _light_name = _dark_name = None
    if _svg is not None:
        import hashlib as _hashlib
        try:
            _raw = _svg.read_text(encoding="utf-8")
            # Two theme variants: light = native Doxygen with a transparent
            # backdrop (white page shows through); dark = recoloured to
            # light-on-dark so it matches docs.opencv.org and blends with the
            # dark page. custom.css shows exactly one per active theme.
            #
            # Filenames are content-hashed: Doxygen names every diagram
            # `<refid>__coll__graph.svg`; if a browser cached an older copy
            # under that fixed name it would keep serving the stale image
            # (query-string busts don't always work — some caches key on path
            # only). A hashed filename is a brand-new URL whenever the SVG
            # content changes, so it can never be served stale.
            _light_txt = _svg_make_transparent(_raw)
            _dark_txt = _svg_dark_variant(_raw)
            _lh = _hashlib.md5(_light_txt.encode("utf-8")).hexdigest()[:10]
            _dh = _hashlib.md5(_dark_txt.encode("utf-8")).hexdigest()[:10]
            _light_name = f"{_svg.stem}.{_lh}.svg"
            _dark_name = f"{_svg.stem}.{_dh}.dark.svg"
            (out_dir / _light_name).write_text(_light_txt, encoding="utf-8")
            (out_dir / _dark_name).write_text(_dark_txt, encoding="utf-8")
        except OSError:
            _light_name = _dark_name = None
    if _light_name is not None:
        # `only-light` / `only-dark` are pydata-sphinx-theme's native
        # theme-aware image classes: the theme shows exactly one per active
        # colour mode (via `display:none !important`), and — critically —
        # exempts `.only-dark` images from its
        # `html[data-theme=dark] .bd-content img { background:#fff }` rule, so
        # our dark (transparent-backdrop) variant blends with the dark page
        # instead of getting a white card behind it.
        lines += [
            f"Collaboration diagram for {qualified}:",
            "",
            f"![Collaboration diagram for {qualified}]({_light_name})"
            "{.opencv-coll-graph .only-light}",
            "",
            f"![Collaboration diagram for {qualified}]({_dark_name})"
            "{.opencv-coll-graph .only-dark}",
            "",
        ]

    data = _read_class_data(cls["refid"], xml_dir)
    if data is None:
        # Fallback for missing XML.
        directive = "doxygenstruct" if cls["kind"] == "struct" else "doxygenclass"
        lines += [
            f"```{{{directive}}} {qualified}",
            ":project: opencv",
            ":members:",
            ":protected-members:",
            ":undoc-members:",
            "```",
            "",
        ]
        out.write_text("\n".join(lines), encoding="utf-8")
        return

    # 1) Summary tables in Doxygen's order.
    for sd_kind, summary_title in _CLASS_SUMMARY_SECTIONS:
        items = data["sections"].get(sd_kind, [])
        if not items:
            continue
        lines.append(f"## {summary_title}")
        lines.append("")
        # Type-bearing sections (functions, variables, typedefs) get a
        # Return/Type column. Enum-bearing public-type sections get
        # rendered as code-block synopses instead (matches the group-page
        # treatment).
        non_enum_items = [m for m in items if m["kind"] != "enum"]
        enum_items = [m for m in items if m["kind"] == "enum"]
        if non_enum_items:
            lines += ["| Return | Name | Description |", "|---|---|---|"]
            for m in non_enum_items:
                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                if m["static"]:
                    ret = "static " + ret
                sig = f"{m['name']}{_md_escape_cell(m['args'])}"
                sig_link = f"[`{sig}`](#{m['id']})"
                lines.append(
                    f"| `{ret}` | {sig_link} | {_md_escape_cell(m['brief'])} |")
            lines.append("")
        for m in enum_items:
            if m["brief"]:
                lines.append(_md_escape_cell(m["brief"]))
                lines.append("")
            lines.append("```cpp")
            lines.extend(_enum_synopsis_lines(m))
            lines.append("```")
            lines.append("")

    # 2) "Detailed Description" section for the class itself.
    if data["detailed"]:
        lines += ["## Detailed Description", "", data["detailed"], ""]

    # 3) Per-member detail blocks. Functions split into ctor/dtor vs
    #    others (mirrors Doxygen's "Constructor & Destructor Documentation"
    #    + "Member Function Documentation"). Variables → "Member Data
    #    Documentation". Typedefs → "Member Typedef Documentation".
    #    Enums → "Member Enumeration Documentation" (synopsis code block).
    class_simple = qualified.rsplit("::", 1)[-1]

    # Doxygen leaves <qualifiedname> empty for the members of some classes
    # (cv::_InputArray is one), so `m["qualified"]` falls back to the bare
    # member name. A bare name makes breathe search the *whole* project: it
    # resolves only when the name+signature is unique across all documented
    # symbols (e.g. `channels(int) const`), but common methods shared with
    # other classes stay ambiguous — `copyTo`, `empty`, `getFlags`, `size`
    # all collide with Mat/UMat/etc. and render as "Unable to resolve".
    # Scope every member to this class so the lookup is unambiguous.
    def _scoped(m: dict) -> str:
        q = m.get("qualified") or m["name"]
        return q if "::" in q else f"{qualified}::{m['name']}"

    typedef_items: list[dict] = []
    enum_items_all: list[dict] = []
    ctor_dtor_items: list[dict] = []
    func_items: list[dict] = []
    var_items: list[dict] = []
    for sd_items in data["sections"].values():
        for m in sd_items:
            if m["kind"] == "typedef":
                typedef_items.append(m)
            elif m["kind"] == "enum":
                enum_items_all.append(m)
            elif m["kind"] == "function":
                if m["name"] == class_simple or m["name"] == f"~{class_simple}":
                    ctor_dtor_items.append(m)
                else:
                    func_items.append(m)
            elif m["kind"] == "variable":
                var_items.append(m)

    def _emit_member_directive(m: dict, directive: str, spec: str) -> list[str]:
        # MyST anchor label so the summary-table `#refid` link resolves.
        return [
            f"({m['id']})=",
            f"```{{{directive}}} {spec}",
            ":project: opencv",
            "```",
            "",
        ]

    if typedef_items:
        lines += ["## Member Typedef Documentation", ""]
        for m in typedef_items:
            lines += _emit_member_directive(m, "doxygentypedef", _scoped(m))

    if enum_items_all:
        # Enums render as code-block synopses (matches the group-page
        # treatment — breathe's `{doxygenenum}` gives the discrete
        # one-signature-per-value layout the user wanted replaced).
        lines += ["## Member Enumeration Documentation", ""]
        for m in enum_items_all:
            lines.append(f"({m['id']})=")
            if m["brief"]:
                lines.append(_md_escape_cell(m["brief"]))
                lines.append("")
            lines.append("```cpp")
            lines.extend(_enum_synopsis_lines(m))
            lines.append("```")
            lines.append("")

    # Dedupe functions: a method can appear in multiple sectiondefs (e.g.
    # the same memberdef appearing in `public-func` and again via a
    # `<member refid>` we inlined). The refid is unique per memberdef.
    def _dedupe(items: list[dict]) -> list[dict]:
        seen, out = set(), []
        for m in items:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            out.append(m)
        return out

    if ctor_dtor_items:
        lines += ["## Constructor & Destructor Documentation", ""]
        for m in _dedupe(ctor_dtor_items):
            spec = _scoped(m) + _function_signature(m)
            lines += _emit_member_directive(m, "doxygenfunction", spec)

    if func_items:
        lines += ["## Member Function Documentation", ""]
        for m in _dedupe(func_items):
            spec = _scoped(m) + _function_signature(m)
            lines += _emit_member_directive(m, "doxygenfunction", spec)

    if var_items:
        lines += ["## Member Data Documentation", ""]
        for m in _dedupe(var_items):
            lines += _emit_member_directive(m, "doxygenvariable", _scoped(m))

    out.write_text("\n".join(lines), encoding="utf-8")


def _patch_namespace_xml_for_breathe(xml_dir: pathlib.Path,
                                     out_dir: pathlib.Path) -> None:
    """Mirror `xml_dir` into `out_dir` via symlinks, then rewrite every
    *non-group* compound XML to inline `<memberdef>` blocks that exist only
    in the group XML.

    Why: Doxygen lists functions declared inside `@addtogroup` regions as
    `<member refid="group__…">` in the parent namespace XML (for `cv::*`
    free functions) or the parent file XML (for global `hal_ni_*`-style
    functions) — *without* a full `<memberdef>` block. The memberdef lives
    only in the group XML file. breathe's function-by-name lookup walks
    `<memberdef>` blocks in namespace/file XMLs and ignores bare refs, so
    directives like `{doxygenfunction} cv::log` or
    `{doxygenfunction} hal_ni_merge8u` fail with "Cannot find function".

    Patching: for each `<member refid>` in a target compound's sectiondef
    whose id targets `group__…`, we open the group XML, find the matching
    `<memberdef id="…">`, and append it into the compound's sectiondef.
    The original XML on disk is untouched."""
    import xml.etree.ElementTree as _ET
    import os as _osmod, shutil as _shutil
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mirror every file from xml_dir into out_dir as a symlink. Cleaning
    #    out_dir each time keeps this idempotent across rebuilds (Doxygen
    #    XML changes are picked up because the symlinks resolve fresh).
    for child in list(out_dir.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            _shutil.rmtree(child)
    for src in xml_dir.iterdir():
        dst = out_dir / src.name
        try:
            _osmod.symlink(src, dst)
        except (OSError, NotImplementedError):
            _shutil.copy2(src, dst)

    # 2) Cache for parsed group XMLs (each is read once even if referenced
    #    by many compounds).
    _group_cache: dict[str, _ET.ElementTree] = {}

    def _load_group(group_id: str):
        if group_id in _group_cache:
            return _group_cache[group_id]
        gx = xml_dir / f"{group_id}.xml"
        if not gx.is_file():
            _group_cache[group_id] = None
            return None
        try:
            _group_cache[group_id] = _ET.parse(gx)
        except _ET.ParseError:
            _group_cache[group_id] = None
        return _group_cache[group_id]

    # 3) For each non-group compound XML (namespace, file, dir, …) patch
    #    `<member refid>` entries that point at group memberdefs.
    #    `index.xml` is the project index (not a compound) → skip it.
    #    `class*.xml`/`struct*.xml`/`union*.xml` already carry full
    #    memberdefs for their methods, but they may *also* have
    #    `<member refid>` from @addtogroup tagged methods — patch them too.
    _SKIP_PREFIXES = ("group", "index")
    for compound_file in xml_dir.glob("*.xml"):
        if any(compound_file.name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        try:
            tree = _ET.parse(compound_file)
        except _ET.ParseError:
            continue
        root = tree.getroot()
        cd = root.find("compounddef")
        if cd is None:
            continue
        existing_ids = {md.get("id") for md in cd.iter("memberdef")}
        patched = False
        for sd in cd.findall("sectiondef"):
            for member in list(sd.findall("member")):
                refid = member.get("refid", "")
                if not refid or refid in existing_ids:
                    continue
                # Member refids inside groups look like
                # "group__core__utils__softfloat_1ga…". The compound id is
                # everything before "_1" (which separates member from
                # compound in Doxygen's id scheme).
                if not refid.startswith("group__"):
                    continue
                sep = refid.find("_1")
                if sep < 0:
                    continue
                group_id = refid[:sep]
                gtree = _load_group(group_id)
                if gtree is None:
                    continue
                for md in gtree.getroot().iter("memberdef"):
                    if md.get("id") == refid:
                        sd.append(md)
                        existing_ids.add(refid)
                        patched = True
                        break
        if patched:
            out_file = out_dir / compound_file.name
            if out_file.is_symlink() or out_file.is_file():
                out_file.unlink()
            tree.write(out_file, encoding="utf-8", xml_declaration=True)


def _generate_api_stubs(modules, xml_dir, out_dir):
    """Generate the full api/ stub tree. Idempotent — wipes and regenerates
    on every sphinx-build so stale stubs from removed modules disappear.

    Two passes:
      1. Walk each module's group hierarchy → emit one group .md per node
         (parent index pages + leaf pages with summary tables + per-member
         detail blocks). `classes_seen` is populated as a side-effect.
      2. Emit one .md per unique inner class — these are the per-class
         subpages the group pages link to."""
    if not modules:
        return
    if not xml_dir.is_dir():
        return  # No XML yet (sphinx-xml not run); degrade silently.
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_lines = [
        "API Reference {#api_root}",
        "=============",
        "",
        "Sphinx-rendered API reference for OpenCV main modules. Each entry",
        "below is a module's umbrella `@defgroup`; sub-pages mirror the",
        "Doxygen subgroup hierarchy.",
        "",
    ]
    classes_seen: dict[str, dict] = {}
    for m in modules:
        tree = _build_api_hierarchy(
            "group__" + m.replace("_", "__"), xml_dir)
        if tree is None:
            continue
        root_lines.append(f"- @subpage api_{tree['name']}")
        _write_api_stub(tree, out_dir, classes_seen)
    # Per-class pages (one per unique refid across all groups). We also
    # seed `_ANCHOR_TO_DOC` directly with refid -> docname so `@ref`
    # cross-references in tutorial markdown (and in any group page) keep
    # working — the per-class page no longer carries a `{#refid}` heading
    # anchor (would duplicate the docname-derived target).
    for cls in classes_seen.values():
        _write_class_stub(cls, out_dir, xml_dir)
        _ANCHOR_TO_DOC[cls["refid"]] = f"api/{_class_page_name(cls['refid'])}"
    (out_dir / "api_root.markdown").write_text(
        "\n".join(root_lines) + "\n", encoding="utf-8")


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

# Doxygen flattens IMAGE_PATH across every `images/` folder under the tutorial
# tree plus the top-level `doc/images/` (per Doxyfile.in). Mirror that behavior
# by building a basename -> doc-root-relative-path index once at import time.
# js_tutorials/ uses `js_assets/` (flat directory at the js_tutorials root)
# instead of per-tutorial `images/` folders. Include both lookup roots so
# `![alt](js_assets/foo.jpg)` references resolve the same way. Contrib trees
# are scanned below: under <m>/tutorials they're indexed against the staged
# tutorials_contrib/<m> symlink; outside <m>/tutorials they use a
# `contrib_modules/<rel>` URL served from html_extra_path (see below).
_IMAGE_INDEX: dict[str, str] = {}
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
        # Prepend `intro` before the first existing module bullet so it
        # leads the sidebar — matches the order in opencv/doc/root.markdown.in
        # ("- @ref intro" sits above "- @ref tutorial_root"). FAQ and
        # Bibliography stay appended at the end (still after the modules).
        if "intro" in _ANCHOR_TO_DOC:
            text = re.sub(r"^- @subpage", "- @subpage intro\n- @subpage",
                          text, count=1, flags=re.MULTILINE)
        if JS_DOC_MODULES:
            text += "\n- @subpage tutorial_js_root\n"
        if PY_DOC_MODULES:
            text += "\n- @subpage tutorial_py_root\n"
        if "faq" in _ANCHOR_TO_DOC:
            text += "\n- @subpage faq\n"
        if "citelist" in _ANCHOR_TO_DOC:
            text += "\n- @subpage citelist\n"

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

    # 3. @code{.lang} ... @endcode → fenced block. Preserve the indent
    #    so blocks nested under a bullet item stay inside the list; for
    #    col-0 @code keep the legacy .strip() form (byte-identical).
    def _code_repl(m: re.Match) -> str:
        indent = m.group("indent") or ""
        lang = _normalize_lang(m.group("lang") or "")
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
    # ({code-block} and `:::` colon-fence forms break inside tab-items.)
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
            rf"^[ \t]*-\s+(?P<prefix>[^\n@]*?)@(?P<kind>subpage|ref)\s+(?P<anchor>[\w-]+)"
            rf'(?:[ \t]+"(?P<disp>[^"\n]+)")?'   # optional `@ref X "Display"`
            rf"(?P<inline>[^\n]*)\n"             # rest of line — prose like ` - desc`
            rf"(?P<desc>{desc_re})",
            re.MULTILINE)

        def repl(m: re.Match) -> str:
            # (kind, doctype, target, title, desc, prefix, inline)
            resolved: list[tuple[str, str, str, str, str, str, str]] = []
            for im in item_pat.finditer(m.group(1)):
                kind = im.group("kind")  # "subpage" or "ref"
                anchor = im.group("anchor")
                disp = im.group("disp")  # explicit `"Display"` overrides scanner title
                # Bullet prefix (e.g. `stitching. ` in `-   stitching. @subpage X`)
                # — kept as plain text before the link so the "others" TOC
                # reads "stitching. High level stitching API" the way Doxygen
                # renders it.
                prefix = (im.group("prefix") or "").strip()
                # Same-line content after the anchor (e.g. ` - build and install...`
                # in tutorials.markdown, or ` (**core**) - a compact module...` in
                # intro.markdown). Previously discarded; keeping it preserves the
                # author's prose and stops bullet lists like intro's module rundown
                # from rendering as link-only stubs.
                inline = im.group("inline") or ""
                desc_lines = [l.strip() for l in (im.group("desc") or "").splitlines() if l.strip()]
                description = " ".join(desc_lines)
                # `@ref` follows redirect chains; `@subpage` does not (it
                # determines navigation, so we want the literal target).
                lookup = _resolve_redirect(anchor) if kind == "ref" else anchor
                if lookup in _ANCHOR_TO_DOC:
                    resolved.append((kind, "internal", _ANCHOR_TO_DOC[lookup],
                                     disp or _ANCHOR_TO_TITLE.get(lookup, lookup),
                                     description, prefix, inline))
                elif lookup in _ANCHOR_TO_EXTERNAL:
                    title, url = _ANCHOR_TO_EXTERNAL[lookup]
                    resolved.append((kind, "external", url, disp or title,
                                     description, prefix, inline))
                elif lookup in _TAG_FILENAMES:
                    title = _TAG_TITLES.get(lookup, lookup)
                    resolved.append((kind, "external", _doxygen_url(lookup),
                                     disp or title, description, prefix, inline))
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
            for t in resolved:
                kind, doctype, target, title = t[0], t[1], t[2], t[3]
                if kind != "subpage":
                    continue
                tt_lines.append("/" + target if doctype == "internal"
                                else f"{title} <{target}>")
            tt_body = "\n".join(tt_lines)

            has_descriptions = any(t[4] for t in resolved)
            has_prefixes     = any(t[5] for t in resolved)
            # `inline` (same-line trailing content) alone does NOT trigger
            # the visible-list mode — tutorials.markdown's master nav uses
            # the `- @subpage X - one-line label` shape and was always meant
            # to render as a plain toctree (cleaner sidebar, no body bullets).
            # Visible mode is reserved for bullets with continuation lines
            # (intro module rundown, py-tutorials TOCs) or a prefix label
            # ("others" TOC). Inline still shows when visible mode is
            # already triggered for other reasons.
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
            # Same-line `inline` content stays attached to the link (matches
            # the source layout — e.g. `[link] - desc` for nav-style bullets
            # or `[link] (**core**) - prose…` for intro's module rundown).
            # `desc` (multi-line continuation) keeps its paragraph break so
            # py-tutorials TOCs still render their description blocks below.
            list_lines = []
            for t in resolved:
                doctype, target, title, desc, prefix, inline = t[1], t[2], t[3], t[4], t[5], t[6]
                href = f"/{target}" if doctype == "internal" else target
                prefix_text = f"{prefix} " if prefix else ""
                list_lines.append(f"- {prefix_text}[{title}]({href}){inline}")
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
        # Internal citelist when the page got staged + scanned; otherwise
        # fall back to the Doxygen build's citelist.html (e.g. ad-hoc
        # sphinx-build with SPHINX_INPUT_ROOT == DOC_ROOT, where the
        # generated page isn't written). The depth-relative URL matches
        # the same trick used for contrib images.
        if "citelist" in _ANCHOR_TO_DOC:
            depth = docname.count("/") if docname else 0
            href = ("../" * depth) + f"citelist.html#CITEREF_{key}"
        else:
            href = f"{DOXYGEN_BASE_URL}citelist.html#CITEREF_{key}"
        return f'<a href="{href}">{label}</a>'
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

    # 10. @next_tutorial / @prev_tutorial  -> drop
    text = re.sub(r"^@(?:next|prev)_tutorial\{[^}]*\}\s*$", "",
                  text, flags=re.MULTILINE)

    # 11. @tableofcontents / [TOC] -> drop. PyData's right sidebar
    #     already shows the per-page outline.
    text = re.sub(r"^(?:@tableofcontents|\[TOC\])\s*$", "",
                  text, flags=re.MULTILINE)

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

    # Depth-relative prefix from the current doc back to the site root,
    # used to point `<img src>` at `<output>/contrib_modules/...` files
    # that html_extra_path publishes (Sphinx can't pathto-rewrite URLs
    # for files outside srcdir, so we compute the ../ count ourselves).
    _depth = docname.count("/") if docname else 0
    _contrib_url_prefix = ("../" * _depth) + "contrib_modules/"

    def _emit_contrib_img(rel_url: str, alt: str) -> str:
        """Raw-HTML <img> (or <figure> if alt is 'Figure ...') for a
        contrib_modules/<rel> path — bypasses Sphinx's image processing
        so the depth-relative URL survives to the rendered HTML."""
        src = _contrib_url_prefix + rel_url
        img = f'<img src="{src}" alt="{alt}"/>'
        if alt.startswith("Figure "):
            return (f'<figure>{img}'
                    f'<figcaption>{alt}</figcaption></figure>')
        return img

    # 12. Image paths "images/foo.png" or "js_assets/foo.jpg" — resolve like
    #     Doxygen's flat IMAGE_PATH: prefer the doc's own asset sibling, then
    #     fall back to a global basename lookup across every tutorial `images/`
    #     folder plus `js_tutorials/js_assets/`. As a final fallback, point at
    #     the consolidated `tutorials/others/images/` dir. For contrib pages,
    #     resolves under `<m>/tutorials/<rest>/images/`. Also matches
    #     `<prefix>/images/...` — some tutorials prefix the directory name
    #     (multiview_calibration does this) which Doxygen IMAGE_PATH ignores
    #     but MyST would resolve literally and miss.
    def _img_repl(m: re.Match) -> str:
        alt = m.group("alt")
        rel = m.group("rel")
        asset_dir = m.group("dir")
        if docname:
            parts = pathlib.Path(docname).parent.parts
            local = None
            if parts and parts[0] in ("tutorials", "js_tutorials", "py_tutorials"):
                local = DOC_ROOT / pathlib.Path(docname).parent / asset_dir / rel
            elif len(parts) >= 2 and parts[0] == "tutorials_contrib":
                # Contrib doc → resolve under <m>/tutorials/<rest>/images/.
                rest = pathlib.Path(*parts[2:]) if len(parts) > 2 else pathlib.Path()
                local = CONTRIB_ROOT / parts[1] / "tutorials" / rest / asset_dir / rel
            if local is not None and local.is_file():
                return f'![{alt}]({asset_dir}/{rel})'
        hit = _IMAGE_INDEX.get(pathlib.Path(rel).name)
        if hit:
            if hit.startswith("contrib_modules/"):
                return _emit_contrib_img(hit[len("contrib_modules/"):], alt)
            return f'![{alt}](/{hit})'
        return f'![{alt}](/tutorials/others/images/{rel})'
    text = re.sub(
        r'!\[(?P<alt>[^\]]*)\]\((?:[^)]*?/)?(?P<dir>images|js_assets)/(?P<rel>[^)]+)\)',
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
    # Translate any tutorial doc — the root index, everything under an enabled
    # main / js / py module, plus (when staged) everything under an enabled
    # contrib module. Also translate API stubs so their `@subpage` / `@ref`
    # lines turn into proper toctree entries; the body's MyST `{doxygengroup}`
    # blocks pass through untouched (no `@` directives to rewrite).
    if not (docname.startswith("tutorials/")
            or docname.startswith("js_tutorials/")
            or docname.startswith("py_tutorials/")
            or docname.startswith("tutorials_contrib/")
            or docname.startswith("api/")
            or docname == "faq"
            or docname == "citelist"
            or docname == "intro"):
        return
    text = source[0]
    # On the master doc, append `- @subpage tutorial_contrib_root` / `api_root`
    # so the contrib + API sites appear in the unified left sidebar without
    # modifying opencv/doc/tutorials/tutorials.markdown on disk.
    if docname == "tutorials/tutorials":
        if CONTRIB_MODULES and "tutorial_contrib_root" in _ANCHOR_TO_DOC:
            text = text.rstrip() + "\n\n- @subpage tutorial_contrib_root\n"
        if API_MODULES and "api_root" in _ANCHOR_TO_DOC:
            text = text.rstrip() + "\n\n- @subpage api_root\n"
    source[0] = _translate(text, docname)


def _patch_cpp_xref_resolver():
    """Guard the C++ domain's xref resolver against an upstream assertion on
    template-class cross-references. Sphinx 8.1.x asserts `parentSymbol`
    inside `_resolve_xref_inner`; some breathe-emitted class-page xrefs
    (e.g. `cv::Affine3<T>`) trigger that path with no parent symbol.
    Treat it as an unresolved xref instead of crashing the whole build."""
    try:
        from sphinx.domains.cpp import CPPDomain
    except ImportError:
        return
    original = CPPDomain._resolve_xref_inner

    def guarded(self, env, fromdocname, builder, typ, target, node, contnode):
        try:
            return original(self, env, fromdocname, builder, typ, target,
                            node, contnode)
        except AssertionError:
            return None, None
    CPPDomain._resolve_xref_inner = guarded


_patch_cpp_xref_resolver()


def _silence_breathe_anon_enum_warning():
    """Suppress the docutils "Invalid C++ declaration: Expected identifier
    in nested name." warning that breathe triggers when rendering an
    *anonymous* nested enum inside a struct (e.g. `cv::MatShape` has an
    enum whose `<name>` element is empty — Doxygen XML allows it, but the
    Sphinx C++ domain parser rejects the resulting declaration).

    The render is otherwise fine (the enum values still appear); only the
    parse-time warning is noise. We filter it via a Python logging filter
    rather than monkey-patching the parser so the same fix survives Sphinx
    version bumps."""
    import logging
    class _AnonEnumFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not (
                "Invalid C++ declaration" in msg
                and "Expected identifier in nested name" in msg
            )
    # docutils warning messages route through both 'sphinx' and 'docutils'
    # loggers depending on entry point; attach to both for coverage.
    for _name in ("sphinx", "docutils"):
        logging.getLogger(_name).addFilter(_AnonEnumFilter())


_silence_breathe_anon_enum_warning()


def setup(app):
    app.connect("source-read", _source_read)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
