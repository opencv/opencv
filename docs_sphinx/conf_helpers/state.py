"""Shared configuration & state for the OpenCV Sphinx wrapper.

Owns everything the doc-build engine reads in common: env-derived paths and
module lists, the Doxygen tag-file maps, bib/citation numbering, the redirect
map, the anchor indexes, and the small constant tables. The sibling engines
(xml_render, stubs, translate, postprocess) pull these via
``from .state import *``; conf.py imports the handful it needs for Sphinx
settings.
"""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap

HERE = pathlib.Path(__file__).resolve().parent.parent
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

# sphinx_design availability. Extension wiring lives in conf.py; _emit_toggles
# uses this flag to choose a {tab-set} vs a plain labeled-section fallback.
try:
    import sphinx_design as _sphinx_design  # noqa: F401
    HAVE_SPHINX_DESIGN = True
except ImportError:
    HAVE_SPHINX_DESIGN = False

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

# -- Breathe availability ----------------------------------------------------
# Extension registration + breathe_projects config live in conf.py; here we
# only detect breathe (its absence empties API_MODULES).
HAVE_BREATHE = False
if API_MODULES:
    try:
        import breathe  # noqa: F401
        HAVE_BREATHE = True
    except ImportError:
        API_MODULES = []

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

# Basename indexes populated at import time by the build module (kept here so
# the translation engine and the build orchestrator share the same dict
# objects).
_IMAGE_INDEX: dict[str, str] = {}
_SNIPPET_INDEX: dict[str, pathlib.Path] = {}

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

__all__ = [
    "HERE", "DOC_ROOT", "OPENCV_ROOT",
    "DOC_MODULES", "JS_DOC_MODULES", "PY_DOC_MODULES",
    "CONTRIB_MODULES", "CONTRIB_ROOT", "SPHINX_INPUT_ROOT", "API_MODULES",
    "_API_XML_DIR", "_PATCHED_XML_DIR",
    "HAVE_SPHINX_DESIGN", "HAVE_BREATHE",
    "DOXYGEN_BASE_URL", "_doxygen_url",
    "_TAG_FILE", "_TAG_FILENAMES", "_TAG_TITLES", "_CV_SYMBOL_URL",
    "_CITE_NUMBER", "_BIB_ENTRIES_SORTED", "_bib_render_all",
    "_REDIRECT_MAP", "_resolve_redirect",
    "_ANCHOR_TO_DOC", "_ANCHOR_TO_EXTERNAL", "_ANCHOR_TO_TITLE", "_HEAD_RE",
    "_scan_internal", "_scan_external",
    "_IMAGE_INDEX", "_SNIPPET_INDEX", "_SNIPPET_BASES",
    "_TOGGLE_LABELS", "_LANG_ALIASES",
]
