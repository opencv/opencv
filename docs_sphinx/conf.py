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

source_suffix = {".md": "markdown", ".markdown": "markdown"}

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
    # contrib module.
    if not (docname.startswith("tutorials/")
            or docname.startswith("js_tutorials/")
            or docname.startswith("py_tutorials/")
            or docname.startswith("tutorials_contrib/")
            or docname == "faq"
            or docname == "citelist"
            or docname == "intro"):
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
