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
    for m in (_os.environ.get("OPENCV_DOC_MODULES") or "photo,objdetect,core,calib3d,features,introduction").split(",")
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
]
exclude_patterns = [
    "**/Thumbs.db", "**/.DS_Store",
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

def _scan_internal(path: pathlib.Path) -> None:
    """Add every {#anchor} and standalone `@anchor NAME` in `path` (file or
    dir) to _ANCHOR_TO_DOC."""
    files = [path] if (path.is_file() and path.suffix == ".markdown") \
        else (list(path.rglob("*.markdown")) if path.is_dir() else [])
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

# External scan: every OTHER module's top-level table_of_content_*.markdown.
for _toc in (DOC_ROOT / "tutorials").glob("*/table_of_content_*.markdown"):
    if _toc.parent.name not in DOC_MODULES:
        _scan_external(_toc)

# Doxygen flattens IMAGE_PATH across every `images/` folder under the tutorial
# tree, so a tutorial can reference `images/foo.png` even when `foo.png` lives
# in a sibling module's `images/` directory. Mirror that behavior by building
# a basename -> doc-root-relative-path index once at import time.
_IMAGE_INDEX: dict[str, str] = {}
for _img in (DOC_ROOT / "tutorials").rglob("images/*"):
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

    # 3b. Plain Markdown fences with a Doxygen-flavored language spec
    #     (e.g. "```.sh") confuse Pygments — strip the leading dot and apply
    #     the same alias map as @code{.lang}.
    text = re.sub(
        r"^(?P<fence>`{3,})(?P<lang>\.?[\w-]+)[ \t]*$",
        lambda m: f"{m.group('fence')}{_normalize_lang(m.group('lang'))}",
        text, flags=re.MULTILINE)

    # @include and @snippet both emit a plain backtick fence with the
    # captured leading indent prefixed to every body line so the fence
    # stays within an enclosing list-item content scope. Plain fences are
    # the only form that survives all observed contexts:
    #   - {code-block} directive form gets swallowed as raw text inside
    #     sphinx_design's `{tab-item}` blocks.
    #   - `:::{code-block}` colon-fence form collides with `{tab-item}`'s
    #     option parser (`:key: value`).
    # CommonMark strips up to N spaces of leading indent from each code
    # line when the fence is indented N spaces, so the rendered code body
    # retains zero leading whitespace.
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

    # 8. @cite KEY -> [KEY]
    text = re.sub(r"@cite\s+([\w-]+)", r"[\1]", text)

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

    # 9. @subpage NAME  (collected blocks -> real toctree).
    #    Enabled modules' anchors become internal toctree entries.
    #    Disabled modules' anchors become external links into the Doxygen
    #    build, so the left sidebar still shows the full module list.
    def _subpage_list_to_toctree(src: str) -> str:
        pat = re.compile(
            r"((?:^[ \t]*-\s+@subpage\s+[\w-]+(?:[^\n]*)\n)+)", re.MULTILINE)
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

    # 12. Image paths "images/foo.png" — resolve like Doxygen's flat IMAGE_PATH:
    #     prefer the doc's own "images/" sibling, then fall back to a global
    #     basename lookup across every tutorial `images/` folder. As a final
    #     fallback, point at the consolidated `tutorials/others/images/` dir
    #     (where modules like `photo` store their assets).
    #     Also matches `<prefix>/images/foo.png` — some tutorials prefix the
    #     directory name (multiview_calibration does this) which Doxygen
    #     IMAGE_PATH ignores but MyST would resolve literally and miss.
    def _img_repl(m: re.Match) -> str:
        rel = m.group("rel")
        if docname:
            local = DOC_ROOT / pathlib.Path(docname).parent / "images" / rel
            if local.is_file():
                return f'{m.group("pre")}images/{rel})'
        hit = _IMAGE_INDEX.get(pathlib.Path(rel).name)
        if hit:
            return f'{m.group("pre")}/{hit})'
        return f'{m.group("pre")}/tutorials/others/images/{rel})'
    text = re.sub(
        r'(?P<pre>!\[[^\]]*\]\()(?:[^)]*?/)?images/(?P<rel>[^)]+)\)',
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

    # 13b. Auto-linkify bare URLs anywhere outside code by wrapping them in
    #      CommonMark `<url>` autolink syntax so MyST renders them as
    #      clickable links (Doxygen does this by default; the alternative
    #      would be installing the `linkify-it-py` package). Skip:
    #        - URLs inside fenced code blocks (line-by-line fence tracker)
    #        - URLs inside inline `code` spans
    #        - URLs already inside markdown link syntax `[text](url)`
    #          (lookbehind `](`)
    #        - URLs already wrapped in autolink `<url>` (lookbehind `<`)
    #        - URLs inside HTML attributes like the @youtube iframe `src="…"`
    #          (lookbehinds `="` and `='`)
    #      Trailing sentence punctuation (.,;:!?) is left outside the
    #      autolink so `See https://x.com.` becomes `See <https://x.com>.`
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
    # Translate any tutorial doc — the root index plus everything under a
    # module we enabled in DOC_MODULES.
    if not docname.startswith("tutorials/"):
        return
    source[0] = _translate(source[0], docname)


def setup(app):
    app.connect("source-read", _source_read)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
