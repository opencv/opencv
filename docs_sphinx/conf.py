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
    for m in (_os.environ.get("OPENCV_DOC_MODULES") or "photo,objdetect,dnn,gpu,others").split(",")
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
] + (["js_tutorials/js_tutorials.markdown"] if JS_DOC_MODULES else []) + [
    f"js_tutorials/{m}/**" for m in JS_DOC_MODULES
]
exclude_patterns = ["**/Thumbs.db", "**/.DS_Store", "**/_old/**"]

myst_enable_extensions = [
    "colon_fence", "deflist", "dollarmath", "amsmath",
    "attrs_inline", "attrs_block", "smartquotes",
]
myst_heading_anchors = 4
suppress_warnings = ["myst.header", "myst.xref_missing", "toc.not_included"]

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

def _scan_internal(path: pathlib.Path) -> None:
    """Add every {#anchor} in `path` (file or dir) to _ANCHOR_TO_DOC.
    Picks up both `.markdown` (the bulk of the tree) and `.md` (the form
    used by ports like dnn/dnn_pytorch_tf_*)."""
    _SUFFIXES = (".markdown", ".md")
    if path.is_file():
        files = [path] if path.suffix in _SUFFIXES else []
    elif path.is_dir():
        files = [p for s in _SUFFIXES for p in path.rglob(f"*{s}")]
    else:
        files = []
    for md in files:
        try:
            head = md.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        rel = md.relative_to(DOC_ROOT).with_suffix("").as_posix()
        for m in re.finditer(r"\{#([\w-]+)\}", head):
            _ANCHOR_TO_DOC[m.group(1)] = rel
        # Capture the first heading's title alongside its anchor so subpage
        # lists with descriptions can render with real link text.
        tm = _HEAD_RE.search(head)
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

# Internal scan: master + every enabled module subtree.
_scan_internal(DOC_ROOT / "tutorials" / "tutorials.markdown")
for _m in DOC_MODULES:
    _scan_internal(DOC_ROOT / "tutorials" / _m)
if JS_DOC_MODULES:
    _scan_internal(DOC_ROOT / "js_tutorials" / "js_tutorials.markdown")
for _m in JS_DOC_MODULES:
    _scan_internal(DOC_ROOT / "js_tutorials" / _m)

# External scan: every OTHER module's top-level table_of_content_*.markdown.
for _toc in (DOC_ROOT / "tutorials").glob("*/table_of_content_*.markdown"):
    if _toc.parent.name not in DOC_MODULES:
        _scan_external(_toc)
# Same for js_tutorials (files are named js_table_of_contents_*.markdown there).
for _toc in (DOC_ROOT / "js_tutorials").glob("*/js_table_of_contents_*.markdown"):
    if _toc.parent.name not in JS_DOC_MODULES:
        _scan_external(_toc)

# Doxygen flattens IMAGE_PATH across every `images/` folder under the tutorial
# tree plus the top-level `doc/images/` (per Doxyfile.in), so a tutorial can
# reference `images/foo.png` even when `foo.png` lives in a sibling module's
# `images/` directory or in `doc/images/`. Mirror that behavior by building a
# basename -> doc-root-relative-path index once at import time.
_IMAGE_INDEX: dict[str, str] = {}
# js_tutorials/ uses `js_assets/` (flat directory at the js_tutorials root)
# instead of per-tutorial `images/` folders. Include both lookup roots so
# `![alt](js_assets/foo.jpg)` references resolve the same way.
for _root in ((DOC_ROOT / "tutorials").rglob("images/*"),
              (DOC_ROOT / "js_tutorials").rglob("images/*"),
              (DOC_ROOT / "js_tutorials" / "js_assets").glob("*"),
              (DOC_ROOT / "images").glob("*")):
    for _img in _root:
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


def _read_snippet(rel_path: str, label: str | None) -> tuple[str, str]:
    """Return (code_text, language) for an @include / @snippet directive."""
    p = next((b / rel_path for b in _SNIPPET_BASES
              if (b / rel_path).is_file()), None)
    if p is None:
        return f"// not found: {rel_path}\n", "text"
    text = p.read_text(encoding="utf-8", errors="replace")
    ext = p.suffix.lower()
    lang = {".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".c": "c",
            ".py": "python", ".java": "java"}.get(ext, "text")
    if label is None:
        return text, lang
    # Doxygen matches `[label]` after any comment-style marker (//, //!, #, ##)
    # anywhere on a line — including labels wrapped in block-comments like
    # `/* //! [label]` or `//! [label] */`.
    pat = re.compile(r"^[^\[\n]*(?://!|//|##|#)[^\[\n]*\[" + re.escape(label)
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
    # 0. Master doc: synthesize a @subpage entry for the js_tutorials root so
    #    step 9 picks it up as a toctree entry. tutorials.markdown has no
    #    direct reference to js_tutorials, and editing it is forbidden, so the
    #    only way to surface js content in the master sidebar is to inject.
    if docname == "tutorials/tutorials" and JS_DOC_MODULES:
        text += "\n- @subpage tutorial_js_root\n"

    # 0b. Doxygen automatic-numbered list items: "-# foo" -> "1. foo". MyST /
    #     CommonMark sequentially numbers identical-marker ordered lists, so
    #     successive "1." items render as 1, 2, 3, ...
    text = re.sub(r"^(?P<indent>[ \t]*)-#[ \t]+",
                  lambda m: f"{m.group('indent')}1. ", text, flags=re.MULTILINE)

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

    # 3. @code{.lang} ... @endcode
    def _code_repl(m: re.Match) -> str:
        lang = (m.group("lang") or "").strip(".") or "text"
        if lang == "none":
            lang = "text"
        return f"\n```{lang}\n{m.group('body').strip()}\n```\n"
    text = re.sub(
        r"@code(?:\{(?P<lang>[^}]*)\})?\s*\n(?P<body>.*?)\n\s*@endcode",
        _code_repl, text, flags=re.DOTALL)

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
    #     Matches the lang token on its own — won't touch `yamllint`, `yml-foo`,
    #     or strings like ` ```yml-config` (none currently exist, but the \b
    #     guard keeps this safe under future edits).
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

    # 4. @include path  /  @includelineno path
    #    (line numbering hint is dropped — MyST fenced blocks don't take :linenos:
    #    and PyData's code-block styling is already legible without it.)
    def _include_repl(m: re.Match) -> str:
        code, lang = _read_snippet(m.group("path"), None)
        return f"\n```{lang}\n{code.rstrip()}\n```\n"
    text = re.sub(r"@include(?:lineno)?\s+(?P<path>\S+)", _include_repl, text)

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
        return f"\n```{lang}\n{code.rstrip()}\n```\n"
    text = re.sub(r"@snippet\s+(?P<path>\S+)\s+(?P<label>[^\n]+?)\s*$",
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

    # 7. @ref name [optional "Display Text"]
    #    Resolution order: enabled-module anchor (internal docname) ->
    #    Doxygen tag (external URL) -> fragment-only fallback. The Doxygen
    #    fallback keeps cross-module refs (e.g. js_setup -> js_imgproc) from
    #    rendering as broken in-page anchors when the target module isn't
    #    onboarded into the Sphinx wrapper yet.
    def _ref_repl(m: re.Match) -> str:
        name = m.group("name"); disp = m.group("disp")
        target = _ANCHOR_TO_DOC.get(name)
        if target:
            return f"[{disp or name}]({'/' + target})"
        if name in _TAG_FILENAMES:
            return f"[{disp or name}]({_doxygen_url(name)})"
        return f"[{disp or name}](#{name})"
    text = re.sub(r'@ref\s+(?P<name>[\w-]+)(?:\s+"(?P<disp>[^"]+)")?',
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
        return f"\n{indent}:::{{{kind}}}\n{body}\n{indent}:::\n"
    text = re.sub(
        r"^(?P<indent>[ \t]*)@(?P<dir>note|see|warning)[ \t]*\n?"
        r"(?P<body>.+?)(?=\n[ \t]*\n|\n[ \t]*@[A-Za-z]|\Z)",
        _admon_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 9. @subpage NAME  (collected blocks -> real toctree).
    #    Enabled modules' anchors become internal toctree entries.
    #    Disabled modules' anchors become external links into the Doxygen
    #    build, so the left sidebar still shows the full module list.
    #    When the source pairs each `- @subpage X` line with an indented
    #    description paragraph (the `js_tutorials` / table_of_contents form),
    #    emit a hidden toctree (for nav) plus a visible bulleted list whose
    #    items show link + description. Without that handling those indented
    #    paragraphs would render as CommonMark code blocks.
    def _subpage_list_to_toctree(src: str) -> str:
        # Bullet items may carry a descriptive prefix before @subpage, e.g.
        # `-   stitching. @subpage tutorial_stitcher` (tutorials/others/...).
        bullet  = r"^[ \t]*-\s+[^\n@]*?@subpage\s+[\w-]+(?:[^\n]*)\n"
        desc_re = r"(?:[ \t]*\n[ \t]+[^\n]+(?:\n[ \t]+[^\n]+)*\n?)*"
        pat = re.compile(rf"((?:{bullet}{desc_re}(?:[ \t]*\n)*)+)", re.MULTILINE)
        item_pat = re.compile(
            rf"^[ \t]*-\s+[^\n@]*?@subpage\s+(?P<anchor>[\w-]+)[^\n]*\n"
            rf"(?P<desc>{desc_re})",
            re.MULTILINE)

        def repl(m: re.Match) -> str:
            resolved: list[tuple[str, str, str, str]] = []
            for im in item_pat.finditer(m.group(1)):
                anchor = im.group("anchor")
                desc_lines = [l.strip() for l in (im.group("desc") or "").splitlines() if l.strip()]
                description = " ".join(desc_lines)
                if anchor in _ANCHOR_TO_DOC:
                    target = _ANCHOR_TO_DOC[anchor]
                    title  = _ANCHOR_TO_TITLE.get(anchor, anchor)
                    resolved.append(("internal", target, title, description))
                elif anchor in _ANCHOR_TO_EXTERNAL:
                    title, url = _ANCHOR_TO_EXTERNAL[anchor]
                    resolved.append(("external", url, title, description))
            if not resolved:
                return ""

            tt_lines = ["/" + t if k == "internal" else f"{n} <{t}>"
                        for k, t, n, _ in resolved]
            tt_body = "\n".join(tt_lines)

            if not any(d for _, _, _, d in resolved):
                return f"\n```{{toctree}}\n:maxdepth: 1\n\n{tt_body}\n```\n"

            # Blank line between link and description forces a "loose" list,
            # so each description renders as its own <p> instead of being
            # merged onto the link's paragraph.
            list_lines = []
            for kind, target, title, desc in resolved:
                href = f"/{target}" if kind == "internal" else target
                list_lines.append(f"- [{title}]({href})")
                if desc:
                    list_lines.append("")
                    list_lines.append(f"  {desc}")
                list_lines.append("")
            return (
                f"\n```{{toctree}}\n:hidden:\n:maxdepth: 1\n\n{tt_body}\n```\n"
                f"\n{chr(10).join(list_lines).rstrip()}\n"
            )
        return pat.sub(repl, src)
    text = _subpage_list_to_toctree(text)

    # 10. @next_tutorial / @prev_tutorial  -> drop
    text = re.sub(r"^@(?:next|prev)_tutorial\{[^}]*\}\s*$", "",
                  text, flags=re.MULTILINE)

    # 11. @tableofcontents -> drop (PyData right sidebar replaces it)
    text = re.sub(r"^@tableofcontents\s*$", "", text, flags=re.MULTILINE)

    # 11b. @cond NAME ... @endcond  -> strip just the markers; if the
    #      enclosed @subpage points to a disabled module it gets dropped
    #      by _subpage_list_to_toctree above.
    text = re.sub(r"^@cond\s+\S+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^@endcond\s*$", "", text, flags=re.MULTILINE)

    # 12. Image paths "images/foo.png" or "js_assets/foo.jpg" — resolve like
    #     Doxygen's flat IMAGE_PATH: prefer the doc's own asset sibling, then
    #     fall back to a global basename lookup across every tutorial `images/`
    #     folder plus `js_tutorials/js_assets/`. As a final fallback, point at
    #     the consolidated `tutorials/others/images/` dir (where modules like
    #     `photo` store their assets).
    def _img_repl(m: re.Match) -> str:
        rel = m.group("rel")
        asset_dir = m.group("dir")
        if docname:
            local = DOC_ROOT / pathlib.Path(docname).parent / asset_dir / rel
            if local.is_file():
                return m.group(0)
        hit = _IMAGE_INDEX.get(pathlib.Path(rel).name)
        if hit:
            return f'{m.group("pre")}/{hit})'
        return f'{m.group("pre")}/tutorials/others/images/{rel})'
    text = re.sub(
        r'(?P<pre>!\[[^\]]*\]\()(?:[\w-]+/)?(?P<dir>images|js_assets)/(?P<rel>[^)]+)\)',
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

    return text


def _source_read(app, docname, source):
    # Translate any tutorial doc — the root index plus everything under a
    # module we enabled in DOC_MODULES / JS_DOC_MODULES.
    if not (docname.startswith("tutorials/") or docname.startswith("js_tutorials/")):
        return
    source[0] = _translate(source[0], docname)


def setup(app):
    app.connect("source-read", _source_read)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
