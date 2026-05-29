"""Per-class "Examples" cross-reference system."""
from __future__ import annotations
import re, pathlib
from .state import *

_EXAMPLE_SOURCE_EXTENSIONS = {
    # Program file types only; headers excluded.
    ".cpp", ".cc", ".cxx", ".c",
    ".py", ".java", ".js", ".ts",
}
_EXAMPLE_LANGUAGE = {
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c": "c",
    ".py": "python", ".java": "java",
    ".js": "javascript", ".ts": "typescript",
}
_EXAMPLE_INCLUDE_SUBTREES = (
    "samples/cpp/tutorial_code/",
    "samples/python/",
    "samples/java/",
    "samples/dnn/",
    "samples/gpu/",
)


def _is_canonical_example(rel_path: str) -> bool:
    """True iff this repo-relative path is a canonical example."""
    if any(rel_path.startswith(p) for p in _EXAMPLE_INCLUDE_SUBTREES):
        return True
    if rel_path.startswith("samples/cpp/"):
        rest = rel_path[len("samples/cpp/"):]
        return "/" not in rest        # direct child only, not nested
    if re.match(r"opencv_contrib/modules/[^/]+/samples/", rel_path):
        return True
    return False


def _example_pagename(display_path: str) -> str:
    """`samples/cpp/pca.cpp` → `samples_cpp_pca_cpp` (Sphinx-safe basename)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", display_path).strip("_").lower()


_EXAMPLE_FILES: list[tuple[str, pathlib.Path, frozenset[str]]] = []
_EXAMPLE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Not scoped to CONTRIB_MODULES: examples surface for all modules.
_example_scan_roots: list[tuple[pathlib.Path, str]] = [
    (OPENCV_ROOT / "samples", "samples"),
]
if CONTRIB_ROOT.is_dir():
    for _module_dir in sorted(CONTRIB_ROOT.iterdir()):
        if not _module_dir.is_dir():
            continue
        _contrib_samples = _module_dir / "samples"
        if _contrib_samples.is_dir():
            _example_scan_roots.append((
                _contrib_samples,
                f"opencv_contrib/modules/{_module_dir.name}/samples",
            ))

for _root, _display_prefix in _example_scan_roots:
    if not _root.is_dir():
        continue
    for _f in _root.rglob("*"):
        if not _f.is_file() or _f.suffix.lower() not in _EXAMPLE_SOURCE_EXTENSIONS:
            continue
        _rel = _f.relative_to(_root).as_posix()
        _display = f"{_display_prefix}/{_rel}"
        if not _is_canonical_example(_display):
            continue
        try:
            _text = _f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        _EXAMPLE_FILES.append((
            _display,
            _f,
            frozenset(_EXAMPLE_TOKEN_RE.findall(_text)),
        ))
_EXAMPLE_FILES.sort(key=lambda t: t[0])

# Display path → source path; only referenced samples (avoids orphan pages).
_EXAMPLE_PAGES_NEEDED: dict[str, pathlib.Path] = {}


_TUTORIAL_LINK_RE = re.compile(
    r"\[([^\]]+)\]\(([^)]*?samples/[^)]+?\.(?:cpp|cc|cxx|c|py|java|js|ts))\)",
    re.IGNORECASE,
)
# Trailing connector phrase, stripped from end of description.
_TUTORIAL_LINK_TRAILER_RE = re.compile(
    r"\s*(?:"
    r"can be found at|can be found in|"
    r"are available (?:at|in)|is available (?:at|in)|"
    r"located at|located in|found at|found in|"
    r"see also|see|here|"
    r"is at|is in|at|in"
    r")\s*$",
    re.IGNORECASE,
)
_TUTORIAL_LEADING_DIRECTIVE_RE = re.compile(
    r"^@(?:note|see|sa|warning|attention|remark|brief|todo|deprecated)\s+",
    re.IGNORECASE,
)
_ANOTHER_LEAD_RE = re.compile(r"^Another\s+(\w)", re.IGNORECASE)


def _extract_link_lead_in(text: str, link_start: int) -> str:
    """Pull the cleaned-up prose right before a tutorial hyperlink."""
    LOOKBACK = 400
    before = text[max(0, link_start - LOOKBACK):link_start]
    cut = max(
        before.rfind("\n\n"),
        before.rfind(". "),
        before.rfind("! "),
        before.rfind("? "),
        before.rfind(":\n"),
    )
    if cut >= 0:
        before = before[cut + 2:]    # skip past the 2-char boundary
    before = _TUTORIAL_LEADING_DIRECTIVE_RE.sub("", before.lstrip())
    before = re.sub(r"\s+", " ", before).strip()
    before = _TUTORIAL_LINK_TRAILER_RE.sub("", before).strip()
    m = _ANOTHER_LEAD_RE.match(before)
    if m:
        article = "An " if m.group(1).lower() in "aeiou" else "A "
        before = article + before[m.start(1):]
    return before


def _scan_tutorial_sample_refs() -> dict[str, str]:
    """Walk every tutorial markdown for sample-file hyperlinks."""
    refs: dict[str, str] = {}
    tutorial_roots = [
        DOC_ROOT / "tutorials",
        DOC_ROOT / "js_tutorials",
        DOC_ROOT / "py_tutorials",
    ]
    for tut_root in tutorial_roots:
        if not tut_root.is_dir():
            continue
        for md in tut_root.rglob("*.markdown"):
            try:
                text = md.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in _TUTORIAL_LINK_RE.finditer(text):
                sample_m = re.search(
                    r"samples/.+\.(?:cpp|cc|cxx|c|py|java|js|ts)",
                    m.group(2), re.IGNORECASE,
                )
                if not sample_m:
                    continue
                sample = sample_m.group(0)
                brief = _extract_link_lead_in(text, m.start())
                if not brief:
                    continue
                if sample not in refs or len(brief) > len(refs[sample]):
                    refs[sample] = brief
    return refs


_TUTORIAL_SAMPLE_REFS: dict[str, str] = _scan_tutorial_sample_refs()


# ----------------------- @example declaration scanner -----------------------

# `*/` guard stops capture before the next comment block's @brief.
_DOXY_EXAMPLE_RE = re.compile(
    r"[@\\]example\s+(\S+)"              # declared path
    r"[^\n]*\n"
    r"(?P<desc>"
    r"(?:"
    r"(?!\s*\*/)"                        # not comment closer
    r"(?!\s*\*?\s*[@\\]\w+)"             # not a new directive
    r"[^\n]*\n"
    r")*"
    r")"
)

_DOXY_PERCENT_ESCAPE_RE = re.compile(r"%(\w+)")


def _resolve_example_path(decl_path: str, header_path: pathlib.Path) -> str | None:
    """Resolve a `@example <path>` to our repo-relative display path."""
    # Case 1: repo-relative.
    abs_main = OPENCV_ROOT / decl_path
    if abs_main.is_file():
        return abs_main.relative_to(OPENCV_ROOT).as_posix()

    # Case 2: module-relative — module root is parent of include/.
    module_root: pathlib.Path | None = None
    for p in header_path.parents:
        if p.name == "include":
            module_root = p.parent
            break
    if module_root is None:
        return None

    abs_mod = module_root / decl_path
    if not abs_mod.is_file():
        return None

    # is_relative_to not startswith: opencv prefixes opencv_contrib.
    if abs_mod.is_relative_to(OPENCV_ROOT):
        return abs_mod.relative_to(OPENCV_ROOT).as_posix()
    contrib_parent = CONTRIB_ROOT.parent.parent
    if abs_mod.is_relative_to(contrib_parent):
        return abs_mod.relative_to(contrib_parent).as_posix()
    return f"opencv_contrib/modules/{module_root.name}/{decl_path}"


def _scan_doxygen_example_decls() -> dict[str, tuple[str, str]]:
    """Walk every module header for `@example` declarations."""
    refs: dict[str, tuple[str, str]] = {}
    roots = [OPENCV_ROOT / "modules"]
    if CONTRIB_ROOT.is_dir():
        roots.append(CONTRIB_ROOT)

    for root in roots:
        if not root.is_dir():
            continue
        for ext in ("*.hpp", "*.h"):
            for header in root.rglob(ext):
                # Only <module>/include/.
                if "/include/" not in header.as_posix():
                    continue
                try:
                    text = header.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for m in _DOXY_EXAMPLE_RE.finditer(text):
                    declared = m.group(1).strip().rstrip("*/").strip()
                    resolved = _resolve_example_path(declared, header)
                    if not resolved:
                        continue
                    raw = m.group("desc") or ""
                    cleaned = re.sub(r"\n\s*\*\s?", " ", raw)
                    cleaned = cleaned.replace("*/", "").strip()
                    cleaned = re.sub(r"\s+", " ", cleaned)
                    cleaned = _DOXY_PERCENT_ESCAPE_RE.sub(r"\1", cleaned)
                    # Keep longest if declared in multiple headers.
                    existing = refs.get(resolved)
                    if existing is None or len(cleaned) > len(existing[1]):
                        refs[resolved] = (declared, cleaned)
    return refs


_DOXY_EXAMPLE_DECLS: dict[str, tuple[str, str]] = _scan_doxygen_example_decls()


def _find_examples_for_class(class_simple: str) -> list[tuple[str, str]]:
    """Canonical sample files mentioning the class name."""
    if not class_simple:
        return []
    candidates = {class_simple}
    if class_simple.startswith("_") and len(class_simple) > 1:
        candidates.add(class_simple[1:])    # _InputArray → InputArray alias
    out: list[tuple[str, str]] = []
    for display, source_path, tokens in _EXAMPLE_FILES:
        # Stage 1: must be declared with @example.
        decl = _DOXY_EXAMPLE_DECLS.get(display)
        if decl is None:
            continue
        # Stage 2: must mention the class.
        if any(c in tokens for c in candidates):
            _EXAMPLE_PAGES_NEEDED[display] = source_path
            declared, _desc = decl
            out.append((declared, _example_pagename(display)))
    return out


def _render_examples_block(examples: list[tuple[str, str]]) -> list[str]:
    """HTML lines for the "Examples" footer; empty list if no matches."""
    if not examples:
        return []
    import html as _html_pkg
    parts = [
        f'<a class="opencv-example-link" '
        f'href="../examples/{_html_pkg.escape(page, quote=True)}.html">'
        f'{_html_pkg.escape(display)}</a>'
        for display, page in examples
    ]
    if len(parts) == 1:
        joined = parts[0]
    else:
        joined = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return [
        '<dl class="opencv-examples">',
        '<dt>Examples</dt>',
        f'<dd>{joined}.</dd>',
        '</dl>',
        "",
    ]

# Boilerplate-paragraph filter for _extract_sample_brief.
_SAMPLE_BRIEF_SKIP_RE = re.compile(
    r"^(?:"
    r"author|date|file|copyright|special\s+thanks|see\s+also|maintainer|"
    r"created|modified|version|license|brief"
    r")\b"
    r"|^\w+\.(?:cpp|cc|cxx|c|hpp|h|py|java|js|ts)\s*$",
    re.IGNORECASE,
)


def _extract_sample_brief(text: str) -> str:
    """Best-effort one-line description of a sample file."""
    # 1) Explicit @brief / \brief — preferred when present.
    m = re.search(
        r"[@\\]brief\s+(.*?)(?:"
        r"\n\s*[*/]?\s*\n"          # paragraph break
        r"|\n\s*[*/]?\s*[@\\]\w+"   # next Doxygen tag
        r"|\*/"                     # end of comment
        r")",
        text, re.DOTALL,
    )
    if m:
        brief = re.sub(r"\n\s*\*?\s*", " ", m.group(1))
        brief = re.sub(r"\s+", " ", brief).strip()
        brief = re.split(r"(?<=[.!?])\s", brief, maxsplit=1)[0].strip()
        if brief:
            return brief

    # 2) First /* ... */ block, split into paragraphs.
    cm = re.match(r"\s*/\*+([\s\S]*?)\*+/", text)
    if not cm:
        return ""
    # .strip() not .rstrip(): leading space would bypass skip regex.
    norm_lines = [
        re.sub(r"^\s*\*+\s?", "", raw).strip()
        for raw in cm.group(1).splitlines()
    ]
    paragraphs: list[list[str]] = [[]]
    for line in norm_lines:
        if line.strip():
            paragraphs[-1].append(line)
        elif paragraphs[-1]:
            paragraphs.append([])
    paragraphs = [p for p in paragraphs if p]

    for para in paragraphs:
        if _SAMPLE_BRIEF_SKIP_RE.match(para[0]):
            continue
        joined = " ".join(line.strip() for line in para if line.strip())
        first = re.split(r"(?<=[.!?])\s", joined, maxsplit=1)[0].strip()
        if first:
            return first
    return ""


# Doxygen markup inside `@example` description text: `@ref` cross-references
# and Markdown image embeds. The generated page skips the `_translate`
# source-read pipeline, so resolve these here via translate's shared indexes.
_BRIEF_REF_RE = re.compile(r'@ref\s+(?P<anchor>[\w:-]+)(?:\s+"(?P<label>[^"]+)")?')
_BRIEF_IMG_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)')


def _resolve_brief_markup(brief: str) -> str:
    """Resolve `@ref` links and bare-filename images in a brief to Markdown."""
    def _ref(m: "re.Match") -> str:
        anchor = _resolve_redirect(m.group("anchor"))
        label = m.group("label")
        target = _ANCHOR_TO_DOC.get(anchor)
        if target:
            return f'[{label or _ANCHOR_TO_TITLE.get(anchor) or anchor}](/{target})'
        if anchor in _TAG_FILENAMES:
            return f'[{label or _TAG_TITLES.get(anchor, anchor)}]({_doxygen_url(anchor)})'
        return f'[{label or anchor}](#{anchor})'
    brief = _BRIEF_REF_RE.sub(_ref, brief)

    def _img(m: "re.Match") -> str:
        alt, src = m.group("alt"), m.group("src")
        # Leave already-pathed / absolute / URL images for MyST to handle.
        if "/" in src or "://" in src:
            return m.group(0)
        hit = _IMAGE_INDEX.get(src)
        if not hit:
            return m.group(0)
        # Hard break after image so following prose drops to a new line.
        return f'![{alt}](/{hit})\\\n'
    return _BRIEF_IMG_RE.sub(_img, brief).strip()


def _generate_example_pages(examples_dir: pathlib.Path) -> None:
    """Write one Sphinx page per sample referenced by an Examples block.

    Uses MyST colon-fence `:::` not backticks: source backticks can
    close a backtick fence prematurely.
    """
    if not _EXAMPLE_PAGES_NEEDED:
        return
    examples_dir.mkdir(parents=True, exist_ok=True)
    for display, source in _EXAMPLE_PAGES_NEEDED.items():
        try:
            body = source.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        language = _EXAMPLE_LANGUAGE.get(source.suffix.lower(), "text")
        # Brief cascade: @example desc, tutorial prose, top comment.
        decl = _DOXY_EXAMPLE_DECLS.get(display)
        declared_path = decl[0] if decl else display
        decl_desc = decl[1] if decl else ""
        brief = (
            decl_desc
            or _TUTORIAL_SAMPLE_REFS.get(display)
            or _extract_sample_brief(body)
        )
        lines = [
            "---",
            "orphan: true",
            "---",
            f"# {declared_path}",
            "",
        ]
        if brief:
            # MyST paragraph (not raw <p>) so embedded @ref / image renders.
            lines.append("{.opencv-example-brief}")
            lines.append(_resolve_brief_markup(brief))
            lines.append("")
        lines.extend([
            f":::{{code-block}} {language}",
            ":linenos:",
            "",
        ])
        lines.extend(body.splitlines())
        lines.append(":::")
        lines.append("")
        (examples_dir / f"{_example_pagename(display)}.md").write_text(
            "\n".join(lines), encoding="utf-8")



__all__ = [
    "_find_examples_for_class",
    "_render_examples_block",
    "_generate_example_pages",
]
