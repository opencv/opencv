# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""Doxygen-flavored .markdown -> MyST translation (the source-read engine)."""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *


def _normalize_lang(lang: str) -> str:
    lang = (lang or "").strip(".").strip().lower() or "text"
    return _LANG_ALIASES.get(lang, lang)


# Snippet-boundary marker lines (e.g. `// [name]`, `## [name]`), stripped
# from rendered code to mirror docs.opencv.org. Matches only a bare
# marker + `[name]` on its own line, so real code embedding a `[token]`
# is left alone.
_SNIPPET_MARKER_RE = re.compile(
    r"^[ \t]*(?://!|//|##|#)[ \t]*\[[\w\- ]+\][ \t]*$"
    r"|^[ \t]*<!--[ \t]*\[[\w\- ]+\][ \t]*-->[ \t]*$"
)

# Doxygen block comments (`/** … */`, `/*! … */`) are decorative chrome in
# OpenCV samples; strip them to mirror docs.opencv.org. Plain `/* … */`,
# `//`, and `///` comments are kept as visible commentary.
# Line-block pass first removes whole-line blocks (consuming the trailing
# newline so no hollow blank remains); inline pass then sweeps the rare
# mid-line block without eating surrounding code.
_DOXY_LINE_BLOCK_RE = re.compile(
    r"^[ \t]*/\*[*!].*?\*/[ \t]*\n?",
    re.DOTALL | re.MULTILINE,
)
_DOXY_INLINE_BLOCK_RE = re.compile(
    r"/\*[*!].*?\*/",
    re.DOTALL,
)


def _strip_snippet_markers(text: str) -> str:
    """Drop snippet-boundary marker lines from a code body (whole line
    removed, no hollow blank left). Runs of 3+ resulting blank lines are
    collapsed to 2, matching docs.opencv.org's rendering."""
    out = [ln for ln in text.split("\n")
           if not _SNIPPET_MARKER_RE.match(ln)]
    joined = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", joined)


def _strip_doxygen_block_comments(text: str) -> str:
    """Drop every `/** … */` / `/*! … */` Doxygen block comment anywhere in
    a snippet (with or without `@tag` directives). Plain `/* … */`, `//`,
    and `///` comments stay intact."""
    text = _DOXY_LINE_BLOCK_RE.sub("", text)
    text = _DOXY_INLINE_BLOCK_RE.sub("", text)
    return text


def _read_snippet(rel_path: str, label: str | None) -> tuple[str, str]:
    """Return (code_text, language) for an @include / @snippet directive."""
    rel_norm = rel_path.lstrip("/")
    p = next((b / rel_norm for b in _SNIPPET_BASES
              if (b / rel_norm).is_file()), None)
    # Doxygen EXAMPLE_RECURSIVE basename lookup.
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
        # `@include`: whole file. Strip Doxygen block comments and snippet
        # markers so the rendered block matches docs.opencv.org (starts at
        # the first `#include`/`import`, no banners or doc-headers).
        return _strip_snippet_markers(
            _strip_doxygen_block_comments(text)), lang
    # Match `[label]` after any comment marker.
    pat = re.compile(r"^[^\[\n]*(?://!|//|##|#|<!--)[^\[\n]*\[" + re.escape(label)
                     + r"\][^\n]*$", re.MULTILINE)
    matches = list(pat.finditer(text))
    if len(matches) < 2:
        return f"// snippet not found: {rel_path} [{label}]\n", lang
    body = text[matches[0].end():matches[1].start()].strip("\n")
    # Strip nested markers and mid-snippet Doxygen blocks: samples often
    # place a `/** @function bar */` doc-header inside a labelled range.
    body = _strip_doxygen_block_comments(body)
    body = _strip_snippet_markers(body)
    lines = body.split("\n")
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    if indents:
        dedent = min(indents)
        lines = [l[dedent:] if len(l) >= dedent else l for l in lines]
    return "\n".join(lines), lang


# Patterns used by `_dedent_dash_hash_indent` to skip code regions.
_DEDENT_AT_CODE_OPEN_RE = re.compile(r"^[ \t]*@code(?:\{[^}]*\})?\s*$")
_DEDENT_AT_CODE_CLOSE_RE = re.compile(r"^[ \t]*@endcode\s*$")
_DEDENT_FENCE_OPEN_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_DEDENT_FENCE_CLOSE_RE = re.compile(r"^[ \t]*([`~]{3,})[ \t]*$")
# Both ordered-list flavours OpenCV sources use — `-#` (Doxygen) and
# `\d+\.` (CommonMark) — indent bodies +4 spaces/level, so both trigger
# the same dedent.
_DEDENT_DASH_HASH_RE = re.compile(r"^[ \t]*(?:-#|\d+\.)[ \t]")
# Directives that steps 4/5 turn into fenced code; their stray-indent
# handling matches `@code` (see step 0a).
_SNIPPET_DIRECTIVE_RE = re.compile(r"^[ \t]*@(?:snippet|include(?:lineno)?)\b")


def _map_list_indent(n: int) -> int:
    """Map 4-space-per-level list indent to 3-space-per-level, preserving
    the 0-3 space remainder within a level. Shared by
    `_dedent_dash_hash_indent` and `_fenced` so both land on the same
    dedented baseline."""
    return (n // 4) * 3 + (n % 4)


def _dedent_keep_blanks(s: str) -> str:
    """Strip the common leading-whitespace prefix from `s`, leaving
    whitespace-only lines untouched. Unlike `textwrap.dedent`, it does NOT
    normalise interior blank lines to empty — preserving them keeps snippet
    bodies byte-stable through the re-indent/re-base round-trip."""
    widths = [len(l) - len(l.lstrip(" \t")) for l in s.split("\n") if l.strip()]
    n = min(widths) if widths else 0
    return "\n".join(l[n:] if l.strip() else l for l in s.split("\n"))


def _fenced(base: str, lang: str, body: str) -> str:
    """Emit a fenced code block at `base` indentation.

    Inside a list item the fence must sit at the item's content baseline
    (relative-0): a column-0 fence terminates the list, and a relative +4
    fence renders as an indented code block showing literal ``` backticks.
    Only relative-0 keeps the fence recognised AND the list open.
    `base=""` => column 0 (top-level, unchanged)."""
    body = _dedent_keep_blanks(body).strip("\n")
    if base:
        body = "\n".join(base + ln if ln.strip() else ln
                         for ln in body.split("\n"))
    return f"\n{base}```{lang}\n{body}\n{base}```\n"


_LIST_MARKER_RE = re.compile(
    r"^(?P<ind> *)(?P<mk>-#|\d+\.|[-*+])(?P<gap> +)(?P<rest>.*)$")


def _dedent_dash_hash_indent(src: str) -> str:
    """Re-indent every list (ordered `-#`/`\\d+\\.` and bullet `-`/`*`/`+`)
    to a canonical shape so deep nesting stays out of CommonMark's 4-space
    indented-code trap.

    OpenCV tutorials indent list bodies 4 spaces/level — exactly where
    CommonMark/MyST reads a line as an indented code block, wrecking fences,
    continuation prose/nested bullets, and inline images inside list bodies.
    The fix normalises the gap after each marker to one space and nests each
    level under its parent's content column, then re-bases body lines to
    that column (keeping hanging indent). Normalising marker *width* (not
    just scaling indent) keeps fixed-width bullets like `-   ` aligned with
    the fence beneath them, which the old proportional 4->3 scaling broke.

    `@code…@endcode` and pre-existing fenced bodies pass through with native
    indentation; only their marker lines and ordinary list content are
    re-based. Lines outside any list are untouched."""
    lines = src.split("\n")
    out: list[str] = []
    # Stack of open list ancestors: original marker/content columns (to test
    # membership) plus the new content column each was re-based to.
    stack: list[tuple[int, int, int]] = []   # (orig_col, orig_cont, new_cont)
    in_at_code = False
    in_fence = False
    fence_char = ""

    def rebase_content(line: str, code_marker: bool = False) -> str:
        # Re-base a non-marker line to its governing marker's new content
        # column, keeping hanging indent past it. Mutates `stack`.
        ind = len(line) - len(line.lstrip(" "))
        while stack and stack[-1][0] >= ind:
            stack.pop()
        if not stack:
            # Outside any list: keep intended indented content as-is, but
            # strip stray indent off a code directive/fence marker so a
            # 4+-space `@code`/`@snippet` loosely attached to a top-level
            # `@note` (uncaptured by the note regex past its blank line)
            # doesn't become a root-level indented-code fence.
            return line.lstrip(" ") if code_marker else line
        _oc, ocont, ncont = stack[-1]
        # Clamp code directives/fences to the item's content baseline
        # (relative-0): extra source offset would land the fence at +N, and
        # at +4 MyST shows literal backticks. Prose keeps its hanging indent.
        new = ncont if code_marker else max(0, ncont + (ind - ocont))
        return " " * new + line[ind:]

    for line in lines:
        # @code…@endcode body passes through unchanged; only the marker
        # lines are re-based, so `_code_repl` later emits the fence at the
        # list-content column (see `_fenced`).
        if in_at_code:
            if _DEDENT_AT_CODE_CLOSE_RE.match(line):
                in_at_code = False
                out.append(rebase_content(line, code_marker=True))
            else:
                out.append(line)
            continue
        if in_fence:
            out.append(line)
            cm = _DEDENT_FENCE_CLOSE_RE.match(line)
            if cm and cm.group(1)[0] == fence_char:
                in_fence = False
            continue
        if _DEDENT_AT_CODE_OPEN_RE.match(line):
            in_at_code = True
            out.append(rebase_content(line, code_marker=True))
            continue
        fm = _DEDENT_FENCE_OPEN_RE.match(line)
        if fm:
            fence_char = fm.group(1)[0]
            in_fence = True
            # Leave a pre-existing ``` fence as authored: only its opener is
            # seen here (body/closer skipped via `in_fence`), so re-basing
            # the opener alone would misalign the block. (`@code`/`@snippet`
            # instead go through `_fenced`, re-indenting the whole block.)
            out.append(line)
            continue

        if not line.strip():
            out.append(line)            # blank: keep, list stays open
            continue

        mm = _LIST_MARKER_RE.match(line)
        if mm:
            oc = len(mm.group("ind"))
            mk = mm.group("mk")
            ocont = oc + len(mk) + len(mm.group("gap"))
            # Pop siblings / deeper levels this marker closes.
            while stack and stack[-1][0] >= oc:
                stack.pop()
            # Nested marker sits at its parent's content column; a top-level
            # marker is dedented 4->3-per-level so a merely-indented list
            # (e.g. `    1. step`) starts at 0-3 spaces and is recognised,
            # not read as an indented code block at >=4 spaces.
            new_mk = stack[-1][2] if stack else _map_list_indent(oc)
            new_cont = new_mk + len(mk) + 1
            out.append(" " * new_mk + mk + " " + mm.group("rest"))
            stack.append((oc, ocont, new_cont))
            continue

        # `@snippet`/`@include` emit fenced code later, so treat them as code
        # markers — a stray-indented one must not keep a 4+-space indent that
        # would render as a root-level indented fence.
        out.append(rebase_content(
            line, code_marker=bool(_SNIPPET_DIRECTIVE_RE.match(line))))

    return "\n".join(out)


def _emit_toggles(tabs: list[tuple[str, str]]) -> str:
    # Re-base tab bodies to column 0: the `tab-item` container is emitted at
    # column 0, so a fence arriving baseline-indented (from a snippet inside
    # an indented `@add_toggle`) would sit at relative +N and render as
    # literal backticks. No-op for the pre-existing column-0 snippet form.
    tabs = [(lang, _dedent_keep_blanks(body)) for lang, body in tabs]
    if HAVE_SPHINX_DESIGN:
        out = ["", "``````{tab-set}"]
        for lang, body in tabs:
            label = _TOGGLE_LABELS.get(lang, lang.title())
            out += [f"`````{{tab-item}} {label}", body, "`````"]
        out += ["``````", ""]
        return "\n".join(out)
    # Fallback without sphinx-design.
    out = [""]
    for lang, body in tabs:
        label = _TOGGLE_LABELS.get(lang, lang.title())
        out += [f"**{label}**", "", body, ""]
    return "\n".join(out)

def _translate(text: str, docname: str | None = None) -> str:
    # 0v. @verbatim — stash body; restored at end.
    _verbatim_stash: dict[str, str] = {}
    def _verbatim_save(body: str, inline: bool) -> str:
        key = f"VERBATIM_{len(_verbatim_stash)}"
        if inline:
            _verbatim_stash[key] = f"`{body.strip()}`"
        else:
            _verbatim_stash[key] = f"\n```text\n{body.strip()}\n```\n"
        return key
    # Block form — run first.
    text = re.sub(
        r"@verbatim[ \t]*\n(?P<body>.*?)\n[ \t]*@endverbatim",
        lambda m: _verbatim_save(m.group("body"), inline=False),
        text, flags=re.DOTALL)
    # Inline form.
    text = re.sub(
        r"@verbatim[ \t]+(?P<body>[^\n]+?)[ \t]+@endverbatim",
        lambda m: _verbatim_save(m.group("body"), inline=True),
        text)

    if docname == "tutorials/tutorials" and not USE_INDEX_LANDING:
        # Lead sidebar with intro; faq/bibliography stay appended.
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

    # 0a. py_tutorials root: promote @ref to @subpage (case-by-case avoids cycles).
    if docname == "py_tutorials/py_tutorials":
        if "py_video" in PY_DOC_MODULES:
            text = re.sub(
                r"@ref\s+tutorial_table_of_content_video\b",
                "@subpage tutorial_py_table_of_contents_video",
                text,
            )
        # Object Detection stays @ref (its C++ page is already in main toctree).

    # 0d. py_video/py_objdetect stub trees -> :orphan: (real link via 0a).
    if docname and (docname.startswith("py_tutorials/py_video/")
                    or docname.startswith("py_tutorials/py_objdetect/")):
        text = "---\norphan: true\n---\n\n" + text

    # 0a-dedent. Re-indent lists below CommonMark's 4-space indented-code
    # threshold — see `_dedent_dash_hash_indent` for the rationale.
    text = _dedent_dash_hash_indent(text)

    # 0a2. Prose wedged between an `@endcode` and the next `@code` (no list,
    # no blank line) is text, but its 4-space source indent makes CommonMark
    # render it as a stray code block. Re-align it to the directives' column.
    # Anchored on `@endcode … @code`, so inert on raw-indented-code pages.
    def _dedent_interblock(m: re.Match) -> str:
        base = m.group("ind")
        mid = "\n".join((base + l.lstrip(" ")) if l.strip() else l
                        for l in m.group("mid").split("\n"))
        return m.group("pre") + mid
    text = re.sub(
        r"(?m)^(?P<pre>(?P<ind>[ \t]*)@endcode[ \t]*\n)"
        r"(?P<mid>(?:[ \t]+\S[^\n]*\n)+?)"
        r"(?=(?P=ind)@code(?:\{[^}]*\})?[ \t]*$)",
        _dedent_interblock, text)

    # 0b. "-# foo" -> "1. foo".
    text = re.sub(r"^(?P<indent>[ \t]*)-#[ \t]+",
                  lambda m: f"{m.group('indent')}1. ", text, flags=re.MULTILINE)

    # 0c. Dedent orphan indented bullets under a paragraph.
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

    # 1. Heading anchors "Title {#name}" -> MyST label + ATX.
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

    # 1b. Trailing setext heading at EOF -> ATX (docutils rejects trailing transition).
    text = re.sub(
        r"^(?P<title>[^\n#=\-][^\n]*?)[ \t]*\n(?P<bar>[=\-])[=\-]{2,}[ \t]*$\s*\Z",
        lambda m: f"{'#' if m.group('bar') == '=' else '##'} {m.group('title').strip()}\n",
        text, flags=re.MULTILINE)

    # 1c. Remaining mid-doc setext H1s -> ATX (so 1d sees them).
    text = re.sub(
        r"^(?P<title>[^\n#=\-][^\n]*?)[ \t]*\n=[=]{2,}[ \t]*$",
        lambda m: f"# {m.group('title').strip()}",
        text, flags=re.MULTILINE)

    # 1c2. Setext H2s starting with digit+dot ("1. Moments\n---") -> ATX ##.
    text = re.sub(
        r"^(?P<title>\d+\.[^\n]+?)[ \t]*\n-[-]{2,}[ \t]*$",
        lambda m: f"## {m.group('title').strip()}",
        text, flags=re.MULTILINE)

    # 1d. Demote every H1 after the first to H2.
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
                    line = '#' + line
            out.append(line)
        return '\n'.join(out)
    text = _demote_extra_h1s(text)

    _ADMON_KIND = {"note": "note", "see": "seealso", "warning": "warning",
                   "sa": "seealso"}
    def _admon_repl(m: re.Match) -> str:
        kind = _ADMON_KIND[m.group("dir")]
        raw = m.group("body")
        # `head` is the matched `@note ` prefix before the body; it ends in a
        # newline only in the next-line form, so a non-newline end means the
        # body rode on the same line as the directive.
        head = m.group(0)[: len(m.group(0)) - len(raw)]
        same_line = not head.endswith("\n")
        # Directive's leading indent (the list-content baseline after step
        # 0a). Re-indent the whole admonition to it so an indented note stays
        # inside its list item instead of breaking the list at column 0.
        indent = head[: len(head) - len(head.lstrip(" \t"))]
        lines = raw.split("\n")
        # Common body indent to strip. Exclude the same-line first line (which
        # has zero leading indent) from the min, else min_ind collapses to 0
        # and continuation lines keep their indent, rendering as a spurious
        # code block inside the note box.
        ind_src = lines[1:] if same_line else lines
        cand = [len(l) - len(l.lstrip()) for l in ind_src if l.strip()]
        min_ind = min(cand) if cand else 0
        def _dedent(l: str) -> str:
            # Strip up to min_ind leading-whitespace chars only — never slice
            # into a less-indented line's text (e.g. the same-line first line).
            i = 0
            while i < min_ind and i < len(l) and l[i] in " \t":
                i += 1
            return l[i:]
        body = "\n".join(_dedent(l) for l in lines).strip()
        block = f":::{{{kind}}}\n{body}\n:::"
        if indent:
            block = "\n".join((indent + ln) if ln.strip() else ln
                              for ln in block.split("\n"))
        return f"\n{block}\n"
    _ac_stash: dict[str, str] = {}
    def _ac_hide(m: re.Match) -> str:
        k = f"\x00AC{len(_ac_stash)}\x00"; _ac_stash[k] = m.group(0); return k
    _t = re.sub(r"@code(?:\{[^}]*\})?.*?@endcode", _ac_hide, text, flags=re.DOTALL)
    _t = re.sub(
        r"^[ \t]*@(?P<dir>note|see|warning|sa)[ \t]*\n?(?P<body>.+?)"
        r"(?=\n[ \t]*\n|\n[ \t]*@[A-Za-z]|\Z)",
        _admon_repl, _t, flags=re.DOTALL | re.MULTILINE)
    for _k, _v in _ac_stash.items():
        _t = _t.replace(_k, _v)
    text = _t

    # MyST definition list with "Parameters" heading.
    def _inline_block_math(s: str) -> str:
        if re.match(r"^\\f\[.+\\f\]$", s.strip()):
            return s
        return re.sub(r"\\f\[(.+?)\\f\]", lambda mm: f"${mm.group(1).strip()}$", s)
    def _param_block_repl(m: re.Match) -> str:
        result, has_param, cur_name, cur_desc = [], False, None, []
        for line in m.group(0).split("\n"):
            pm = re.match(r"@param\s+(\S+)\s+(.*)", line.strip())
            rm = re.match(r"@return\s+(.*)", line.strip())
            if pm:
                if cur_name:
                    result.append(f"`{cur_name}`\n: {_inline_block_math(' '.join(cur_desc))}")
                cur_name, cur_desc, has_param = pm.group(1), [pm.group(2).strip()], True
            elif rm:
                if cur_name:
                    result.append(f"`{cur_name}`\n: {_inline_block_math(' '.join(cur_desc))}")
                    cur_name = None
                result.append(f"*(return value)*\n: {_inline_block_math(rm.group(1).strip())}")
            elif cur_name and line.strip():
                cur_desc.append(line.strip())
        if cur_name:
            result.append(f"`{cur_name}`\n: {_inline_block_math(' '.join(cur_desc))}")
        header = "\n**Parameters**\n\n" if has_param else "\n"
        return header + "\n\n".join(result) + "\n"
    text = re.sub(
        r"((?:^@(?:param\s+\S+|return)\s+[^\n]+\n(?:[ \t]+[^\n]+\n)*)+)",
        _param_block_repl, text, flags=re.MULTILINE)

    # 2. Doxygen LaTeX math markers; preserve indent so blocks inside list items stay in the list.
    def _split_adj_math(m: re.Match) -> str:
        indent = m.group("indent")
        return m.group(0).replace("\\f]\\f[", f"\\f]\n{indent}\\f[")
    text = re.sub(r"^(?P<indent>[ \t]*)[^\n]*\\f\]\\f\[",
                  _split_adj_math, text, flags=re.MULTILINE)
    def _fblock(m: re.Match) -> str:
        ind = m.group("indent")
        # Re-base the math body to fence baseline `ind` (relative-0 of the
        # list item, matching `_fenced`); at the deeper source offset it lands
        # at relative +4 and block recognition breaks.
        body = _dedent_keep_blanks(m.group("body")).strip("\n").strip()
        reindent = lambda b: "\n".join((ind + ln) if ln.strip() else ln
                                       for ln in b.split("\n"))
        if "\\\\" in body:
            body = re.sub(r"\n\s*\n", "\n", body)
            return f"\n{ind}```{{math}}\n{reindent(body)}\n{ind}```\n"
        return f"\n{ind}$$\n{reindent(body)}\n{ind}$$\n"
    text = re.sub(r"^(?P<indent>[ \t]*)\\f\[(?P<body>.+?)\\f\]",
                  _fblock, text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"\\f\[(.+?)\\f\]",
                  lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
                  text, flags=re.DOTALL)
    text = re.sub(r"\\f\$(.+?)\\f\$", lambda m: f"${m.group(1)}$",
                  text, flags=re.DOTALL)

    # 2b. \bordermatrix is Plain-TeX; convert to a `matrix` env, `\cr` -> `\\`.
    text = re.sub(
        r"\\bordermatrix\s*\{([^}]*)\}",
        lambda m: r"\begin{matrix}" + m.group(1).replace(r"\cr", r"\\")
                  + r"\end{matrix}",
        text)

    # 3. @code{.lang} ... @endcode -> fenced block at the list-content
    # baseline (see `_fenced`). Step 0a already dedented the @code marker
    # lines, so the captured `indent` is the right column.
    def _code_repl(m: re.Match) -> str:
        lang = _normalize_lang(m.group("lang") or "")
        return _fenced(m.group("indent") or "", lang, m.group("body"))
    text = re.sub(
        r"^(?P<indent>[ \t]*)@code(?:\{(?P<lang>[^}]*)\})?\s*\n(?P<body>.*?)\n[ \t]*@endcode",
        _code_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 3b. Tilde fences `~~~~{.lang}` -> backtick fences (MyST reads `{.lang}` as directive).
    def _tilde_repl(m: re.Match) -> str:
        lang = (m.group("lang") or "").strip().lstrip(".") or "text"
        if lang == "none":
            lang = "text"
        return f"\n```{lang}\n{m.group('body').strip()}\n```\n"
    text = re.sub(
        r"^~~~+(?:[ \t]*\{(?P<lang>[^}\n]+)\})?[ \t]*\n"
        r"(?P<body>.*?)\n^~~~+[ \t]*$",
        _tilde_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 3c. Fence lexer `yml` -> `yaml` (Pygments warns on 'yml'); runs after 3/3b.
    text = re.sub(
        r"^(?P<fence>`{3,}|~{3,})yml\b(?P<rest>[ \t]*)$",
        lambda m: f"{m.group('fence')}yaml{m.group('rest')}",
        text, flags=re.MULTILINE)

    # 3d. \htmlonly ... \endhtmlonly -> `{raw} html`.
    _depth = len(docname.split("/")) - 1 if docname else 1
    _to_root = "../" * _depth
    def _htmlonly_repl(m: re.Match) -> str:
        body = re.sub(r'src="(?:\.\./)+(?P<f>js_[^"]+)"',
                      lambda mm: f'src="{_to_root}js_tutorials/{mm.group("f")}"',
                      m.group("body"))
        body = re.sub(r'\s*onload="[^"]*"', ' height="700px"', body)
        return f"\n```{{raw}} html\n{body.strip()}\n```\n"
    text = re.sub(
        r"\\htmlonly\s*\n(?P<body>.*?)\n\s*\\endhtmlonly",
        _htmlonly_repl, text, flags=re.DOTALL)

    # 3e. Plain fences with Doxygen lang spec ("```.sh") -> strip dot, alias-map.
    text = re.sub(
        r"^(?P<fence>`{3,})(?P<lang>\.?[\w-]+)[ \t]*$",
        lambda m: f"{m.group('fence')}{_normalize_lang(m.group('lang'))}",
        text, flags=re.MULTILINE)

    # Backtick fence at the list-content baseline (see `_fenced`). Step 0a
    # already dedented the `@include`/`@snippet` line, so `indent` is the
    # baseline. A top-level snippet has indent "" (column 0); one inside an
    # `@add_toggle` is re-based to column 0 by `_emit_toggles`.
    def _emit_codeblock(indent: str, lang: str, body: str) -> str:
        return _fenced(indent, lang, body)

    # 4. @include path / @includelineno path.
    def _include_repl(m: re.Match) -> str:
        code, lang = _read_snippet(m.group("path"), None)
        return _emit_codeblock(m.group("indent") or "", lang, code)
    text = re.sub(r"^(?P<indent>[ \t]*)@include(?:lineno)?\s+(?P<path>\S+)",
                  _include_repl, text, flags=re.MULTILINE)

    # 4b. Drop a stray @snippet right after @end_toggle (duplicate of tab-set).
    text = re.sub(
        r"(^([ \t]*)@end_toggle[ \t]*\n)\2@snippet[^\n]*\n",
        r"\1",
        text, flags=re.MULTILINE)

    # 4b. @htmlinclude path -> raw HTML embed; lift just <body> for full docs.
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
        # Allow quoted attrs so a `=>` inside onload="..." doesn't end the tag.
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

    # 6. @add_toggle_LANG ... @end_toggle -> one tab-set
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
                # A repeated language starts a NEW tab-set, not a duplicate
                # tab: `cpp/python/cpp` must render as [C++|Python] then [C++].
                # Distinct languages (cpp/java/python) still merge into one.
                _nxt = re.match(r"@add_toggle_(\w+)", src[j + k.end():])
                if _nxt and _nxt.group(1) in {t[0] for t in tabs}:
                    # Rewind to the toggle's line start so the outer loop's
                    # `^`-anchored opener re-matches it: the inner
                    # `@end_toggle\s*\n?` may have eaten the next line's
                    # leading indent, leaving the toggle stranded mid-line.
                    j = src.rfind("\n", 0, j + k.end()) + 1
                    break
                j += k.end()
            if not tabs:
                out.append(src[m.start():m.start() + 1]); i = m.start() + 1; continue
            out.append(_emit_toggles(tabs))
            i = j
        return "".join(out)
    text = _toggle_collapse(text)

    def _link_repl(m: re.Match) -> str:
        target = m.group("target")
        disp = (m.group("disp") or "").strip().replace('"', '')
        return f'@ref {target} "{disp}"' if disp else f"@ref {target}"
    text = re.sub(
        r"@link\s+(?P<target>[\w-]+)(?P<disp>.*?)@endlink",
        _link_repl, text, flags=re.DOTALL)

    # API stub rewrites — extended from `api/core_basic` to every `api/`
    # page so the Functions/Typedefs detail blocks on `core_array`,
    # `core_utils`, `core_cluster`, … get the same per-token token
    # linkifier (step 8g) that turns `InputArray`, `OutputArray`,
    # `Mat`, `_Tp`, etc. inside the signature codespans into individual
    # `<a>` anchors. The pre-existing regex-driven steps (8a, 8b, 8e,
    # 8i, 8j and the Vec-rows / cv::Ptr rewrites) are no-ops on pages
    # whose markdown doesn't match their patterns, so widening the gate
    # is safe.
    if docname and (docname.startswith("api/")
                    or docname.startswith("main_modules/")
                    or docname.startswith("extra_modules/")):
        # Idempotency guard: the stub generator now emits the
        # "Shorter aliases for the most popular specializations of
        # Vec<T,n>" section itself (via the `@name` named-group path
        # in `_summary_block`), so the source MD already contains
        # both the heading and the Vec rows. The legacy rewrite
        # below also extracts the rows and injects a fresh section
        # before "## Typedef Documentation" — without this guard the
        # heading renders TWICE, the first instance an empty table.
        # Skip the rewrite when the source MD already has the
        # heading.
        if "## Shorter aliases for the most popular specializations of Vec<T,n>" not in text:
            _vec_rows_re = re.compile(
                r"(?:^\| `Vec<[^`]*` \| [^\n]*\n)+", re.MULTILINE)
            _vm = _vec_rows_re.search(text)
            if _vm:
                _vec_rows = _vm.group(0)
                text = text[:_vm.start()] + text[_vm.end():]
                _shorter = (
                    "## Shorter aliases for the most popular specializations of "
                    "Vec<T,n>\n\n"
                    # Carry the `.api-typedef-table` class so the section
                    # inherits the same table styling (and the light-mode
                    # blue Type-cell anchor rule) as the main Typedefs table
                    # above.
                    "{.api-typedef-table}\n"
                    "| Type | Name | Description |\n"
                    "|---|---|---|\n"
                    + _vec_rows + "\n")
                text = text.replace(
                    "## Typedef Documentation",
                    _shorter + "## Typedef Documentation",
                    1)

        # 8c. `{doxygentypedef} cv::Ptr` -> hand-rolled cpp:type (breathe skips C++11 aliases).
        text = re.sub(
            r"```\{doxygentypedef\} cv::Ptr\s*\n:project: opencv\s*\n```",
            "```{eval-rst}\n"
            ".. cpp:namespace:: cv\n"
            ".. cpp:type:: template<typename _Tp> Ptr = std::shared_ptr<_Tp>\n"
            "```",
            text)

        # 8e. Classes table rows: append template params + "View details" link.
        def _rewrite_class_row(m: re.Match) -> str:
            kind = m.group("kind")
            name = m.group("name")       # 'cv::dnn::BackendNode'
            page = m.group("page")       # 'classcv_1_1dnn_1_1BackendNode'
            desc = m.group("desc").strip()
            short = name.split("::")[-1]
            tparams = _CLASS_TEMPLATE_DISPLAY.get(short, "")
            link_label = f"{name}{tparams}"
            more = (f'<a class="opencv-class-more" '
                    f'href="{page}.html#detailed-description">View details</a>')
            desc_out = f"{desc} {more}" if desc else more
            return (f"| `{kind}` [`{link_label}`]({page}.md) "
                    f"| {desc_out} |")
        text = re.sub(
            r"\| \[`(?P<kind>class|struct) (?P<name>cv::[^`]+)`\]"
            r"\((?P<page>(?:class|struct)cv_1_1[A-Za-z0-9_]+)\.md\)"
            r" \| (?P<desc>[^\n|]*?) \|",
            _rewrite_class_row, text)


    if docname == "main_modules/core_basic":
        # 8c. `{doxygentypedef} cv::Ptr` -> hand-rolled cpp:type (breathe skips C++11 aliases).
        text = re.sub(
            r"```\{doxygentypedef\} cv::Ptr\s*\n:project: opencv\s*\n```",
            "```{eval-rst}\n"
            ".. cpp:namespace:: cv\n"
            ".. cpp:type:: template<typename _Tp> Ptr = std::shared_ptr<_Tp>\n"
            "```",
            text)

        # 8a. Name-column typedef anchors -> slugified detail-block id.
        text = re.sub(
            r"\[`(?P<name>[A-Za-z_][A-Za-z0-9_]*)`\]"
            r"\(#(?P<ref>group__[a-z0-9_]+?_1[a-z0-9]+)\)",
            lambda m: (f"[`{m.group('name')}`]"
                       f"(#{re.sub(r'_+', '-', m.group('ref'))})"),
            text)

        # 8i. Functions-table rows: split broken-anchor link into name link + params span.
        def _rewrite_function_row(m: re.Match) -> str:
            ret = m.group("ret").strip()
            name = m.group("name")
            params = m.group("params")
            desc = m.group("desc")
            slug = _func_slug(name)
            return (f"| {ret} | [`cv::{name}`](#{slug}) "
                    f"`({params})` | {desc} |")
        text = re.sub(
            r"\| (?P<ret>[^|\n]*?) \| "
            r"\[`(?P<name>[^(`\n]+?)\((?P<params>[^`\n]*)\)`\]"
            r"\(#group__[a-z0-9_]+?_1[a-z0-9]+\) \| "
            r"(?P<desc>[^\n|]*) \|",
            _rewrite_function_row, text)

        # 8j. Inject a raw HTML anchor before surviving `{doxygenfunction}` (dedup; first wins).
        _seen_anchors: set[str] = set()
        def _inject_func_anchor(m: re.Match) -> str:
            qname = m.group("qname").strip()       # "cv::log" / "cv::operator!="
            slug = _func_slug(qname.rsplit("::", 1)[-1])
            if slug in _seen_anchors:
                return m.group(0)
            _seen_anchors.add(slug)
            return f'<a id="{slug}"></a>\n\n{m.group(0)}'
        text = re.sub(
            r"^```\{doxygenfunction\} (?P<qname>[^\n(]+)\([^\n]*\n"
            r":project: opencv\n"
            r"```",
            _inject_func_anchor, text, flags=re.MULTILINE)

        # 8b. Type-column class code spans -> inline HTML link to sibling api page.
        if _LIVE_CLASS_URL:
            def _linkify_class_codespan(m: re.Match) -> str:
                cls = m.group("cls")
                rest = m.group("rest")
                full = _LIVE_CLASS_URL.get(cls)
                if not full:
                    return m.group(0)
                href = pathlib.PurePosixPath(full).name  # same module directory
                rest_esc = (rest.replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;"))
                return (f'<code class="docutils literal notranslate">'
                        f'<a class="reference internal" href="{href}">{cls}</a>'
                        f'{rest_esc}</code>')
            text = re.sub(
                r"`(?P<cls>[A-Z][A-Za-z0-9_]*)(?P<rest><[^`\n]*>)`",
                _linkify_class_codespan, text)

        # 8g. Linkify tokens step 8b missed: inner 8b `<code>`, then plain code spans.
        if _LOCAL_CLASS_URL or _LOCAL_TYPEDEF_URL:
            def _token_url(tok: str) -> str | None:
                # Tokens absent from the tagfile stay plain.
                return _LOCAL_CLASS_URL.get(tok) or _LOCAL_TYPEDEF_URL.get(tok)
            # Match an optional `cv::` prefix so the anchor spans `cv::Name`.
            _tok_re = re.compile(r"(?:cv::)?_?[A-Za-z][A-Za-z0-9_]*")
            def _bare(tok: str) -> str:
                return tok[4:] if tok.startswith("cv::") else tok
            def _anchor_text(tok: str) -> str:
                # Encode `::` so the later cv-linkifier doesn't nest a second anchor.
                return tok.replace("::", "&#58;&#58;")
            def _linkify_html_segment(seg: str) -> str:
                def _sub(m: re.Match) -> str:
                    url = _token_url(_bare(m.group(0)))
                    if not url:
                        return m.group(0)
                    return (f'<a class="reference internal" '
                            f'href="{url}">{_anchor_text(m.group(0))}</a>')
                return _tok_re.sub(_sub, seg)
            def _linkify_inside_code(m: re.Match) -> str:
                inner = m.group("inner")
                out, i, n = [], 0, len(m.group("inner"))
                while i < n:
                    if inner.startswith("<a ", i):
                        j = inner.find("</a>", i)
                        if j < 0:
                            out.append(inner[i:]); break
                        out.append(inner[i:j + 4]); i = j + 4
                    else:
                        k = inner.find("<a ", i)
                        if k < 0:
                            out.append(_linkify_html_segment(inner[i:])); break
                        out.append(_linkify_html_segment(inner[i:k])); i = k
                return m.group("open") + "".join(out) + m.group("close")
            text = re.sub(
                r'(?P<open><code class="docutils literal notranslate">)'
                r'(?P<inner>.*?)(?P<close></code>)',
                _linkify_inside_code, text, flags=re.DOTALL)

    # 8g. Linkify recognized type tokens in code spans across every API
    # page (Functions summary + Function Documentation detail blocks).
    # Was previously gated to `main_modules/core_basic` only — that
    # left parameter types in other module pages (calib, dnn, …) as
    # plain code chips even though their local typedef/class targets
    # exist. Pass 1 walks any existing `<code>` HTML emitted by
    # `_func_row_split_md` (function name already wrapped in `<a>`;
    # remaining text tokens get individual anchors). Pass 2 walks
    # remaining markdown code spans for any rows still using the
    # backticked-signature form.
    if (docname and (docname.startswith("api/")
                     or docname.startswith("main_modules/")
                     or docname.startswith("extra_modules/"))
            and (_LOCAL_CLASS_URL or _LOCAL_TYPEDEF_URL)):
        def _token_url(tok: str) -> str | None:
            # Tokens absent from the tagfile stay plain.
            return _LOCAL_CLASS_URL.get(tok) or _LOCAL_TYPEDEF_URL.get(tok)
        # Match an optional `cv::` prefix so the anchor spans `cv::Name`.
        _tok_re = re.compile(r"(?:cv::)?_?[A-Za-z][A-Za-z0-9_]*")
        def _bare(tok: str) -> str:
            return tok[4:] if tok.startswith("cv::") else tok
        def _anchor_text(tok: str) -> str:
            # Encode `::` so the later cv-linkifier doesn't nest a second anchor.
            return tok.replace("::", "&#58;&#58;")
        def _linkify_html_segment(seg: str) -> str:
            def _sub(m: re.Match) -> str:
                url = _token_url(_bare(m.group(0)))
                if not url:
                    return m.group(0)
                return (f'<a class="reference internal" '
                        f'href="{url}">{_anchor_text(m.group(0))}</a>')
            return _tok_re.sub(_sub, seg)
        # Pass 1: walk every `<code class="docutils literal notranslate">…</code>`
        # block; skip any `<a>` already inside (function-name anchor from
        # `_func_row_split_md`), linkify recognized tokens in the rest.
        def _linkify_inside_code(m: re.Match) -> str:
            inner = m.group("inner")
            out, i, n = [], 0, len(inner)
            while i < n:
                if inner.startswith("<a ", i):
                    j = inner.find("</a>", i)
                    if j < 0:
                        out.append(inner[i:]); break
                    out.append(inner[i:j + 4]); i = j + 4
                else:
                    k = inner.find("<a ", i)
                    if k < 0:
                        out.append(_linkify_html_segment(inner[i:])); break
                    out.append(_linkify_html_segment(inner[i:k])); i = k
            return m.group("open") + "".join(out) + m.group("close")
        text = re.sub(
            r'(?P<open><code class="docutils literal notranslate">)'
            r'(?P<inner>.*?)(?P<close></code>)',
            _linkify_inside_code, text, flags=re.DOTALL)
        # Pass 2: remaining markdown code spans — the Type-column typedef
        # chips (`InputArray`, `Mat_<uchar>`, `const _InputArray &`, …)
        # and any other backticked tokens that MyST only turns into
        # `<code><span class="pre">…</span></code>` at render time, AFTER
        # this step has run (so pass 1's `<code>` walk can't reach them,
        # and postprocess's inline walker can't either — Sphinx's
        # `<span class="pre">` wrapper defeats its body regex). Markdown
        # links are masked first so anchored Name-column cells stay intact.
        def _linkify_markdown_codespan(m: re.Match) -> str:
            content = m.group("content")
            # Prefix-aware: `cv::Name` is one hit so the anchor covers both.
            hits = [(t.start(), t.end(), t.group(0)) for t in
                    _tok_re.finditer(content)
                    if _token_url(_bare(t.group(0)))]
            if not hits:
                return m.group(0)
            from html import escape as _esc
            parts, last = [], 0
            for s, e, tok in hits:
                parts.append(_esc(content[last:s]))
                parts.append(f'<a class="reference internal" '
                             f'href="{_token_url(_bare(tok))}">'
                             f'{_anchor_text(tok)}</a>')
                last = e
            parts.append(_esc(content[last:]))
            return (f'<code class="docutils literal notranslate">'
                    f'{"".join(parts)}</code>')
        _masked: list[str] = []
        def _mask(m: re.Match) -> str:
            _masked.append(m.group(0))
            return f"\x00MDLINK{len(_masked)-1}\x00"
        text = re.sub(r"\[(?:`[^`\n]+`|<br>)+\]\([^)\n]+\)", _mask, text)
        text = re.sub(r"`(?P<content>[^`\n]+?)`",
                      _linkify_markdown_codespan, text)
        text = re.sub(r"\x00MDLINK(\d+)\x00",
                      lambda m: _masked[int(m.group(1))], text)

    # 6c. Bullet lists of @subpage/@ref -> toctree + visible list. Runs BEFORE step 7.
    def _subpage_list_to_toctree(src: str) -> str:
        bullet  = r"^[ \t]*-\s+[^\n@]*?@(?:subpage|ref)\s+[\w-]+(?:[^\n]*)\n"
        desc_re = r"(?:(?:[ \t]*\n)*(?:[ \t]+[^\n]+\n)+)*"
        pat = re.compile(rf"((?:{bullet}{desc_re}(?:[ \t]*\n)*)+)", re.MULTILINE)
        item_pat = re.compile(
            rf"^[ \t]*-\s+(?P<prefix>[^\n@]*?)@(?P<kind>subpage|ref)\s+(?P<anchor>[\w-]+)"
            rf'(?:[ \t]+"(?P<disp>[^"\n]+)")?'
            rf"(?P<inline>[^\n]*)\n"
            rf"(?P<desc>{desc_re})",
            re.MULTILINE)

        def repl(m: re.Match) -> str:
            resolved: list[tuple[str, str, str, str, str, str, str]] = []
            for im in item_pat.finditer(m.group(1)):
                kind = im.group("kind")
                anchor = im.group("anchor")
                disp = im.group("disp")
                prefix = (im.group("prefix") or "").strip()
                inline = im.group("inline") or ""
                desc_lines = [l.strip() for l in (im.group("desc") or "").splitlines() if l.strip()]
                description = " ".join(desc_lines)
                # @ref follows redirects; @subpage uses literal target.
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
                # No item resolved — not really a subpage/toctree list (e.g. a
                # bullet whose `@ref` is an inline cross-reference). Leave it
                # untouched so its body (baseline-indented code/images) isn't
                # swallowed as a bullet description and dropped; step 7 still
                # links the bare `@ref`.
                return m.group(1)

            # toctree gets only @subpage (internal @ref would create cycles).
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
            # `inline` alone stays a plain toctree; visible mode needs desc/prefix.
            if not (has_descriptions or has_prefixes):
                if tt_body:
                    return f"\n```{{toctree}}\n:maxdepth: 1\n:titlesonly:\n\n{tt_body}\n```\n"
                return ""

            # Hidden toctree (subpages) + visible list (all items).
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

    # 7. @ref name -> internal docname / Doxygen tag URL / fragment fallback.
    def _ref_repl(m: re.Match) -> str:
        name = m.group("name"); disp = m.group("disp")
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
    # `:` allowed for qualified C++ identifiers.
    text = re.sub(r'@ref\s+(?P<name>[\w:-]+)(?:\s+"(?P<disp>[^"]+)")?',
                  _ref_repl, text)

    # 7c. cv.Name -> Markdown link using _CV_SYMBOL_URL. Routed through
    # `_apply_outside_code` so fenced code (including nested in tab-sets) and
    # inline spans are skipped — a prior brittle-regex approach mis-paired
    # fence widths and leaked `[cv.foo](#anchor)` into tab-set code bodies.
    if _CV_SYMBOL_URL:
        _cv_dot_re = re.compile(
            r'(?<!\[)(?<!\()cv\.([A-Za-z][A-Za-z0-9_]*)')
        def _cvlink_repl(m: re.Match) -> str:
            url = _CV_SYMBOL_URL.get(m.group(1))
            return f'[cv.{m.group(1)}]({url})' if url else m.group(0)
        text = _apply_outside_code(
            text, lambda chunk: _cv_dot_re.sub(_cvlink_repl, chunk))

    # 8. @cite KEY -> `[N]` HTML anchor to citelist (N from opencv.bib order).
    def _cite_repl(m: re.Match) -> str:
        key = m.group("key")
        num = _CITE_NUMBER.get(key)
        label = f"[{num}]" if num is not None else f"[{key}]"
        if "citelist" in _ANCHOR_TO_DOC:
            depth = docname.count("/") if docname else 0
            href = ("../" * depth) + f"citelist.html#CITEREF_{key}"
        else:
            href = f"{DOXYGEN_BASE_URL}citelist.html#CITEREF_{key}"
        return f'<a href="{href}">{label}</a>'
    # Keys may contain ':'/'.' segments (Ma:2003:IVI, BT.709, Wulff:CVPR:2015);
    # match those without swallowing a trailing sentence period.
    text = _apply_outside_code(text, lambda chunk: re.sub(
        r"@cite\s+(?P<key>[\w-]+(?:[.:][\w-]+)*)", _cite_repl, chunk))

    # Skip plain-lowercase keys (e.g. "pattern", "eigenfaces") in the bare pass:
    # they collide with ordinary words and mis-cite prose. Mixed-case/digit keys
    # (Zhang2000, BT2017, LBP) still link bare; explicit @cite always works.
    _bare_keys = [k for k in _CITE_NUMBER if not re.fullmatch(r"[a-z]+", k)]
    if _bare_keys and docname != "citelist":
        _CITE_KEY_RE = re.compile(
            r"(?<![\[\w])(?P<key>"
            + "|".join(re.escape(k) for k in _bare_keys)
            + r")(?![\w\]])"
        )
        def _bare_cite_repl(m: re.Match) -> str:
            key = m.group("key")
            num = _CITE_NUMBER.get(key)
            label = f"[{num}]" if num is not None else f"[{key}]"
            if "citelist" in _ANCHOR_TO_DOC:
                depth = docname.count("/") if docname else 0
                href = ("../" * depth) + f"citelist.html#CITEREF_{key}"
            else:
                href = f"{DOXYGEN_BASE_URL}citelist.html#CITEREF_{key}"
            return f'<a href="{href}">{label}</a>'
        text = _apply_outside_code(
            text, lambda chunk: _CITE_KEY_RE.sub(_bare_cite_repl, chunk))

    # 8b. @youtube{ID} -> responsive raw-HTML embed.
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

    # 8c. @note / @see / @warning handled at step 1b above.

    # 8d. Dedent indented descriptions after `- @subpage X`.
    def _dedent_subpage_descriptions(src: str) -> str:
        # Continuation indent: 4+ spaces/tabs or a single leading tab.
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

    # 9z. Doxygen group/member URLs -> local cross-ref when we built that page.
    def _doxy_to_local(m: re.Match) -> str:
        base, anchor = m.group("base"), m.group("anchor")
        if anchor:
            tgt = f"{base}_1{anchor}"
            return f"](#{tgt})" if tgt in _LOCAL_MEMBER_IDS else m.group(0)
        tgt = f"api_{base[len('group__'):].replace('__', '_')}"
        return f"](#{tgt})" if tgt in _ANCHOR_TO_DOC else m.group(0)
    text = re.sub(
        r"\]\(" + re.escape(DOXYGEN_BASE_URL)
        + r"(?:[\w-]+/)*?(?P<base>group__\w+)\.html(?:#(?P<anchor>\w+))?\)",
        _doxy_to_local, text)

    # 10. @next_tutorial / @prev_tutorial -> drop
    text = re.sub(r"^@(?:next|prev)_tutorial\{[^}]*\}\s*$", "",
                  text, flags=re.MULTILINE)

    # 11. @tableofcontents / [TOC] -> drop
    text = re.sub(r"^(?:@tableofcontents|\[TOC\])\s*$", "",
                  text, flags=re.MULTILINE)

    # 11b. @cond/@endcond and @parblock/@endparblock -> strip markers.
    text = re.sub(r"^@cond\s+\S+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^@endcond\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*@parblock\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*@endparblock\s*$", "", text, flags=re.MULTILINE)

    # 11c. @anchor NAME -> MyST label "(NAME)=".
    text = re.sub(r"^@anchor\s+(?P<name>[\w-]+)\s*$",
                  lambda m: f"({m.group('name')})=",
                  text, flags=re.MULTILINE)

    # 11d. `-#` -> `1.` (indent preserved).
    text = re.sub(r"^(?P<indent>[ \t]*)-#(?P<sp>[ \t]+)",
                  lambda m: f"{m.group('indent')}1.{m.group('sp')}",
                  text, flags=re.MULTILINE)

    # 11e. Normalize 5+ space bullet markers to 3 (avoids code-block reading).
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

    # 11f. Strip one 4-space level from bullets right after a heading.
    text = re.sub(
        r"(^#{1,6}[ \t][^\n]+\n(?:[ \t]*\n)*)((?:    [ \t]*[-*+][^\n]*\n)+)",
        lambda m: m.group(1) + re.sub(r"^    ", "", m.group(2), flags=re.MULTILINE),
        text, flags=re.MULTILINE)

    # Depth-relative prefix to site root for contrib_modules/ <img src> URLs.
    _depth = docname.count("/") if docname else 0
    _contrib_url_prefix = ("../" * _depth) + "contrib_modules/"

    def _emit_contrib_img(rel_url: str, alt: str) -> str:
        """Raw-HTML <img>/<figure> for a contrib_modules/<rel> path."""
        src = _contrib_url_prefix + rel_url
        img = f'<img src="{src}" alt="{alt}"/>'
        if alt.startswith("Figure "):
            return (f'<figure>{img}'
                    f'<figcaption>{alt}</figcaption></figure>')
        return img

    # 12. Image paths -> doc-sibling, global basename, or tutorials/others/images/.
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

    # 12b. Bare image filename (no dir prefix, e.g. "shape.jpg") -> _IMAGE_INDEX lookup.
    def _bare_img_repl(m: re.Match) -> str:
        rel = m.group("rel")
        if docname:
            local = DOC_ROOT / pathlib.Path(docname).parent / "images" / rel
            if local.is_file():
                return f'{m.group("pre")}images/{rel})'
        hit = _IMAGE_INDEX.get(rel)
        return f'{m.group("pre")}/{hit})' if hit else m.group(0)
    text = re.sub(
        r'(?P<pre>!\[[^\]]*\]\()(?P<rel>[A-Za-z0-9_.-]+\.[A-Za-z]{2,4})\)',
        _bare_img_repl, text)

    # 12c. Cross-tree contrib image refs -> raw-HTML <img>, depth-relative URL.
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

    # 12d. Blank line between consecutive `Label: ![](image)` lines (skip table rows).
    text = re.sub(
        r"^(?P<line>(?!\|)[^\n]*!\[[^\]]*\]\([^)]+\)[^\n]*)\n"
        r"(?=(?!\|)[^\n]*!\[[^\]]*\]\([^)]+\))",
        r"\g<line>\n\n", text, flags=re.MULTILINE)

    # 12e. `![Figure N: caption](url)` -> MyST `{figure}`.
    text = re.sub(
        r"^(?P<indent>[ \t]*)!\[(?P<caption>Figure\s[^\]]+)\]\((?P<url>[^)]+)\)\s*$",
        lambda m: (f"{m.group('indent')}:::{{figure}} {m.group('url')}\n"
                   f"{m.group('indent')}{m.group('caption')}\n"
                   f"{m.group('indent')}:::"),
        text, flags=re.MULTILINE)

    # 13. Wrap front-matter table in a `.opencv-meta-table` div.
    def _wrap_front_matter(src: str) -> str:
        pat = re.compile(
            r"(^\|[^\n]*\|[ \t]*\n"
            r"\|[ \t]*-:[ \t]*\|[ \t]*:-[ \t]*\|[ \t]*\n"
            r"(?:\|[^\n]*\|[ \t]*\n)+)",
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            return f":::{{div}} opencv-meta-table\n\n{m.group(1)}\n:::\n"
        return pat.sub(repl, src, count=1)
    text = _wrap_front_matter(text)

    # 14a. Auto-link `cv.SymbolName` in prose.
    text = _linkify_cv_symbols(text)

    # 14b. Wrap bare URLs in `<...>` (runs after 14a).
    text = _linkify_bare_urls(text)

    # 14c. Resolve `opencv_source_code/<path>` Doxygen alias -> GitHub link.
    text = _linkify_opencv_source_code(text)

    # 14d. Resolve Doxygen `#funcName` cross-references in prose -> link.
    text = _linkify_dox_hash_refs(text)

    # 15. Restore @verbatim stash (see step 0v).
    for _vk, _vv in _verbatim_stash.items():
        text = text.replace(_vk, _vv)

    return text



_BARE_URL_RE = re.compile(
    # Block a markdown-link URL via a 2-char lookbehind on `](`, instead of
    # excluding a bare `(` — so a URL in plain parens, e.g.
    # `PyPI (https://pypi.org/...)`, still gets autolinkified while
    # markdown-link/autolink/HTML-attribute exclusions stay intact.
    r"(?<!\]\()"
    r"(?<![<\[\w\"'=])"
    r"(?P<url>https?://[^\s<>()`\"']+[^\s<>()`\"'.,;:!?])"
)
# `cv.X`/`cv::X`; lookbehind blocks `frame.cv.X`-style false positives.
_CV_SYMBOL_RE = re.compile(
    r"(?<![/\w.:\[])(?P<sep>cv(?:\.|::))(?P<sym>[A-Za-z_]\w*)(?P<parens>\(\))?"
)
# Bare `funcName()` (Doxygen auto-linked these).
_BARE_FN_RE = re.compile(r"(?<![/\w.<>\"])(?P<sym>[A-Za-z_]\w{2,})\(\)")
_FENCED_BLOCK_RE = re.compile(
    r"^(?P<fence>[`~]{3,})[^\n]*\n[\s\S]*?\n(?P=fence)[ \t]*$",
    re.MULTILINE,
)
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*?`+")
# ATX heading line; exempted from the auto-linkifier.
_ATX_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]")
# Per-line fence detector for `_code_regions`. Deliberately allows ANY
# leading whitespace (not CommonMark's ≤3): our pipeline emits fences at
# the 4+-space indent of their originating `@include`/`@snippet` inside an
# `@add_toggle`, and under the strict limit those slipped past, leaking
# `[cv.foo](#anchor)` into code blocks. Worst case of the relaxation is
# MORE text protected from the linkifiers, not less.
_FENCE_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$"
)


def _code_regions(src: str) -> list[tuple[int, int]]:
    """Return sorted `[(start, end), ...]` byte ranges spanning every fenced
    code block (fence lines included) in `src`.

    Handles MyST/Sphinx nested fences where the outer container is wider than
    the inner code fence (e.g. `{tab-set}` (6) > `{tab-item}` (5) >
    ` ```python` (3)). Only *code*-block ranges are returned; a `{directive}`
    info string marks a container, whose body the caller still walks so its
    nested prose stays linkifiable. Inside a code fence, further fence-shaped
    lines are inert per CommonMark §4.5."""
    out: list[tuple[int, int]] = []
    # Each stack entry: (fence_char, fence_width, is_code, opener_offset).
    stack: list[tuple[str, int, bool, int]] = []
    pos = 0
    for line in src.splitlines(keepends=True):
        line_start = pos
        next_pos = pos + len(line)
        ln = line.rstrip("\n").rstrip("\r")
        m = _FENCE_LINE_RE.match(ln)
        if m is None:
            pos = next_pos
            continue
        fc = m.group("fence")[0]
        fw = len(m.group("fence"))
        info = m.group("info").strip()
        # Inside a code fence nothing matters except the matching closer.
        if stack and stack[-1][2]:
            top_char, top_width, _, opener_start = stack[-1]
            if fc == top_char and fw >= top_width and not info:
                out.append((opener_start, next_pos))
                stack.pop()
            pos = next_pos
            continue
        # Outside code: an info-less fence of the same char & ≥ width closes
        # the nearest open fence (a container, per the branch above); a fence
        # with info opens a new block.
        if stack and not info:
            top_char, top_width, _, _ = stack[-1]
            if fc == top_char and fw >= top_width:
                stack.pop()
                pos = next_pos
                continue
        # Opener: `{…}` info → container, anything else (language tag, plain
        # text, or empty) → code block per CommonMark.
        is_code = (not info) or (not info.startswith("{"))
        stack.append((fc, fw, is_code, line_start))
        # Shield a breathe directive's opener line: its argument (e.g.
        # `{doxygenstruct} cv::MSTEdge`) must reach breathe unlinkified.
        if info.startswith("{doxygen"):
            out.append((line_start, next_pos))
        pos = next_pos
    # Unclosed code fences: extend protection to EOF so we don't mangle
    # the tail of a pathologically-truncated document.
    while stack:
        char, width, is_code, opener_start = stack.pop()
        if is_code:
            out.append((opener_start, len(src)))
    out.sort()
    return out


def _apply_outside_code(src: str, transform) -> str:
    """Apply `transform` to regions outside fenced and inline code, using
    `_code_regions` so nested container > code fences (tab-set > tab-item >
    ` ```python`) are excluded correctly."""
    def _segment(text: str) -> str:
        out, last = [], 0
        for cm in _INLINE_CODE_RE.finditer(text):
            out.append(transform(text[last:cm.start()]))
            out.append(cm.group(0))
            last = cm.end()
        out.append(transform(text[last:]))
        return "".join(out)
    out, last = [], 0
    for s, e in _code_regions(src):
        out.append(_segment(src[last:s]))
        out.append(src[s:e])
        last = e
    out.append(_segment(src[last:]))
    return "".join(out)


def _linkify_bare_urls(src: str) -> str:
    return _apply_outside_code(src,
        lambda chunk: _BARE_URL_RE.sub(r"<\g<url>>", chunk))

_OPENCV_SOURCE_CODE_RE = re.compile(
    r"\bopencv_source_code/(?P<path>[\w./\-]+)"
)


def _linkify_opencv_source_code(src: str) -> str:
    def _repl(m: re.Match) -> str:
        path = m.group("path")
        url = f"https://github.com/opencv/opencv/blob/5.x/{path}"
        return f"[opencv_source_code/{path}]({url})"
    return _apply_outside_code(
        src, lambda chunk: _OPENCV_SOURCE_CODE_RE.sub(_repl, chunk))

_DOX_HASH_REF_RE = re.compile(
    r"(?<![{(#\w])#(?P<name>[A-Za-z_]\w*)\b"
)


def _linkify_dox_hash_refs(src: str) -> str:
    if not _CV_SYMBOL_URL:
        return src
    def _repl(m: re.Match) -> str:
        name = m.group("name")
        url = _CV_SYMBOL_URL.get(name)
        if not url:
            return m.group(0)
        return f"[{name}]({url})"
    return _apply_outside_code(
        src, lambda chunk: _DOX_HASH_REF_RE.sub(_repl, chunk))


def _linkify_cv_symbols(src: str) -> str:
    if not _CV_SYMBOL_URL:
        return src
    def _local_url(sym: str) -> str | None:
        """Prefer a Sphinx-local URL when one exists for `sym`.
        `_LOCAL_CLASS_URL` and `_LOCAL_TYPEDEF_URL` are populated from
        the tagfile but rewritten to point at the local api/ tree
        (e.g. `classcv_1_1Mat.html`, `core_basic.html#vec2b`). If no
        local entry exists, return None so the caller drops the link —
        we never bounce readers off-site for symbols whose Sphinx
        version isn't present in this build.
        """
        return _LOCAL_CLASS_URL.get(sym) or _LOCAL_TYPEDEF_URL.get(sym)
    def repl_cv(m: re.Match) -> str:
        sym = m.group("sym")
        url = _local_url(sym)
        if not url:
            return m.group(0)
        sep = m.group("sep")
        parens = m.group("parens") or ""
        return f'<a href="{url}">{sep}{sym}{parens}</a>'
    def repl_bare(m: re.Match) -> str:
        sym = m.group("sym")
        url = _local_url(sym)
        if not url:
            return m.group(0)
        return f'<a href="{url}">{sym}()</a>'
    def transform(chunk: str) -> str:
        # Headings are definitions — don't linkify.
        out_lines = []
        for line in chunk.split("\n"):
            if _ATX_HEADING_RE.match(line):
                out_lines.append(line)
                continue
            line = _CV_SYMBOL_RE.sub(repl_cv, line)
            line = _BARE_FN_RE.sub(repl_bare, line)
            out_lines.append(line)
        return "\n".join(out_lines)
    return _apply_outside_code(src, transform)


_referenced_docs_cache: set[str] | None = None


def _referenced_docs() -> set[str]:
    """Docnames reachable from some @subpage/@ref (computed once)."""
    global _referenced_docs_cache
    if _referenced_docs_cache is None:
        _referenced_docs_cache = {
            _ANCHOR_TO_DOC[a] for a in _REFERENCED_ANCHORS
            if a in _ANCHOR_TO_DOC
        }
    return _referenced_docs_cache


_TUTORIAL_PREFIXES = ("tutorials/", "js_tutorials/",
                      "py_tutorials/", "tutorials_contrib/")


def _source_read(app, docname, source):
    # Translate tutorial docs + API stubs; doxygengroup bodies pass through.
    if not (docname.startswith("tutorials/")
            or docname.startswith("js_tutorials/")
            or docname.startswith("py_tutorials/")
            or docname.startswith("tutorials_contrib/")
            or docname.startswith("main_modules/")
            or docname.startswith("extra_modules/")
            or docname == "faq"
            or docname == "citelist"
            or docname == "intro"
            or docname == "index"):
        return
    text = source[0]
    # Master doc: append contrib/api roots without editing tutorials.markdown.
    if docname == "tutorials/tutorials" and not USE_INDEX_LANDING:
        if CONTRIB_MODULES and "tutorial_contrib_root" in _ANCHOR_TO_DOC:
            text = text.rstrip() + "\n\n- @subpage tutorial_contrib_root\n"
        if API_MODULES and "api_root" in _ANCHOR_TO_DOC:
            text = text.rstrip() + "\n\n- @subpage api_root\n"
    out = _translate(text, docname)
    # Mark unreferenced tutorial pages :orphan: (skip master/front-matter/referenced).
    if (docname.startswith(_TUTORIAL_PREFIXES)
            and docname != "tutorials/tutorials"
            and docname not in _referenced_docs()
            and not out.lstrip().startswith("---")):
        out = "---\norphan: true\n---\n\n" + out
    source[0] = out
