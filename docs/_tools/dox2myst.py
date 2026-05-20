"""Convert one OpenCV Doxygen tutorial (.markdown) -> MyST (.md).

Tagfile: download once from https://docs.opencv.org/5.x/opencv.tag.
Local refs: optional JSON map of {doxygen_anchor: sphinx_docname} so @ref to
pages we host locally becomes an internal link instead of an external one.

Usage:
  python3 dox2myst.py <input.markdown> <output.md>
                      [--tag /path/to/opencv.tag]
                      [--local /path/to/local_refs.json]
                      [--out-doc <sphinx_docname_of_output>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DOCS_BASE = "https://docs.opencv.org/5.x/"


_ADMON_KINDS = {
    "note": "note", "warning": "warning", "attention": "attention",
    "important": "important", "tip": "tip",
}
_ADMON_START = re.compile(
    r"^(?P<indent>[ \t]*)@(?P<tag>note|warning|attention|important|tip)"
    r"(?P<sep>[: \t]+)?(?P<rest>.*)$"
)
_OTHER_DIRECTIVE = re.compile(
    r"^[ \t]*@(see|cite|param|return|brief|details|since|throws|sa|"
    r"prev_tutorial|next_tutorial|tableofcontents|youtube|todo|subpage|"
    r"ref|cond|endcond|snippet)\b"
)


_TOGGLE_LABELS = {
    "cpp": "C++", "python": "Python", "java": "Java",
    "javascript": "JavaScript", "csharp": "C#", "fortran": "Fortran",
}
_EMPTY_TOGGLE_RE = re.compile(
    r"^[ \t]*@add_toggle_\w+[ \t]*\n[ \t]*@end_toggle[ \t]*\n?",
    re.M,
)
_TOGGLE_RE = re.compile(
    r"^[ \t]*@add_toggle_(?P<lang>\w+)[ \t]*\n(?P<body>.*?)\n[ \t]*@end_toggle[ \t]*$",
    re.S | re.M,
)


def _convert_toggle_blocks(text: str) -> str:
    """Replace runs of consecutive `@add_toggle_X ... @end_toggle` blocks with
    a sphinx-design tab-set containing one tab-item per block."""

    def _block_to_tab(m: re.Match) -> str:
        lang = m.group("lang")
        label = _TOGGLE_LABELS.get(lang, lang.title())
        body = m.group("body").strip("\n")
        return f"@@TAB_ITEM@@ {lang}|{label}\n{body}\n@@END_TAB_ITEM@@"

    text = _EMPTY_TOGGLE_RE.sub("", text)
    text = _TOGGLE_RE.sub(_block_to_tab, text)

    lines = text.splitlines(keepends=False)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].lstrip().startswith("@@TAB_ITEM@@"):
            tabs: list[tuple[str, str, list[str]]] = []
            seen_keys: set[str] = set()
            while i < n:
                ln = lines[i].lstrip()
                if ln.startswith("@@TAB_ITEM@@"):
                    raw = ln[len("@@TAB_ITEM@@"):].strip()
                    sync_key, _, label = raw.partition("|")
                    if not label:
                        label = sync_key
                    if sync_key in seen_keys:
                        break
                    seen_keys.add(sync_key)
                    body: list[str] = []
                    i += 1
                    while i < n and not lines[i].lstrip().startswith("@@END_TAB_ITEM@@"):
                        body.append(lines[i])
                        i += 1
                    if i < n and lines[i].lstrip().startswith("@@END_TAB_ITEM@@"):
                        i += 1
                    tabs.append((sync_key, label, body))
                    while i < n and not lines[i].strip():
                        i += 1
                    continue
                break
            if len(tabs) == 1:
                out.extend(tabs[0][2])
                out.append("")
            else:
                out.append("::::{tab-set}")
                for sync_key, label, body in tabs:
                    out.append(f":::{{tab-item}} {label}")
                    out.append(f":sync: {sync_key}")
                    out.append("")
                    out.extend(body)
                    out.append(":::")
                out.append("::::")
                out.append("")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _extract_admonitions(text: str) -> str:
    lines = text.splitlines(keepends=False)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        m = _ADMON_START.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent = m.group("indent")
        ind_len = len(indent)
        kind = _ADMON_KINDS[m.group("tag")]
        first = (m.group("rest") or "").lstrip(":").lstrip()
        body: list[str] = []
        if first:
            body.append(first)
        j = i + 1
        saw_blank = False
        while j < n:
            ln = lines[j]
            stripped = ln.lstrip(" ")
            cur_indent = len(ln) - len(stripped)
            if not stripped:
                body.append("")
                saw_blank = True
                j += 1
                continue
            if cur_indent < ind_len:
                break
            if cur_indent <= ind_len and saw_blank:
                break
            if _ADMON_START.match(ln) or _OTHER_DIRECTIVE.match(ln):
                break
            body.append(ln[ind_len:] if cur_indent >= ind_len else ln)
            saw_blank = False
            j += 1
        while body and not body[-1].strip():
            body.pop()

        # Per-paragraph dedent: keep first paragraph as-is, then for each
        # blank-line-separated paragraph strip its own leading indent so
        # nested @code blocks etc. don't trigger CommonMark code-block parsing.
        paragraphs: list[list[str]] = []
        current: list[str] = []
        for ln in body:
            if not ln.strip():
                if current:
                    paragraphs.append(current)
                    current = []
                paragraphs.append([])
            else:
                current.append(ln)
        if current:
            paragraphs.append(current)
        normalised: list[str] = []
        for para in paragraphs:
            if not para:
                normalised.append("")
                continue
            indents = [len(ln) - len(ln.lstrip(" ")) for ln in para]
            strip = min(indents)
            for ln in para:
                normalised.append(ln[strip:] if len(ln) >= strip else ln)

        out.append(f"{indent}:::{{{kind}}}")
        for b in normalised:
            out.append(indent + b if b else "")
        out.append(f"{indent}:::")
        i = j
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _convert_numbered_steps(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    step_start = re.compile(r"^(\s*)-#\s+(.*?)\s*$")
    section_break = re.compile(r"^#{1,6}\s|^\(\S+\)=\s*$")

    while i < n:
        m = step_start.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent = m.group(1)
        title = m.group(2)
        out.append(f"{indent}1. {title}\n")
        i += 1
        body_indent = len(indent) + 4
        while i < n:
            ln = lines[i]
            if step_start.match(ln) or section_break.match(ln):
                break
            stripped = ln.lstrip(" ")
            cur_indent = len(ln) - len(stripped)
            if not stripped.strip():
                out.append(ln)
                i += 1
                continue
            if cur_indent >= body_indent:
                content_col = len(indent) + 3
                out.append(" " * content_col + ln[body_indent:])
            else:
                out.append(ln)
            i += 1

    return "".join(out)


class TagIndex:
    def __init__(self, tagfile: Path):
        self.by_name: dict[str, str] = {}
        self.pages: dict[str, tuple[str, str]] = {}
        self.groups: dict[str, tuple[str, str]] = {}
        self.cites: dict[str, str] = {}
        self.docanchors: dict[str, tuple[str, str, str]] = {}
        self._load(tagfile)

    def _load(self, tagfile: Path) -> None:
        root = ET.parse(tagfile).getroot()
        for compound in root.iter("compound"):
            kind = compound.get("kind")
            cname = (compound.findtext("name") or "").strip()
            cfile = (compound.findtext("filename") or "").strip()
            ctitle = (compound.findtext("title") or "").strip()

            if kind == "page" and cname and cfile:
                self.pages[cname] = (DOCS_BASE + cfile, ctitle)
            if kind == "group" and cname and cfile:
                self.groups.setdefault(cname, (DOCS_BASE + cfile, ctitle))
            if kind in ("class", "struct") and cname and cfile:
                self.by_name.setdefault(cname, DOCS_BASE + cfile)

            for da in compound.findall("docanchor"):
                anchor = (da.text or "").strip()
                page_file = (da.get("file") or "").strip()
                anchor_title = (da.get("title") or "").strip()
                if anchor and page_file:
                    self.docanchors.setdefault(
                        anchor, (cname, DOCS_BASE + page_file + "#" + anchor, anchor_title)
                    )

            for m in compound.findall("member"):
                mname = (m.findtext("name") or "").strip()
                mfile = (m.findtext("anchorfile") or "").strip()
                manchor = (m.findtext("anchor") or "").strip()
                if not (mname and mfile and manchor):
                    continue
                url = DOCS_BASE + mfile + "#" + manchor
                self.by_name.setdefault(mname, url)
                if kind in ("class", "struct") and cname:
                    self.by_name.setdefault(f"{cname}::{mname}", url)
                if manchor.startswith("CITEREF_"):
                    self.cites[mname] = url

    def lookup(self, name: str) -> tuple[str, str] | None:
        if name in self.pages:
            url, title = self.pages[name]
            return url, title
        if name in self.docanchors:
            _page, url, title = self.docanchors[name]
            return url, title
        if name in self.groups:
            url, title = self.groups[name]
            return url, title
        if "::" in name and name in self.by_name:
            return self.by_name[name], ""
        if name in self.by_name:
            return self.by_name[name], ""
        if ("cv::" + name) in self.by_name:
            return self.by_name["cv::" + name], ""
        if name.startswith("cv::"):
            bare = name[4:]
            if bare in self.pages:
                url, title = self.pages[bare]
                return url, title
            if bare in self.groups:
                url, title = self.groups[bare]
                return url, title
            if bare in self.by_name:
                return self.by_name[bare], ""
        return None


def _relative_docname(from_doc: str, to_doc: str) -> str:
    from_dir = os.path.dirname(from_doc)
    rel = os.path.relpath(to_doc, from_dir or ".")
    return rel.replace(os.sep, "/")


def transform(text: str, tags: TagIndex, *,
              local_refs: dict[str, str] | None = None,
              out_doc: str = "") -> str:
    local_refs = local_refs or {}
    m = re.match(
        r"^(?P<title>[^\n]+?)(?:\s*\{\#(?P<anchor>[^}]+)\})?\s*\n=+\s*\n",
        text,
    )
    if m:
        title = m.group("title").strip()
        rest = text[m.end():]
        rest = re.sub(r"^(#{1,5})(\s)", r"#\1\2", rest, flags=re.M)
        text = f"# {title}\n\n" + rest

    text = re.sub(r"^(\s*```)\.([A-Za-z0-9_+-]+)\s*$", r"\1\2", text, flags=re.M)

    def _dedent_fence(m: re.Match) -> str:
        head, body, tail = m.group(1), m.group(2), m.group(3)
        lines = body.splitlines()
        indents = [len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip()]
        if not indents:
            return m.group(0)
        n = min(indents)
        if n == 0:
            return m.group(0)
        body = "\n".join(
            ln[n:] if len(ln) >= n and not ln[:n].strip() else ln
            for ln in lines
        )
        return head + body + tail
    text = re.sub(r"(```[A-Za-z0-9_+-]*\n)(.*?)(\n```)", _dedent_fence, text, flags=re.S)

    text = re.sub(r"^@tableofcontents\s*$", "", text, flags=re.M)
    text = re.sub(r"^\[TOC\]\s*$", "", text, flags=re.M)
    text = re.sub(r"^@cond\s+\S+\s*$", "", text, flags=re.M)
    text = re.sub(r"^@endcond\s*$", "", text, flags=re.M)

    text = re.sub(
        r"^[ \t]*@anchor[ \t]+([A-Za-z_][A-Za-z0-9_:]*)[ \t]*$",
        lambda m: f'<a id="{m.group(1).replace("_", "-")}"></a>',
        text,
        flags=re.M,
    )

    verbatim_stash: list[str] = []

    def _stash_verbatim(m: re.Match) -> str:
        verbatim_stash.append(m.group(1).strip())
        return f"\x00VERBATIM_{len(verbatim_stash) - 1}\x00"
    text = re.sub(r"@verbatim\b(.+?)@endverbatim",
                  _stash_verbatim, text, flags=re.S)

    inline_link_defs: dict[str, str] = {}
    for m in re.finditer(r"^\[([^\]]+)\]:\s+(\S.*?)\s*$", text, flags=re.M):
        inline_link_defs[m.group(1).strip()] = m.group(2).strip()

    def _inline_ref_link(m: re.Match) -> str:
        label = m.group(1)
        url = inline_link_defs.get(label.strip())
        if url is None:
            return m.group(0)
        return f"[{label}]({url})"
    if inline_link_defs:
        text = re.sub(r"(?<![\[\]\\])\[([^\]\n]+)\](?![\(\[:])",
                      _inline_ref_link, text)
    def _admon_parblock(m: re.Match) -> str:
        indent = m.group(1)
        kind = m.group(2)
        body = m.group(3).rstrip()
        body_lines = body.splitlines()
        body_indents = [len(ln) - len(ln.lstrip(" "))
                        for ln in body_lines if ln.strip()]
        strip_n = min(body_indents) if body_indents else 0
        body_lines = [ln[strip_n:] if len(ln) >= strip_n else ln
                      for ln in body_lines]
        body_text = "\n".join(indent + ln if ln else "" for ln in body_lines)
        return f"{indent}:::{{{kind}}}\n{body_text}\n{indent}:::\n"
    text = re.sub(
        r"^([ \t]*)@(note|warning|attention|important|tip)[ \t]*\n"
        r"[ \t]*@parblock[ \t]*\n(.*?)@endparblock[ \t]*$",
        _admon_parblock, text, flags=re.S | re.M,
    )
    text = re.sub(r"@parblock\b[ \t]*", "", text)
    text = re.sub(r"@endparblock\b[ \t]*", "", text)

    text = re.sub(r"(?<!\\)\\n(?=\s)", "<br>", text)
    text = re.sub(r"^[ \t]*<br\s*/?>[ \t]*$", "", text, flags=re.M)

    text = _convert_toggle_blocks(text)
    text = re.sub(r"^@(prev|next)_tutorial\{[^}]+\}\s*$", "", text, flags=re.M)

    text = re.sub(r"^([^\n]+?)\s*\{#([A-Za-z_][A-Za-z0-9_]*)\}\s*\n=+\s*$",
                  r"(\2)=\n\1\n=========", text, flags=re.M)
    text = re.sub(r"^([^\n]+?)\s*\{#([A-Za-z_][A-Za-z0-9_]*)\}\s*\n-+\s*$",
                  r"(\2)=\n\1\n--------", text, flags=re.M)
    text = re.sub(r"^(#{1,6}\s+.+?)\s*\{#([A-Za-z_][A-Za-z0-9_]*)\}\s*$",
                  r"(\2)=\n\1", text, flags=re.M)

    text = re.sub(r"\\f\]\s*\\f\[", "\\\\f]\n\n\\\\f[", text)

    def _bordermatrix_sub(m: re.Match) -> str:
        body = m.group(1)
        first_row = body.split(r"\cr")[0]
        ncols = first_row.count("&") + 1
        cols = "c" * ncols
        rows = [r.strip() for r in body.split(r"\cr") if r.strip()]
        return r"\begin{array}{" + cols + "}" + r" \\ ".join(rows) + r"\end{array}"
    text = re.sub(r"\\bordermatrix\s*\{(.+?)\}", _bordermatrix_sub, text, flags=re.S)

    text = re.sub(r"^([ \t]*)\\f\[(.+?)\\f\]",
                  lambda m: f"{m.group(1)}$$\n{m.group(1)}{m.group(2).strip()}\n{m.group(1)}$$",
                  text, flags=re.S | re.M)
    text = re.sub(r"\\f\$(.+?)\\f\$", lambda m: f"${m.group(1)}$",
                  text, flags=re.S)

    def _escape_mul_stars_in_italic(m: re.Match) -> str:
        inner = re.sub(r"(?<!\\)\*(?=\d)", r"\\*", m.group(1))
        return f"*{inner}*"
    text = re.sub(
        r"\*(?=[A-Za-z(])((?:[^*\n]|\*(?=\d))+)\*(?=[ \t,;:.()\n]|$)",
        _escape_mul_stars_in_italic, text, flags=re.M)

    def _escape_unmatched_italic(m: re.Match) -> str:
        return "\\*" + m.group(1).replace("*", r"\*") + r"\*"
    text = re.sub(r"\*([^*\n]+\*[^*\n]+)\\\*(?![^*\n]*\*)", _escape_unmatched_italic, text)

    text = re.sub(r"^([^\n]+?)\n-{3,}\s*\n", r"## \1\n\n", text, flags=re.M)

    def _html_heading_to_md(m: re.Match) -> str:
        level = int(m.group(1))
        return "#" * level + " " + m.group(2).strip()
    text = re.sub(r"(?i)<[Hh]([1-6])>(.*?)</[Hh][1-6]>", _html_heading_to_md, text)

    text = _convert_numbered_steps(text)

    def snippet_sub(m: re.Match) -> str:
        indent = m.group(1)
        path, tag = m.group(2).strip(), m.group(3).strip()
        ext = Path(path).suffix.lower()
        lang = {".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".cxx": "cpp",
                ".py": "python", ".js": "javascript",
                ".java": "java", ".sh": "bash", ".bash": "bash",
                ".cmake": "cmake", ".cmd": "bat", ".bat": "bat",
                ".xml": "xml", ".yaml": "yaml", ".yml": "yaml"}.get(ext, "cpp")
        return (
            "\n" + indent + "```{doxysnippet} " + path + "\n"
            + indent + ":tag: " + tag + "\n"
            + indent + ":language: " + lang + "\n"
            + indent + "```\n"
        )
    text = re.sub(r"^([ \t]*)@snippet[ \t]+(\S+)[ \t]+(.+?)[ \t]*$",
                  snippet_sub, text, flags=re.M)

    def include_sub(m: re.Match) -> str:
        indent = m.group(1)
        path = m.group(2).strip()
        ext = Path(path).suffix.lower()
        lang = {".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".cxx": "cpp",
                ".py": "python", ".js": "javascript",
                ".java": "java", ".sh": "bash", ".bash": "bash",
                ".cmake": "cmake", ".cmd": "bat", ".bat": "bat",
                ".xml": "xml", ".yaml": "yaml", ".yml": "yaml"}.get(ext, "cpp")
        return (
            "\n" + indent + "```{doxyinclude} " + path + "\n"
            + indent + ":language: " + lang + "\n"
            + indent + "```\n"
        )
    text = re.sub(r"^([ \t]*)@include[ \t]+(\S+)[ \t]*$",
                  include_sub, text, flags=re.M)

    text = _extract_admonitions(text)

    def code_sub(m: re.Match) -> str:
        indent = m.group(1)
        lang = (m.group(2) or "").lstrip(".")
        body = m.group(3)
        body_lines = body.strip("\n").splitlines()
        body_indents = [len(ln) - len(ln.lstrip(" "))
                        for ln in body_lines if ln.strip()]
        strip_n = min(body_indents) if body_indents else 0
        body_lines = [ln[strip_n:] if len(ln) >= strip_n else ln for ln in body_lines]
        body_text = "\n".join(indent + ln if ln else "" for ln in body_lines)
        return f"\n{indent}```{lang}\n{body_text}\n{indent}```\n"
    text = re.sub(r"^([ \t]*)@code(?:\{([^}]+)\})?[ \t]*\n(.*?)@endcode",
                  code_sub, text, flags=re.S | re.M)

    def verbatim_block(m: re.Match) -> str:
        indent = m.group(1)
        body = m.group(2)
        body_lines = body.strip("\n").splitlines()
        body_indents = [len(ln) - len(ln.lstrip(" "))
                        for ln in body_lines if ln.strip()]
        strip_n = min(body_indents) if body_indents else 0
        body_lines = [ln[strip_n:] if len(ln) >= strip_n else ln for ln in body_lines]
        body_text = "\n".join(indent + ln if ln else "" for ln in body_lines)
        return f"\n{indent}```\n{body_text}\n{indent}```\n"
    text = re.sub(r"^([ \t]*)@verbatim[ \t]*\n(.*?)@endverbatim",
                  verbatim_block, text, flags=re.S | re.M)

    def admon_sub(kind: str):
        def _sub(m: re.Match) -> str:
            indent = m.group(1)
            body = m.group(2).rstrip()
            body_lines = body.splitlines()
            body_indents = [len(ln) - len(ln.lstrip(" "))
                            for ln in body_lines if ln.strip()]
            strip_n = min(body_indents) if body_indents else 0
            body_lines = [ln[strip_n:] if len(ln) >= strip_n else ln for ln in body_lines]
            body_text = "\n".join(indent + ln if ln else "" for ln in body_lines)
            return f"{indent}:::{{{kind}}}\n{body_text}\n{indent}:::\n"
        return _sub


    def see_block(m: re.Match) -> str:
        block = m.group(0)
        targets = [ln.strip()[len("@see"):].strip()
                   for ln in block.splitlines() if ln.strip().startswith("@see")]
        items = "\n".join(f"- {t}" for t in targets if t)
        return ":::{seealso}\n" + items + "\n:::\n"
    text = re.sub(
        r"(?:^@see[ \t]+.+\n)(?:[ \t]*\n?^@see[ \t]+.+\n)*",
        see_block, text, flags=re.M,
    )

    def youtube_sub(m: re.Match) -> str:
        vid = m.group(1).strip()
        return (
            "\n```{raw} html\n"
            '<div class="responsive-iframe" '
            'style="position:relative;padding-bottom:56.25%;height:0;'
            'overflow:hidden;max-width:100%;margin:1.5rem 0;">\n'
            '  <iframe style="position:absolute;top:0;left:0;width:100%;'
            'height:100%;border:0;" '
            f'src="https://www.youtube-nocookie.com/embed/{vid}?rel=0" '
            'title="YouTube video" '
            'allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
            'gyroscope; picture-in-picture" allowfullscreen></iframe>\n'
            "</div>\n"
            "```\n"
        )
    text = re.sub(r"@youtube\{([^}]+)\}", youtube_sub, text)

    def cite_sub(m: re.Match) -> str:
        key = m.group(1)
        url = tags.cites.get(key) or (
            DOCS_BASE + "d0/de3/citelist.html#CITEREF_" + key
        )
        return f"[\\[{key}\\]]({url})"
    text = re.sub(r"@cite\s+([A-Za-z0-9_:]+)", cite_sub, text)

    def _resolve(target: str, override: str | None) -> str:
        title = ""
        result = tags.lookup(target)
        if result is not None:
            _, title = result
        if target in local_refs:
            dst = local_refs[target]
            href = _relative_docname(out_doc, dst) if out_doc else dst
            label = override or title or target
            return f"[{label}]({href}.md)"
        if target in tags.docanchors:
            page_name, _url, anchor_title = tags.docanchors[target]
            if page_name in local_refs:
                dst = local_refs[page_name]
                href = _relative_docname(out_doc, dst) if out_doc else dst
                label = override or anchor_title or target
                return f"[{label}]({href}.md#{target.replace('_', '-')})"
        if result is None:
            return f"`{override or target}`"
        url, _ = result
        label = override or title or target
        return f"[{label}]({url})"

    def _resolve_url_only(target: str) -> str:
        if target in local_refs:
            dst = local_refs[target]
            href = _relative_docname(out_doc, dst) if out_doc else dst
            return f"{href}.md"
        if target in tags.docanchors:
            page_name, _url, _title = tags.docanchors[target]
            if page_name in local_refs:
                dst = local_refs[page_name]
                href = _relative_docname(out_doc, dst) if out_doc else dst
                return f"{href}.md#{target.replace('_', '-')}"
            return _url
        result = tags.lookup(target)
        return result[0] if result else target

    text = re.sub(
        r"\]\(@ref\s+([A-Za-z_][A-Za-z0-9_:]*)\)",
        lambda m: f"]({_resolve_url_only(m.group(1))})",
        text,
    )

    ref_pat = re.compile(
        r'@ref\s+([A-Za-z_][A-Za-z0-9_:]*)(?:\s+"([^"]+)")?'
    )
    text = ref_pat.sub(lambda m: _resolve(m.group(1), m.group(2)), text)

    sub_pat = re.compile(
        r'@subpage\s+([A-Za-z_][A-Za-z0-9_:]*)(?:\s+"([^"]+)")?'
    )
    text = sub_pat.sub(lambda m: _resolve(m.group(1), m.group(2)), text)

    def asset_link(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        return f"[{label}](images/{target})"
    text = re.sub(
        r"\[([^\]]*)\]\(([^/)\s]+\.(?:png|jpg|jpeg|gif|svg|webp))\)",
        asset_link, text, flags=re.I,
    )

    text = _autolink_cv_symbols(text, tags)
    text = _NOLINK_PREFIX_RE.sub("", text)

    while True:
        new = re.sub(
            r":::\{(note|warning|attention|important|tip|seealso)\}\n"
            r"((?:(?!\n:::)[\s\S])*?)\n"
            r":::[ \t]*\n+"
            r":::\{\1\}\n"
            r"((?:(?!\n:::)[\s\S])*?)\n"
            r":::",
            lambda m: f":::{{{m.group(1)}}}\n{m.group(2).rstrip()}\n{m.group(3).lstrip()}\n:::",
            text,
        )
        if new == text:
            break
        text = new

    def _restore_verbatim_lines(text: str) -> str:
        out_lines: list[str] = []
        ph_re = re.compile(r"\x00VERBATIM_(\d+)\x00")
        for ln in text.splitlines():
            m = ph_re.search(ln)
            if not m:
                out_lines.append(ln)
                continue
            indent_match = re.match(r"^(\s*)", ln)
            indent = indent_match.group(1) if indent_match else ""
            cursor = 0
            pieces_pre: list[str] = []
            blocks: list[list[str]] = []
            pieces_post: list[str] = []
            cur_pre = []
            for sub in ph_re.finditer(ln):
                cur_pre.append(ln[cursor:sub.start()])
                idx = int(sub.group(1))
                body = verbatim_stash[idx]
                blocks.append(body.splitlines() if body else [])
                cursor = sub.end()
            cur_post = ln[cursor:]
            inline_text_parts = cur_pre + [cur_post]
            text_before = inline_text_parts[0].rstrip()
            if text_before.strip():
                out_lines.append(text_before)
            for i, body_lines in enumerate(blocks):
                out_lines.append("")
                out_lines.append(indent + "```")
                for bl in body_lines:
                    out_lines.append(indent + bl if bl else "")
                out_lines.append(indent + "```")
                out_lines.append("")
                between = inline_text_parts[i + 1].strip()
                if between:
                    out_lines.append(indent + between)
        return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")
    text = _restore_verbatim_lines(text)

    text = _indent_orphan_code_fences(text)
    text = _fix_overindented_sub_bullets(text)
    text = _convert_module_bullets_to_table(text)
    text = _convert_rowspan_tables(text)
    text = _deindent_toplevel_numbered_lists(text)
    text = _convert_images_to_figures(text)
    text = _wrap_metadata_table(text)

    text = _ensure_blank_lines_around_images(text)
    text = _ensure_blank_lines_around_display_math(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.lstrip("\n")


def _split_md_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"): s = s[1:]
    if s.endswith("|"): s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator(line: str) -> bool:
    cells = _split_md_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-+:?", c) for c in cells)


def _render_md_inline_to_html(text: str) -> str:
    placeholders: list[str] = []

    def stash(html: str) -> str:
        placeholders.append(html)
        return f"\x00{len(placeholders) - 1}\x00"

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: stash(f'<a href="{m.group(2)}">{m.group(1)}</a>'),
        text,
    )
    text = re.sub(r"`([^`]+)`",
                  lambda m: stash(f"<code>{m.group(1)}</code>"), text)
    text = re.sub(r"(?<![A-Za-z0-9_])\*\*([^*]+)\*\*(?![A-Za-z0-9_])", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![A-Za-z0-9_*])_([^_]+)_(?![A-Za-z0-9_*])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![A-Za-z0-9_])\*([^*]+)\*(?![A-Za-z0-9_])", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], text)
    return text


def _table_to_html(headers: list[str], rows: list[list[str]],
                   is_metadata: bool = False) -> str:
    ncols = len(headers)
    n = len(rows)
    rowspans = [[0] * ncols for _ in range(n)]
    colspans = [[1] * ncols for _ in range(n)]
    suppressed = [[False] * ncols for _ in range(n)]

    has_explicit_rowspan = any(
        rows[r][c].strip() == "^" for r in range(n) for c in range(ncols)
    )
    for c in range(ncols):
        r = 0
        while r < n:
            base = r
            while r + 1 < n and (
                rows[r + 1][c].strip() == "^"
                or (
                    not has_explicit_rowspan
                    and not rows[r + 1][c].strip()
                    and rows[base][c].strip()
                )
            ):
                r += 1
                rowspans[base][c] += 1
                suppressed[r][c] = True
            r += 1

    for r in range(n):
        for c in range(ncols):
            if c == 0 or suppressed[r][c]:
                continue
            if not rows[r][c].strip():
                prev = c - 1
                while prev >= 0 and suppressed[r][prev]:
                    prev -= 1
                if prev >= 0:
                    colspans[r][prev] += 1
                    suppressed[r][c] = True

    table_class = "table opencv-rowspan-table"
    if is_metadata:
        table_class += " opencv-meta-table"
    out = [f'<div class="pst-scrollable-table-container"><table class="{table_class}">']
    if any(h.strip() for h in headers):
        out.append("<thead><tr>")
        for h in headers:
            out.append(f"<th>{_render_md_inline_to_html(h)}</th>")
        out.append("</tr></thead>")
    out.append("<tbody>")
    for r in range(n):
        out.append("<tr>")
        for c in range(ncols):
            if suppressed[r][c]:
                continue
            attrs = ""
            if rowspans[r][c]:
                attrs += f' rowspan="{rowspans[r][c] + 1}"'
            if colspans[r][c] > 1:
                attrs += f' colspan="{colspans[r][c]}"'
            out.append(f"<td{attrs}>{_render_md_inline_to_html(rows[r][c])}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


_META_KEYS = {
    "original author", "original authors", "author", "authors",
    "compatibility",
}


def _wrap_metadata_table(text: str) -> str:
    """If the first markdown table on the page looks like the
    `Original author / Compatibility` metadata block, wrap it in a div
    so CSS can shrink it to fit content."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("|"):
        if lines[i].lstrip().startswith("##"):
            return text
        i += 1
    if i >= len(lines):
        return text
    if i + 1 >= len(lines) or not _is_separator(lines[i + 1]):
        return text
    j = i + 2
    while j < len(lines) and lines[j].lstrip().startswith("|"):
        j += 1
    rows = [_split_md_row(ln.rstrip("\n")) for ln in lines[i + 2:j]]
    if not rows:
        return text
    keys = {(r[0].strip().lower() if r else "") for r in rows}
    if not (keys & _META_KEYS):
        return text
    block = "".join(lines[i:j])
    wrapped = ":::{div} opencv-meta-table\n\n" + block + "\n:::\n"
    return "".join(lines[:i]) + wrapped + "".join(lines[j:])


def _indent_orphan_code_fences(text: str) -> str:
    """Code fences at column 0 sandwiched between indented continuation lines
    or between list-item lines get indented so they stay inside the list."""

    list_re = re.compile(r"^(\s*)([0-9]+\.|[-*+])(\s+)")

    def line_indent(s: str) -> int:
        if not s.strip():
            return -1
        return len(s) - len(s.lstrip(" "))

    def list_continuation_indent(s: str) -> int:
        m = list_re.match(s)
        if not m:
            return -1
        return len(m.group(0))

    lines = text.splitlines(keepends=True)
    n = len(lines)
    i = 0
    while i < n:
        ln_full = lines[i]
        stripped = ln_full.lstrip(" ")
        cur_fence_indent = len(ln_full) - len(stripped)
        if stripped.startswith("```"):
            j = i + 1
            while j < n and not lines[j].rstrip("\n").lstrip(" ").startswith("```"):
                j += 1
            if j >= n:
                break
            p = i - 1
            while p >= 0 and not lines[p].strip():
                p -= 1
            nxt = j + 1
            while nxt < n and not lines[nxt].strip():
                nxt += 1

            prev_line = lines[p] if p >= 0 else ""
            next_line = lines[nxt] if nxt < n else ""

            prev_target = list_continuation_indent(prev_line)
            if prev_target < 0:
                prev_target = line_indent(prev_line) if prev_line.strip() else -1
            next_target = list_continuation_indent(next_line)
            if next_target < 0:
                next_target = line_indent(next_line) if next_line.strip() else -1

            candidates = [x for x in (prev_target, next_target) if x >= 0]
            target = min(candidates) if candidates else cur_fence_indent

            if target != cur_fence_indent:
                if target > cur_fence_indent:
                    pad = " " * (target - cur_fence_indent)
                    for k in range(i, j + 1):
                        if lines[k].strip():
                            lines[k] = pad + lines[k]
                else:
                    strip_n = cur_fence_indent - target
                    for k in range(i, j + 1):
                        if lines[k].startswith(" " * strip_n):
                            lines[k] = lines[k][strip_n:]
            i = j + 1
        else:
            i += 1
    return "".join(lines)


def _fix_overindented_sub_bullets(text: str) -> str:
    """Normalize over-indented sub-bullets within col-0 -   list items.

    In Doxygen source, sub-bullets inside a -   parent item are sometimes
    at col 8 or col 12 instead of col 4. CommonMark treats col 4+4=8+
    content as an indented code block within the parent item body, so the
    sub-bullets disappear. This pass de-indents them to col 4 and adjusts
    their continuation lines by the same amount.
    """
    lines = text.splitlines(keepends=False)
    _parent_re = re.compile(r"^[-*+]   ")
    _bullet_re = re.compile(r"^(\s+)([-*+])\s+")
    in_fence = False
    in_parent = False
    cur_excess = 0
    cur_bullet_col = 0
    out: list[str] = []

    for ln in lines:
        s = ln.lstrip()
        if s.startswith("```"):
            in_fence = not in_fence
            out.append(ln)
            continue

        if in_fence:
            out.append(ln)
            continue

        if _parent_re.match(ln):
            in_parent = True
            cur_excess = 0
            out.append(ln)
            continue

        if not in_parent:
            out.append(ln)
            continue

        if not ln.strip():
            out.append(ln)
            continue

        col = len(ln) - len(ln.lstrip())

        if col == 0:
            in_parent = False
            cur_excess = 0
            out.append(ln)
            continue

        m = _bullet_re.match(ln)
        if m:
            bullet_col = len(m.group(1))
            if bullet_col > 4:
                cur_excess = bullet_col - 4
                cur_bullet_col = bullet_col
                out.append(ln[cur_excess:])
            else:
                cur_excess = 0
                out.append(ln)
        elif cur_excess > 0 and col >= cur_bullet_col:
            out.append(ln[cur_excess:])
        else:
            if col < cur_bullet_col:
                cur_excess = 0
            out.append(ln)

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _deindent_toplevel_numbered_lists(text: str) -> str:
    """Strip 4-space indent from numbered list items that follow a plain paragraph.

    Doxygen allows `    1. item` after a paragraph; CommonMark treats it as an
    indented code block. Only de-indent when the numbered list is at the top
    level (not inside a list item). Determined by scanning backwards past other
    4-space-indented numbered items to find the anchor line: if the anchor is at
    column 0 and is not a list bullet, we are at the top level and should
    de-indent.
    """
    lines = text.splitlines(keepends=False)
    _list_bullet = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]")
    _indented_num = re.compile(r"^    (\d+\.)( .+)$")
    _indented_num_bare = re.compile(r"^    \d+\.")

    def _anchor(idx: int) -> str:
        j = idx - 1
        while j >= 0:
            ln = lines[j]
            if not ln.strip():
                j -= 1
                continue
            if _indented_num_bare.match(ln):
                j -= 1
                continue
            return ln
        return ""

    _indented_prose = re.compile(r"^    (\S.*)$")

    in_fence = False
    out: list[str] = []
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            m = _indented_num.match(ln)
            if m:
                anchor = _anchor(i)
                anchor_indent = len(anchor) - len(anchor.lstrip())
                if anchor_indent == 0 and not _list_bullet.match(anchor):
                    ln = m.group(1) + m.group(2)
            else:
                mp = _indented_prose.match(ln)
                if mp:
                    anchor = _anchor(i)
                    if anchor.rstrip() == "::::":
                        ln = mp.group(1)
        out.append(ln)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _ensure_blank_lines_around_images(text: str) -> str:
    """Ensure standalone image lines have blank lines before and after them.

    Without surrounding blank lines, Sphinx renders the image inline within
    the surrounding text paragraph instead of as a block element.
    """
    lines = text.splitlines(keepends=False)
    _img_line = re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$")
    in_fence = False
    out: list[str] = []
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and _img_line.match(ln):
            prev = out[-1].strip() if out else ""
            if prev:
                out.append("")
            out.append(ln)
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt:
                out.append("")
        else:
            out.append(ln)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _ensure_blank_lines_around_display_math(text: str) -> str:
    """Ensure every $$ display-math delimiter has a blank line before and after it.

    Without surrounding blank lines, MyST can misparse inline $...$ on adjacent
    lines as display math, causing wide spacing around math variables.
    """
    lines = text.splitlines(keepends=False)
    out: list[str] = []
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped == "$$":
            prev = out[-1].strip() if out else ""
            if prev:
                out.append("")
            out.append(ln)
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt:
                out.append("")
        else:
            out.append(ln)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _convert_images_to_figures(text: str) -> str:
    out: list[str] = []
    pos = 0
    in_fence = False
    safe = []
    for ln in text.splitlines(keepends=True):
        stripped = ln.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            safe.append(ln)
            continue
        if in_fence:
            safe.append(ln)
            continue
        m = re.match(r"^([ \t]*)!\[(.*?)\]\(([^)]+)\)\s*$", ln)
        if m and m.group(2).strip():
            indent, alt, url = m.group(1), m.group(2), m.group(3)
            safe.append(f"{indent}```{{figure}} {url}\n")
            safe.append(f"{indent}:alt: {alt}\n\n")
            safe.append(f"{indent}{alt}\n")
            safe.append(f"{indent}```\n")
        else:
            safe.append(ln)
    return "".join(safe)


def _convert_rowspan_tables(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            header_line = lines[i].rstrip("\n")
            j = i + 2
            data_lines: list[str] = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                data_lines.append(lines[j].rstrip("\n"))
                j += 1
            headers = _split_md_row(header_line)
            rows = [_split_md_row(dl) for dl in data_lines]
            rows = [r + [""] * (len(headers) - len(r)) for r in rows]
            has_rowspan = any(any(c.strip() == "^" for c in r) for r in rows)
            has_middle_empty = any(
                not r[c].strip() and c > 0 and c < len(r) - 1
                for r in rows for c in range(len(r))
            )
            has_implicit_rowspan = False
            last_col = len(headers) - 1
            for c in range(last_col):
                consecutive_empty = 0
                seen_content = False
                for row in rows:
                    if row[c].strip():
                        if consecutive_empty >= 1:
                            has_implicit_rowspan = True
                            break
                        consecutive_empty = 0
                        seen_content = True
                    elif seen_content:
                        consecutive_empty += 1
                if seen_content and consecutive_empty >= 1:
                    has_implicit_rowspan = True
                if has_implicit_rowspan:
                    break
            if has_rowspan or has_middle_empty or has_implicit_rowspan:
                keys = {(r[0].strip().lower() if r else "") for r in rows}
                is_meta = bool(keys & _META_KEYS)
                out.append("\n```{raw} html\n")
                out.append(_table_to_html(headers, rows, is_metadata=is_meta))
                out.append("\n```\n")
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "".join(out)


_MODULE_BULLET_RE = re.compile(
    r"^[ \t]*-[ \t]+\[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)[ \t]+"
    r"\(\*\*(?P<name>[^*]+)\*\*\)[ \t]*-[ \t]*(?P<desc>.+?)[ \t]*$",
    re.M,
)


def _convert_module_bullets_to_table(text: str) -> str:
    """Detect runs of `- [Title](url) (**name**) - description` bullets and
    convert each run to an opencv-module-table list-table."""

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _MODULE_BULLET_RE.match(lines[i].rstrip("\n"))
        if not m:
            out.append(lines[i])
            i += 1
            continue
        rows: list[tuple[str, str, str, str]] = []
        while i < len(lines):
            ln = lines[i].rstrip("\n")
            mm = _MODULE_BULLET_RE.match(ln)
            if mm:
                rows.append((mm.group("title"), mm.group("url"),
                             mm.group("name"), mm.group("desc")))
                i += 1
                while i < len(lines):
                    cont = lines[i]
                    if cont.startswith("    ") and cont.strip():
                        t, u, n, d = rows[-1]
                        rows[-1] = (t, u, n, (d + " " + cont.strip()).strip())
                        i += 1
                    else:
                        break
                continue
            tail = re.match(r"^[ \t]*-[ \t]+(\.\.\..+)$", ln)
            if tail and rows:
                rows.append(("", "", "", tail.group(1).strip()))
                i += 1
                while i < len(lines):
                    cont = lines[i]
                    if cont.startswith("    ") and cont.strip():
                        t, u, n, d = rows[-1]
                        rows[-1] = (t, u, n, (d + " " + cont.strip()).strip())
                        i += 1
                    else:
                        break
                continue
            break
        if len(rows) < 2:
            for t, u, n, d in rows:
                if u:
                    out.append(f"- [{t}]({u}) (**{n}**) - {d}\n")
                else:
                    out.append(f"- {d}\n")
            continue
        out.append("```{list-table}\n")
        out.append(":class: opencv-module-table\n")
        out.append(":widths: 22 78\n")
        out.append(":header-rows: 1\n\n")
        out.append("* - Module\n  - Description\n\n")
        for t, u, n, d in rows:
            if u:
                out.append(f"* - [`{n}`]({u})\n  - {t} — {d}\n")
            else:
                out.append(f"* - \n  - {d}\n")
        out.append("```\n")
    return "".join(out)


_PROTECT_RE = re.compile(
    r"(?P<fence>```.*?```)"
    r"|(?P<dmath>\$\$.*?\$\$)"
    r"|(?P<imath>(?<!\\)\$[^\n$]+?\$)"
    r"|(?P<code>`[^`\n]+`)"
    r"|(?P<link>\[[^\]]+\]\([^)]+\))"
    r"|(?P<image>!\[[^\]]*\]\([^)]+\))",
    re.S,
)

_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")

_CV_SYMBOL_RE = re.compile(r"(?<!%)\bcv::[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*\b")
_HASH_REF_RE = re.compile(r"(?<![A-Za-z0-9_%])#([A-Z][A-Za-z0-9_]*)")
_CV_MACRO_RE = re.compile(r"(?<!%)\b(CV_[A-Z0-9][A-Z0-9_]*)\b")
_NOLINK_PREFIX_RE = re.compile(r"%(?=(?:cv::[A-Za-z_]|CV_[A-Z0-9]|#?[A-Z][A-Za-z0-9_]*))")


def _autolink_cv_symbols(text: str, tags: TagIndex) -> str:
    out: list[str] = []
    pos = 0
    for m in _PROTECT_RE.finditer(text):
        out.append(_link_segment(text[pos:m.start()], tags))
        chunk = m.group(0)
        if m.lastgroup == "code":
            chunk = _maybe_unwrap_code_to_link(chunk, tags)
        out.append(chunk)
        pos = m.end()
    out.append(_link_segment(text[pos:], tags))
    return "".join(out)


def _maybe_unwrap_code_to_link(code_span: str, tags: TagIndex) -> str:
    inner = code_span[1:-1]
    m = re.fullmatch(r"#([A-Z0-9][A-Za-z0-9_]*)(?:<[^>]*>)?(?:\([^)]*\))?", inner)
    target = m.group(1) if m else None
    if target is None:
        m = re.fullmatch(
            r"(cv::[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)"
            r"(?:<[^>]*>)?(?:\([^)]*\))?",
            inner)
        target = m.group(1) if m else None
    if not target:
        return code_span
    result = tags.lookup(target)
    if result is None:
        return code_span
    url, _ = result
    return f"[{inner}]({url})"


_EXISTING_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


def _spans_in(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _EXISTING_LINK_RE.finditer(text)]


def _link_segment(segment: str, tags: TagIndex) -> str:
    def make_repl(group_idx: int):
        def repl(m: re.Match) -> str:
            spans = _spans_in(segment)
            if any(s <= m.start() < e for s, e in spans):
                return m.group(0)
            name = m.group(group_idx)
            result = tags.lookup(name)
            if result is None:
                return m.group(0)
            url, _ = result
            return f"[{name}]({url})"
        return repl

    segment = _CV_SYMBOL_RE.sub(
        lambda m: (
            m.group(0) if any(s <= m.start() < e for s, e in _spans_in(segment))
            else (lambda r: f"[{m.group(0)}]({r[0]})" if r else m.group(0))(tags.lookup(m.group(0)))
        ),
        segment,
    )
    segment = _HASH_REF_RE.sub(make_repl(1), segment)
    segment = _CV_MACRO_RE.sub(make_repl(1), segment)
    segment = _NOLINK_PREFIX_RE.sub("", segment)
    return segment

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--tag", type=Path, default=Path("/tmp/opencv.tag"))
    ap.add_argument("--local", type=Path, default=None)
    ap.add_argument("--out-doc", default="")
    args = ap.parse_args()

    if not args.tag.exists():
        print(f"Tagfile not found: {args.tag}\n"
              "Download with: curl -sSL -o /tmp/opencv.tag "
              "https://docs.opencv.org/5.x/opencv.tag", file=sys.stderr)
        return 2

    local_refs: dict[str, str] = {}
    if args.local and args.local.exists():
        local_refs = json.loads(args.local.read_text(encoding="utf-8"))

    tags = TagIndex(args.tag)
    src = args.input.read_text(encoding="utf-8")
    out = transform(src, tags, local_refs=local_refs, out_doc=args.out_doc)

    if args.output.exists():
        existing = args.output.read_text(encoding="utf-8")
        m = re.match(r"^(---\n.*?\n---\n)\s*", existing, re.S)
        if m:
            out = m.group(1) + "\n" + out

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out, encoding="utf-8")
    print(f"Wrote {args.output} ({len(out)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
