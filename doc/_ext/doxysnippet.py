"""Sphinx directive that mimics Doxygen ``@snippet``.

Reads a source file, extracts the block delimited by ``//! [tag]`` /
``# [tag]`` markers, and renders it as a code block. Paths resolve
relative to the OpenCV repo root (parent of the docs project).
"""

from __future__ import annotations

import re
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective


_MARKER_RE_TEMPLATE = (
    r"^\s*(?:"
    r"//!|//|#+|"                    # C/C++/Python/shell line comments
    r"<!--\s*(?://!?\s*)?|"          # HTML/XML comment, optionally prefixed with //! (e.g. <!-- //! [tag] -->)
    r"\*\s*|/\*\s*"                  # C block-comment styles
    r")\s*\[{tag}\]"
    r"\s*(?:-->|\*/)?\s*$"           # optional comment closers
)


def _repo_root(app) -> Path:
    return Path(app.confdir).parent


def _resolve_source(app, rel: str) -> Path | None:
    repo = _repo_root(app)
    direct = repo / rel
    if direct.exists():
        return direct
    rel_parts = Path(rel).parts  # e.g. ('android', 'mobilenet-objdetect', 'gradle', 'AndroidManifest.xml')
    name = rel_parts[-1]
    search_roots = getattr(app.config, "doxysnippet_search_paths", None) or [
        "samples", "apps", "modules", "doc/tutorials", "doc",
    ]
    best: Path | None = None
    best_score = -1
    for root in search_roots:
        for hit in (repo / root).rglob(name):
            if not hit.is_file():
                continue
            # Score = number of rel_parts components that appear as a suffix of hit.parts
            hit_parts = hit.parts
            score = 0
            for k in range(1, len(rel_parts) + 1):
                if hit_parts[-k:] == rel_parts[-k:]:
                    score = k
                else:
                    break
            if score > best_score:
                best_score = score
                best = hit
    return best


class DoxySnippetDirective(SphinxDirective):
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        "tag": directives.unchanged_required,
        "language": directives.unchanged,
        "dedent": directives.nonnegative_int,
    }

    def run(self):
        rel = self.arguments[0].strip()
        tag = self.options["tag"].strip()
        lang = self.options.get("language", "cpp")
        dedent = self.options.get("dedent")

        src = _resolve_source(self.env.app, rel)
        if src is None:
            return [self.state.document.reporter.warning(
                f"doxysnippet: file not found: {rel}", line=self.lineno)]

        text = src.read_text(encoding="utf-8", errors="replace").splitlines()
        marker = re.compile(_MARKER_RE_TEMPLATE.format(tag=re.escape(tag)))
        positions = [i for i, ln in enumerate(text) if marker.match(ln)]
        if len(positions) < 2:
            return [self.state.document.reporter.warning(
                f"doxysnippet: tag [{tag}] not found (need 2 markers) in {rel}",
                line=self.lineno)]
        start, end = positions[0] + 1, positions[1]
        block = text[start:end]

        while block and not block[0].strip():
            block.pop(0)
        while block and not block[-1].strip():
            block.pop()

        if dedent:
            block = [ln[dedent:] if len(ln) >= dedent else ln for ln in block]
        else:
            indents = [
                len(ln) - len(ln.lstrip(" \t"))
                for ln in block if ln.strip()
            ]
            common = min(indents) if indents else 0
            if common:
                block = [ln[common:] if len(ln) >= common else ln for ln in block]

        literal = nodes.literal_block("\n".join(block), "\n".join(block))
        literal["language"] = lang
        literal["classes"].append("doxysnippet")
        literal["source"] = str(src)
        return [literal]


_LANG_BY_EXT = {
    ".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".cxx": "cpp", ".cc": "cpp",
    ".py": "python", ".js": "javascript", ".java": "java",
    ".sh": "bash", ".bash": "bash", ".cmake": "cmake",
    ".cmd": "bat", ".bat": "bat", ".xml": "xml",
    ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
}


class DoxyIncludeDirective(SphinxDirective):
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        "language": directives.unchanged,
        "dedent": directives.nonnegative_int,
    }

    def run(self):
        rel = self.arguments[0].strip()
        src = _resolve_source(self.env.app, rel)
        if src is None:
            return [self.state.document.reporter.warning(
                f"doxyinclude: file not found: {rel}", line=self.lineno)]
        text = src.read_text(encoding="utf-8", errors="replace")
        lang = self.options.get("language") or _LANG_BY_EXT.get(
            Path(rel).suffix.lower(), "cpp"
        )
        literal = nodes.literal_block(text, text)
        literal["language"] = lang
        literal["classes"].append("doxyinclude")
        literal["source"] = str(src)
        return [literal]


def setup(app):
    app.add_directive("doxysnippet", DoxySnippetDirective)
    app.add_directive("doxyinclude", DoxyIncludeDirective)
    app.add_config_value("doxysnippet_search_paths", None, "html")
    return {"version": "0.2", "parallel_read_safe": True}
