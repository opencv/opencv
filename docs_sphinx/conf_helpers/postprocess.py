# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""build-finished hook: inline coll-diagram SVGs, strip Breathe clutter, re-theme Doxygen."""
from __future__ import annotations
import pathlib, re

from .state import (_doxy_page_to_local, _DOXY_ANCHOR_TO_MEMBER, DOXYGEN_BASE_URL,
                    _API_XML_DIR)


def _doxy_parent_page(page: str, api_dir: pathlib.Path) -> str:
    """Nested types (e.g. `structcv_1_1SparseMat_1_1Hdr`) get no standalone
    Sphinx page — they're documented inline on the enclosing class. Walk up the
    `_1_1`-separated scope to the nearest ancestor page that DOES exist locally.
    Returns "" if no ancestor page exists."""
    stem = page[:-5] if page.endswith(".html") else page
    rest = None
    for pref in ("class", "struct", "union"):
        if stem.startswith(pref):
            rest = stem[len(pref):]
            break
    if rest is None:
        return ""
    while "_1_1" in rest:
        rest = rest.rsplit("_1_1", 1)[0]
        for pref in ("class", "struct", "union"):
            cand = f"{pref}{rest}.html"
            if (api_dir / cand).is_file():
                return cand
    return ""


def _inline_collaboration_svgs(api_dir: pathlib.Path,
                              image_dir: pathlib.Path) -> None:
    """Inline coll-diagram SVGs so their links work; idempotent."""
    import re
    if not api_dir.is_dir():
        return
    img_re = re.compile(
        r'<img alt="(?P<alt>[^"]*)" '
        r'class="(?P<cls>opencv-coll-graph[^"]*)" '
        r'src="\.\./_images/(?P<file>[^"]+\.svg)"\s*/?>')
    href_re = re.compile(r'xlink:href="(?P<path>[^"]+)"')

    def _rewrite_href(m: "re.Match") -> str:
        path = m.group("path")
        if "://" in path:
            return m.group(0)
        base = path.rsplit("/", 1)[-1]
        page, _, frag = base.partition("#")
        local = _doxy_page_to_local(page)
        if not (api_dir / local).is_file():
            # Nested type with no own page -> enclosing class page (inline docs).
            parent = _doxy_parent_page(page, api_dir)
            if parent:
                local = parent
        member = _DOXY_ANCHOR_TO_MEMBER.get(frag) if frag else None
        if member:
            return f'xlink:href="{local}#{member}"'
        return f'xlink:href="{local}"'

    for html in api_dir.glob("*.html"):
        text = html.read_text(encoding="utf-8")
        if "opencv-coll-graph" not in text:
            continue

        def _inline(m: "re.Match") -> str:
            svg_path = image_dir / m.group("file")
            if not svg_path.is_file():
                return m.group(0)
            svg = svg_path.read_text(encoding="utf-8")
            start = svg.find("<svg")
            if start < 0:
                return m.group(0)
            svg = href_re.sub(_rewrite_href, svg[start:])
            # carry theme classes + alt for dark mode / a11y
            return svg.replace(
                "<svg ",
                f'<svg class="{m.group("cls")}" role="img" '
                f'aria-label="{m.group("alt")}" ', 1)

        new = img_re.sub(_inline, text)
        if new != text:
            html.write_text(new, encoding="utf-8")


def _strip_breathe_class_clutter(api_dir: pathlib.Path) -> None:
    """Drop Breathe's duplicate class signature header; idempotent."""
    import re
    if not api_dir.is_dir():
        return
    section_re = re.compile(
        r'(<section id="detailed-description"[^>]*>)'
        r'(?P<body>[\s\S]*?)'
        r'(</section>)'
    )
    dl_re = re.compile(
        r'<dl[^>]*\bclass="[^"]*\bclass\b[^"]*"[^>]*>\s*'
        r'<dt[^>]*>[\s\S]*?</dt>\s*'
        r'<dd>(?P<dd>[\s\S]*?)</dd>\s*'
        r'</dl>'
    )
    subclassed_re = re.compile(r'<p>Subclassed by[\s\S]*?</p>\s*')

    for h in api_dir.glob("classcv*.html"):
        text = h.read_text(encoding="utf-8")
        if "detailed-description" not in text:
            continue

        def _strip_section(sm):
            head, body, tail = sm.group(1), sm.group("body"), sm.group(3)

            def _strip_dl(dm):
                dd_body = dm.group("dd").strip()
                dd_body = subclassed_re.sub("", dd_body).strip()
                return dd_body

            new_body = dl_re.sub(_strip_dl, body, count=1)
            return head + new_body + tail

        new = section_re.sub(_strip_section, text, count=1)
        if new != text:
            h.write_text(new, encoding="utf-8")


def _generate_search_map(out_dir: pathlib.Path) -> None:
    """Write _static/search_map.js: stem→Sphinx-path for every built HTML page."""
    import json
    skip = {"_static", "_sources", "_images", "_sphinx_design_static"}
    mapping = {}
    for f in out_dir.rglob("*.html"):
        rel = f.relative_to(out_dir)
        if rel.parts[0] in skip:
            continue
        mapping[f.stem] = rel.as_posix()
    lines = ["var sphinxPageMap = {"]
    for k, v in sorted(mapping.items()):
        lines.append(f"  {json.dumps(k)}: {json.dumps(v)},")
    lines.append("};")
    (out_dir / "_static" / "search_map.js").write_text("\n".join(lines), encoding="utf-8")

_SYM = r"(?:group__|classcv|structcv|unioncv|namespacecv)\w*\.html"

def _localize_doxygen_links(out_dir: pathlib.Path) -> None:
    """Point symbol-page links at the local Sphinx page when we built it:
    docs.opencv.org URLs, and bare relative names that 404 off the API dir."""
    import os
    skip = {"_static", "_sources", "_images", "_sphinx_design_static"}
    page_paths: dict[str, str] = {}
    for f in out_dir.rglob("*.html"):
        rel = f.relative_to(out_dir)
        if rel.parts and rel.parts[0] in skip:
            continue
        page_paths.setdefault(f.name, rel.as_posix())
    ext_re = re.compile(
        r'(?P<a><a class="reference external"\s+)?'
        r'href="' + re.escape(DOXYGEN_BASE_URL) + r'(?:[\w-]+/)*?'
        r'(?P<page>' + _SYM + r')(?:#(?P<frag>\w+))?"')
    bare_re = re.compile(r'href="(?P<page>' + _SYM + r')(?:#(?P<frag>[\w:.-]+))?"')

    for html in out_dir.rglob("*.html"):
        rel = html.relative_to(out_dir)
        if rel.parts and rel.parts[0] in skip:
            continue
        text = html.read_text(encoding="utf-8")
        cur = rel.parent.as_posix()

        def _href(local: str, anchor: str) -> str | None:
            target = page_paths.get(local)
            if not target:
                return None
            href = target if cur == "." else os.path.relpath(target, cur)
            return f'href="{href}{anchor}"'

        def _ext(m: "re.Match") -> str:
            member = _DOXY_ANCHOR_TO_MEMBER.get(m.group("frag") or "")
            h = _href(_doxy_page_to_local(m.group("page")),
                      f"#{member}" if member else "")
            if not h:
                return m.group(0)
            # Flip the now-local link from external to internal styling.
            return f'<a class="reference internal" {h}' if m.group("a") else h

        def _bare(m: "re.Match") -> str:
            frag = m.group("frag")
            return _href(m.group("page"), f"#{frag}" if frag else "") or m.group(0)

        new = bare_re.sub(_bare, ext_re.sub(_ext, text))
        if new != text:
            html.write_text(new, encoding="utf-8")


def _drop_moved_stub_search_entries() -> None:
    """Drop moved-tutorial stub pages from the Doxygen search index."""
    doxy_html = _API_XML_DIR.parent / "html"
    search_dir = doxy_html / "search"
    if not search_dir.is_dir():
        return
    stubs = set()
    for h in doxy_html.rglob("*table_of_content*.html"):
        try:
            if "has been moved to this page" in h.read_text(encoding="utf-8", errors="ignore"):
                stubs.add(h.name)
        except OSError:
            pass
    if not stubs:
        return
    for js in search_dir.glob("*.js"):
        lines = js.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        # Drop only single-target stub entries; keep multi-target terms.
        kept = [ln for ln in lines if not (
                ln.count("['../") == 1 and any(f"/{s}'" in ln for s in stubs))]
        if len(kept) != len(lines):
            js.write_text("".join(kept), encoding="utf-8")


def _inline_coll_graphs_on_finish(app, exception):
    """build-finished entry point."""
    if exception is not None:
        return
    out = pathlib.Path(app.outdir)
    for _api in ("main_modules", "extra_modules"):
        _inline_collaboration_svgs(out / _api, out / "_images")
        _strip_breathe_class_clutter(out / _api)
    _localize_doxygen_links(out)
    _drop_moved_stub_search_entries()
    _generate_search_map(out)
