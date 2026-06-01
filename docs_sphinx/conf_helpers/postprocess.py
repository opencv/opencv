"""build-finished hook: inline coll-diagram SVGs, strip Breathe clutter."""
from __future__ import annotations
import pathlib, re

from .state import DOXYGEN_BASE_URL


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
        if (api_dir / base).is_file():
            return f'xlink:href="{base}"'
        rel = path.lstrip("./")
        return f'xlink:href="{DOXYGEN_BASE_URL}{rel}"'

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


def _inline_coll_graphs_on_finish(app, exception):
    """build-finished entry point."""
    if exception is not None:
        return
    out = pathlib.Path(app.outdir)
    for _api in ("main_modules", "extra_modules"):
        _inline_collaboration_svgs(out / _api, out / "_images")
        _strip_breathe_class_clutter(out / _api)
    _generate_search_map(out)
