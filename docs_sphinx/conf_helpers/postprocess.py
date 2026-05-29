"""build-finished hook: inline collaboration-diagram SVGs.

After the HTML is written, replace each collaboration-diagram ``<img>`` with
the SVG inlined into the page so its internal class links become clickable and
resolve against the local Sphinx site. conf.py connects
``_inline_coll_graphs_on_finish`` to Sphinx's ``build-finished`` event.
"""
from __future__ import annotations
import pathlib, re

from .state import DOXYGEN_BASE_URL


def _inline_collaboration_svgs(api_dir: pathlib.Path,
                              image_dir: pathlib.Path) -> None:
    """After the HTML is written, replace each collaboration-diagram `<img>`
    with the SVG inlined into the page.

    Why: an SVG embedded via `<img>` is rendered as a flat picture — its
    internal `<a>` links are dead. Inlining the `<svg>` makes those links
    clickable, and (being inside the page) they resolve against our Sphinx
    site. Each `xlink:href` is rewritten from the legacy Doxygen path
    (`../../d6/d50/classcv_1_1Size__.html`) to the matching Sphinx class page
    when we generate one (`classcv_1_1Size__.html`, same `api/` dir), else to
    the upstream docs.opencv.org page so the link still resolves.

    Connected to `build-finished`, so it runs once the output exists. It is
    idempotent: a page whose diagram is already inlined has no `<img>` left to
    match, so a re-run (incremental build) is a no-op."""
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
        if "://" in path:                       # already absolute/external
            return m.group(0)
        base = path.rsplit("/", 1)[-1]
        if (api_dir / base).is_file():           # we generate this class page
            return f'xlink:href="{base}"'
        rel = path.lstrip("./")                  # strip leading ../ and ./
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
            start = svg.find("<svg")             # drop xml decl / doctype / comments
            if start < 0:
                return m.group(0)
            svg = href_re.sub(_rewrite_href, svg[start:])
            # Carry the img's theme classes + alt onto the inline <svg> so the
            # light/dark swap still works and it stays accessible.
            return svg.replace(
                "<svg ",
                f'<svg class="{m.group("cls")}" role="img" '
                f'aria-label="{m.group("alt")}" ', 1)

        new = img_re.sub(_inline, text)
        if new != text:
            html.write_text(new, encoding="utf-8")


def _inline_coll_graphs_on_finish(app, exception):
    if exception is not None:
        return
    out = pathlib.Path(app.outdir)
    _inline_collaboration_svgs(out / "api", out / "_images")
