# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""build-finished hook: inline coll-diagram SVGs, strip Breathe clutter, re-theme Doxygen."""
from __future__ import annotations
import pathlib, re

from .state import (_doxy_page_to_local, _DOXY_ANCHOR_TO_MEMBER, DOXYGEN_BASE_URL,
                    _LOCAL_CLASS_URL, _LOCAL_TYPEDEF_URL, _FILE_URL, _API_XML_DIR, DOC_ROOT,
                    _CV_SYMBOL_URL)


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
            svg = svg.replace(
                "<svg ",
                f'<svg class="{m.group("cls")}" role="img" '
                f'aria-label="{m.group("alt")}" ', 1)
            # Wrap in a scroll box so large graphs (e.g. file include graphs)
            # stay fully reachable instead of being clipped — scrolls when wider
            # than the content area, fits otherwise.
            return f'<div class="opencv-graph-scroll">{svg}</div>'

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

def _fix_gapi_images(out_dir: pathlib.Path) -> None:
    """Copy gapi doc images to _images/ and fix src paths in gapi.html."""
    gapi_html = out_dir / "extra_modules" / "gapi.html"
    if not gapi_html.is_file():
        return
    images_dir = out_dir / "_images"
    images_dir.mkdir(exist_ok=True)

    text = gapi_html.read_text(encoding="utf-8")

    def _fix_src(m):
        raw_path = m.group(1)
        if "contrib_modules" not in raw_path:
            return m.group(0)
        # raw_path is relative to extra_modules/ in the browser,
        # but the file lives at out_dir / raw_path (no extra_modules prefix)
        src_file = out_dir / raw_path
        if not src_file.is_file():
            return m.group(0)
        dest = images_dir / src_file.name
        if not dest.is_file():
            import shutil as _shutil
            _shutil.copy2(src_file, dest)
        return f'src="../_images/{src_file.name}"'

    new_text = re.sub(r'src="([^"]+)"', _fix_src, text)
    if new_text != text:
        gapi_html.write_text(new_text, encoding="utf-8")

def _copy_js_tryit_files(out_dir: pathlib.Path) -> None:
    """Copy js_*.html Try-it pages + assets so iframe src="../../js_*.html" resolves."""
    import shutil, os
    js_assets = DOC_ROOT / "js_tutorials" / "js_assets"
    dest = out_dir / "js_tutorials"
    if not js_assets.is_dir() or not dest.is_dir():
        return
    for src in js_assets.iterdir():
        if src.is_file():
            dst = dest / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
    # opencv.js from CMake (OPENCV_JS_PATH). Needed both alongside the Try-it
    # pages (relative `src="opencv.js"`) AND at the doc-site root, where
    # external/js_usage docs link https://docs.opencv.org/<ver>/opencv.js.
    opencv_js = os.environ.get("OPENCV_JS_PATH", "")
    if opencv_js and pathlib.Path(opencv_js).is_file():
        for dst in (dest / "opencv.js", dest.parent / "opencv.js"):
            if not dst.exists():
                shutil.copy2(opencv_js, dst)
    # Extra assets referenced by Try-it pages but not in js_assets/.
    _opencv_root = DOC_ROOT.parent
    for _name, _src in {
        "box.mp4":           _opencv_root / "samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box.mp4",
        "space_shuttle.jpg": DOC_ROOT / "tutorials/dnn/images/space_shuttle.jpg",
        "roi.jpg":           DOC_ROOT / "py_tutorials/py_core/py_basic_ops/images/roi.jpg",
    }.items():
        if _src.is_file() and not (dest / _name).exists():
            shutil.copy2(_src, dest / _name)


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
    # `#include` links -> local file-reference page (same Doxygen file stem).
    inc_re = re.compile(
        r'<a class="reference external opencv-include-link" '
        r'href="[^"]*doc/doxygen/html/[^"]*?(?P<file>[^/"]+\.html)">(?P<path>[^<]+)</a>')
    strip_re = re.compile(
        r'<a\b[^>]*\bhref="(?:' + re.escape(DOXYGEN_BASE_URL) + r'|[^"]*doc/doxygen/html/)'
        r'(?![^"]*javadoc)[^"]*"[^>]*>(?P<txt>[^<]*)</a>')

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

        def _inc(m: "re.Match") -> str:
            h = _href(m.group("file"), "")
            if not h:
                return m.group(0)
            return f'<a class="reference internal opencv-include-link" {h}>{m.group("path")}</a>'

        new = inc_re.sub(_inc, bare_re.sub(_bare, ext_re.sub(_ext, text)))
        new = strip_re.sub(lambda m: m.group("txt"), new)
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


# Pygments emits `<span class="n">NAME</span>` (and `class="nc"`/`"nf"`
# for class/function tokens) inside its rendered `<pre>` for every C++
# identifier in a code block. The example pages, snippet pages, tutorial
# samples, and any `:::{code-block} cpp` fence all go through Pygments,
# so by the time the HTML is written the code blocks have full syntax
# colouring but ZERO clickable tokens — `Mat`, `InputArray`,
# `getOptimalDFTSize`, etc. are inert text.
#
# This pass wraps each such span in an `<a class="reference internal"
# href="…">` when the token resolves via `_LOCAL_CLASS_URL` /
# `_LOCAL_TYPEDEF_URL`, mirroring what the API-stub renderer already
# does for inline `<programlisting>`. Idempotent: skips spans already
# inside an `<a>`.
_PYG_IDENT_SPAN_RE = re.compile(
    r'(?P<prefix><span class="(?:n|nc|nf|nb|nv|na)">)'
    r'(?P<name>[A-Za-z_][A-Za-z0-9_]*)'
    r'(?P<suffix></span>)'
)

# Pygments preprocessor-file-name span:
#   `<span class="cpf">&quot;opencv2/core.hpp&quot;</span>`  (or &lt;…&gt;)
# The quote characters are HTML-escaped (`&quot;` / `&lt;` / `&gt;`).
# Capture the inner path so we can wrap it in an `<a>` linking to the
# local Doxygen file page (`_FILE_URL` map). The quote chars stay
# outside the anchor — mirrors how the enum-detail `#include` line is
# rendered (`#include <a href="…">opencv2/core.hpp</a>`).
_PYG_CPF_SPAN_RE = re.compile(
    r'(?P<prefix><span class="cpf">)'
    r'(?P<openq>&quot;|&lt;)'
    r'(?P<path>[A-Za-z0-9_./+\-]+\.[A-Za-z0-9]+)'
    r'(?P<closeq>&quot;|&gt;)'
    r'(?P<suffix></span>)'
)


# C++ free-function linkifier. The `(` lookahead means only call sites match,
# never a like-named local (`log`, `min`, …); it also makes the pass idempotent
# and class-safe (a wrapped/constructor name is followed by `</a>`, not `(`).
_PYG_CPP_PRE_RE = re.compile(
    r'(?P<open><div class="highlight-cpp[^"]*"><div class="highlight"><pre>)'
    r'(?P<body>.*?)(?P<close></pre>)', re.DOTALL)
_PYG_CALL_SPAN_RE = re.compile(
    r'(?P<span><span class="n">)(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<end></span>)'
    r'(?=<span class="p">\()'
)
# Python free-function calls. Requires a `cv.`/`cv2.` prefix (OpenCV's Python
# API is always module-qualified) so bare builtins / other modules never link.
_PYG_PY_PRE_RE = re.compile(
    r'(?P<open><div class="highlight-(?:python|py|pycon|ipython3?|default)[^"]*">'
    r'<div class="highlight"><pre>)(?P<body>.*?)(?P<close></pre>)', re.DOTALL)
_PYG_PY_CALL_RE = re.compile(
    r'(?P<pre><span class="n">cv2?</span><span class="o">\.</span>)'
    r'(?P<span><span class="n">)(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<end></span>)'
    r'(?=<span class="p">\()'
)


def _linkify_code_blocks(html_dir: pathlib.Path) -> None:
    """Walk every `.html` under `html_dir` and turn known identifier
    tokens inside Pygments-rendered `<pre>` blocks into clickable
    anchors. The substitution is scoped to spans inside `<pre>` so we
    don't accidentally repaint inline `<code class="n">` chips in
    prose; the rule above already targets only Pygments span classes
    that Pygments uses inside its `<pre>` output."""
    if not (_LOCAL_CLASS_URL or _LOCAL_TYPEDEF_URL or _FILE_URL or _CV_SYMBOL_URL):
        return
    if not html_dir.is_dir():
        return
    import os

    # basename -> path relative to the html root. Stored URLs are relative to
    # main_modules/, so without re-pointing they 404 when linked from a page at
    # a different depth (e.g. a tutorial referencing `core_basic.html#…`).
    _skip_dirs = {"_static", "_sources", "_images", "_sphinx_design_static"}
    _page_paths: dict[str, str] = {}
    for _f in html_dir.rglob("*.html"):
        _rel = _f.relative_to(html_dir)
        if _rel.parts and _rel.parts[0] in _skip_dirs:
            continue
        _page_paths.setdefault(_f.name, _rel.as_posix())

    def _rel_local(url: str, current_html: pathlib.Path) -> "str | None":
        # Relativize a bare local Sphinx page URL to the current page. Returns
        # None when the target page wasn't built (e.g. a struct documented
        # inline, with no standalone page) so the caller drops the link instead
        # of emitting a 404. Non-local URLs (http/already-relative) pass through.
        page, sep, frag = url.partition("#")
        if page.startswith(("http://", "https://", "../", "/")):
            return url
        tgt = _page_paths.get(page)
        if not tgt:
            return None
        rel = os.path.relpath(html_dir / tgt, start=current_html.parent)
        return rel + (sep + frag if sep else "")

    def _resolve(name: str) -> str | None:
        return _LOCAL_CLASS_URL.get(name) or _LOCAL_TYPEDEF_URL.get(name)

    # Free functions only (classes/typedefs already get a local link via
    # `_resolve`). Emits the docs.opencv.org URL; `_localize_doxygen_links`
    # (run right after) rewrites it to the local Sphinx page when one exists.
    def _resolve_fn(name: str) -> str | None:
        if name in _LOCAL_CLASS_URL or name in _LOCAL_TYPEDEF_URL:
            return None
        return _CV_SYMBOL_URL.get(name)

    def _wrap_call(m: "re.Match") -> str:
        url = _resolve_fn(m.group("name"))
        if not url:
            return m.group(0)
        return (f'<a class="reference external" href="{url}">'
                f'{m.group("span")}{m.group("name")}{m.group("end")}</a>')

    def _rewrite_cpp_calls(m: "re.Match") -> str:
        return (m.group("open")
                + _PYG_CALL_SPAN_RE.sub(_wrap_call, m.group("body"))
                + m.group("close"))

    def _wrap_py_call(m: "re.Match") -> str:
        url = _resolve_fn(m.group("name"))
        if not url:
            return m.group(0)
        return (m.group("pre")
                + f'<a class="reference external" href="{url}">'
                + f'{m.group("span")}{m.group("name")}{m.group("end")}</a>')

    def _rewrite_py_calls(m: "re.Match") -> str:
        return (m.group("open")
                + _PYG_PY_CALL_RE.sub(_wrap_py_call, m.group("body"))
                + m.group("close"))

    # `<pre>…</pre>` blocks only — keeps the substitution from touching
    # inline `<span class="n">` runs that may appear in other contexts.
    _PRE_BLOCK_RE = re.compile(r"<pre>(.*?)</pre>", re.DOTALL)

    # Relative path from each rendered `.html` file's directory to the
    # Doxygen html tree (which sits alongside `docs_sphinx/html/` at
    # `doc/doxygen/html/`). Reused per file so paths render with the
    # right number of `../` segments regardless of subdir depth.
    _DOXY_ROOT = html_dir.parent.parent / "doc" / "doxygen" / "html"

    def _doxy_rel(html_path: pathlib.Path, file_url: str) -> str:
        target = _DOXY_ROOT / file_url
        try:
            return os.path.relpath(target, start=html_path.parent)
        except ValueError:
            return f"../../../doc/doxygen/html/{file_url}"

    def _wrap_span(m: re.Match, current_html: pathlib.Path) -> str:
        name = m.group("name")
        url = _resolve(name)
        if not url:
            return m.group(0)
        url = _rel_local(url, current_html)
        if url is None:        # target page not built — don't emit a 404 link
            return m.group(0)
        return (f'<a class="reference internal" href="{url}">'
                f'{m.group("prefix")}{name}{m.group("suffix")}</a>')

    def _wrap_cpf(m: re.Match, current_html: pathlib.Path) -> str:
        path = m.group("path")
        # Only opencv headers — `<iostream>`, `<stdio.h>` are not in
        # `_FILE_URL` and stay plain.
        file_url = _FILE_URL.get(path)
        if not file_url:
            return m.group(0)
        href = _doxy_rel(current_html, file_url)
        # Keep the opening/closing quotes outside the `<a>` (so they
        # render as plain `"` / `<`/`>`), and put the link on just
        # the path text — same shape the enum-detail `#include` line
        # already uses elsewhere.
        return (f'{m.group("prefix")}{m.group("openq")}'
                f'<a class="reference external opencv-include-link" '
                f'href="{href}">{path}</a>'
                f'{m.group("closeq")}{m.group("suffix")}')

    def _rewrite_pre(m: re.Match, current_html: pathlib.Path) -> str:
        inner = m.group(1)
        # Skip spans already wrapped: if `<a …>` immediately precedes
        # the `<span class="n">…</span>`, leave it. The Pygments
        # output doesn't generate `<a>` itself, so the only place
        # `<a>` appears in the inner text is from a prior pass — we
        # detect by scanning for `<a `/`</a>` pairs and only rewrite
        # text outside them.
        out: list[str] = []
        i, n = 0, len(inner)
        while i < n:
            if inner.startswith("<a ", i):
                j = inner.find("</a>", i)
                if j < 0:
                    out.append(inner[i:])
                    break
                out.append(inner[i:j + 4])
                i = j + 4
            else:
                k = inner.find("<a ", i)
                if k < 0:
                    seg = inner[i:]
                    seg = _PYG_IDENT_SPAN_RE.sub(lambda mm: _wrap_span(mm, current_html), seg)
                    seg = _PYG_CPF_SPAN_RE.sub(
                        lambda mm: _wrap_cpf(mm, current_html), seg)
                    out.append(seg)
                    break
                seg = inner[i:k]
                seg = _PYG_IDENT_SPAN_RE.sub(lambda mm: _wrap_span(mm, current_html), seg)
                seg = _PYG_CPF_SPAN_RE.sub(
                    lambda mm: _wrap_cpf(mm, current_html), seg)
                out.append(seg)
                i = k
        return "<pre>" + "".join(out) + "</pre>"

    for html in html_dir.rglob("*.html"):
        try:
            text = html.read_text(encoding="utf-8")
        except OSError:
            continue
        if "<pre>" not in text:
            continue
        new_text = _PRE_BLOCK_RE.sub(
            lambda m: _rewrite_pre(m, html), text)
        # Function-call passes run after the class/typedef pass so constructors
        # are already wrapped and skipped by the `(` lookahead.
        new_text = _PYG_CPP_PRE_RE.sub(_rewrite_cpp_calls, new_text)
        new_text = _PYG_PY_PRE_RE.sub(_rewrite_py_calls, new_text)
        if new_text != text:
            try:
                html.write_text(new_text, encoding="utf-8")
            except OSError:
                pass


_INLINE_CODE_RE = re.compile(
    r'<code class="docutils literal notranslate">(?P<body>[^<]*?)</code>'
)
_A_BLOCK_RE = re.compile(r'<a\b[^>]*>.*?</a>', re.DOTALL)
_INLINE_TOK_RE = re.compile(r'(?:cv::)?_?[A-Za-z][A-Za-z0-9_]*')


def _linkify_inline_code(html_dir: pathlib.Path) -> None:
    if not (_LOCAL_CLASS_URL or _LOCAL_TYPEDEF_URL):
        return
    if not html_dir.is_dir():
        return

    def _resolve(name: str) -> str | None:
        bare = name[4:] if name.startswith("cv::") else name
        return _LOCAL_CLASS_URL.get(bare) or _LOCAL_TYPEDEF_URL.get(bare)

    def _anchor_text(tok: str) -> str:
        return tok.replace("::", "&#58;&#58;")

    def _linkify_body(body: str) -> str:
        def _sub(m: re.Match) -> str:
            tok = m.group(0)
            url = _resolve(tok)
            if not url:
                return tok
            return (f'<a class="reference internal" '
                    f'href="{url}">{_anchor_text(tok)}</a>')
        return _INLINE_TOK_RE.sub(_sub, body)

    def _rewrite_code(m: re.Match) -> str:
        body = m.group("body")
        new_body = _linkify_body(body)
        if new_body == body:
            return m.group(0)
        return (f'<code class="docutils literal notranslate">'
                f'{new_body}</code>')

    for html in html_dir.rglob("*.html"):
        try:
            text = html.read_text(encoding="utf-8")
        except OSError:
            continue
        if 'class="docutils literal notranslate"' not in text:
            continue
        masked: list[str] = []
        def _mask(m: re.Match) -> str:
            masked.append(m.group(0))
            return f"\x00A{len(masked) - 1}\x00"
        masked_text = _A_BLOCK_RE.sub(_mask, text)
        new_masked = _INLINE_CODE_RE.sub(_rewrite_code, masked_text)
        if new_masked == masked_text:
            continue
        new_text = re.sub(
            r"\x00A(\d+)\x00",
            lambda mm: masked[int(mm.group(1))],
            new_masked,
        )
        try:
            html.write_text(new_text, encoding="utf-8")
        except OSError:
            pass


# Center the active nav entry in the scrollable left sidebar on load so a
# module far down the list isn't below the fold. The theme has no such hook.
_AUTOSCROLL_SNIPPET = (
    '<script id="opencv-sidebar-autoscroll">'
    "document.addEventListener('DOMContentLoaded',function(){"
    "var s=document.querySelector('.bd-sidebar-primary');if(!s)return;"
    "var a=s.querySelectorAll('a.current'),t=a[a.length-1];if(!t)return;"  # deepest = current page
    "var sr=s.getBoundingClientRect(),tr=t.getBoundingClientRect();"
    "s.scrollTop+=(tr.top-sr.top)-(s.clientHeight-tr.height)/2;});"
    "</script>"
)


def _inject_sidebar_autoscroll(out_dir: pathlib.Path) -> None:
    """Inject the sidebar auto-scroll script before </body>; idempotent."""
    for html in out_dir.rglob("*.html"):
        text = html.read_text(encoding="utf-8")
        if "opencv-sidebar-autoscroll" in text or "</body>" not in text:
            continue
        html.write_text(
            text.replace("</body>", _AUTOSCROLL_SNIPPET + "</body>", 1),
            encoding="utf-8")


_TOC_NAV_RE = re.compile(r'id="pst-page-toc-nav".*?</nav>', re.S)
_TOC_HREF_RE = re.compile(r'href="#([^"]+)"')
_ANY_ID_RE = re.compile(r'id="([^"]+)"')
_DETAIL_SEC_RE = re.compile(r'(<section id="detailed-description"[^>]*>)')


def _repair_dangling_toc_anchors(out_dir: pathlib.Path) -> None:
    """Stub a missing #anchor for any secondary-TOC link with no target element.
    A dangling anchor makes the theme scroll-spy throw, which aborts init and
    kills the collapse-sidebar button on that page. Idempotent."""
    for html in out_dir.rglob("*.html"):
        text = html.read_text(encoding="utf-8")
        nav = _TOC_NAV_RE.search(text)
        if not nav:
            continue
        ids = set(_ANY_ID_RE.findall(text))
        missing = [a for a in _TOC_HREF_RE.findall(nav.group(0)) if a not in ids]
        if not missing:
            continue
        stubs = "".join(f'<span id="{a}"></span>' for a in missing)
        new = _DETAIL_SEC_RE.sub(lambda m: m.group(1) + stubs, text, count=1)
        if new != text:
            html.write_text(new, encoding="utf-8")


def _redirect_orphan_duplicates(app, out_dir: pathlib.Path) -> None:
    """An orphaned main_modules/* page that duplicates an in-nav extra_modules
    page redirects to that twin, which has proper section nav. Idempotent."""
    ti = app.env.toctree_includes
    seen, stack = set(), [app.env.config.root_doc]
    while stack:
        d = stack.pop()
        if d in seen:
            continue
        seen.add(d)
        stack.extend(ti.get(d, []))
    for doc in set(app.env.all_docs):
        if not doc.startswith("main_modules/") or doc in seen:
            continue
        base = doc.split("/", 1)[1]
        if "extra_modules/" + base not in seen:
            continue
        html = out_dir / (doc + ".html")
        if not html.is_file():
            continue
        text = html.read_text(encoding="utf-8")
        if "opencv-dup-redirect" in text:
            continue
        target = f"../extra_modules/{base}.html"
        snippet = (
            f'<link rel="canonical" href="{target}">'
            f'<script id="opencv-dup-redirect">location.replace("{target}"+location.hash)</script>'
            f'<noscript><meta http-equiv="refresh" content="0;url={target}"></noscript>'
        )
        new = text.replace("<head>", "<head>" + snippet, 1)
        if new != text:
            html.write_text(new, encoding="utf-8")


# DNN engine-selection topic: symbols → their API anchor (same dir as the page).
_DNN_ENGINE_LINKS = {
    "readNet": "dnn.html#readnet",
    "readNetFromONNX": "dnn.html#readnetfromonnx",
    "ENGINE_NEW": "dnn.html#enginetype",
    "ENGINE_CLASSIC": "dnn.html#enginetype",
    "ENGINE_AUTO": "dnn.html#enginetype",
    "ENGINE_ORT": "dnn.html#enginetype",
    "EngineType": "dnn.html#enginetype",
    "DNN_BACKEND_CUDA": "dnn.html#backend",
    "DNN_BACKEND_OPENVINO": "dnn.html#backend",
    "DNN_TARGET_CUDA": "dnn.html#target",
    "setPreferableBackend": "classcv_1_1dnn_1_1Net.html#setpreferablebackend",
    "setPreferableTarget": "classcv_1_1dnn_1_1Net.html#setpreferabletarget",
}
# Whole inline-code spans (qualified forms) → anchor; bare names reuse the above.
_DNN_ENGINE_INLINE = {
    "cv::dnn::readNet()": "dnn.html#readnet",
    "net.forward()": "classcv_1_1dnn_1_1Net.html#forward",
    "readNet*()": "dnn.html#readnet",
    **_DNN_ENGINE_LINKS,
}
_INLINE_CODE_SPAN_RE = re.compile(
    r'<code class="docutils literal notranslate">'
    r'(?:<span class="pre">)?([^<]+)(?:</span>)?</code>')
_PYG_TOKEN_SPAN_RE = re.compile(r'<span class="(n|nc|nf|nb|nv|na)">(\w+)</span>')


def _linkify_dnn_engine_selection(out_dir: pathlib.Path) -> None:
    """Make DNN engine/backend symbols clickable on the engine-selection topic,
    in both inline code and highlighted code blocks. Idempotent."""
    page = out_dir / "main_modules" / "dnn_engine_selection.html"
    if not page.is_file():
        return
    text = page.read_text(encoding="utf-8")
    if 'href="dnn.html#enginetype"' in text:        # already linkified
        return

    def _inline(m: "re.Match") -> str:
        href = _DNN_ENGINE_INLINE.get(m.group(1).strip())
        return (f'<code class="docutils literal notranslate">'
                f'<a class="reference internal" href="{href}">'
                f'<span class="pre">{m.group(1)}</span></a></code>'
                ) if href else m.group(0)

    def _token(m: "re.Match") -> str:
        href = _DNN_ENGINE_LINKS.get(m.group(2))
        return (f'<a class="reference internal" href="{href}">'
                f'<span class="{m.group(1)}">{m.group(2)}</span></a>'
                ) if href else m.group(0)

    text = _PYG_TOKEN_SPAN_RE.sub(_token, _INLINE_CODE_SPAN_RE.sub(_inline, text))
    page.write_text(text, encoding="utf-8")


# HAL page: cv::* API symbols -> their module-page anchors (same dir as the page).
_HAL_INLINE = {
    "cv::resize": "imgproc_transform.html#resize",
    "cv::cvtColor": "imgproc_color_conversions.html#cvtcolor",
    "cv::gemm": "core_array.html#gemm",
}


def _linkify_hal_page(out_dir: pathlib.Path) -> None:
    """Make the cv::* API symbols on the HAL page clickable. Idempotent.
    HAL-only tokens (cv_hal_*, NOT_IMPLEMENTED, WITH_*, …) have no API page and
    are intentionally left plain."""
    page = out_dir / "main_modules" / "hal.html"
    if not page.is_file():
        return
    text = page.read_text(encoding="utf-8")
    if 'href="imgproc_transform.html#resize"' in text:   # already linkified
        return

    def _inline(m: "re.Match") -> str:
        href = _HAL_INLINE.get(m.group(1).strip())
        return (f'<code class="docutils literal notranslate">'
                f'<a class="reference internal" href="{href}">'
                f'<span class="pre">{m.group(1)}</span></a></code>'
                ) if href else m.group(0)

    page.write_text(_INLINE_CODE_SPAN_RE.sub(_inline, text), encoding="utf-8")


# Universal Intrinsics tutorial: link the intrinsics/types in prose code, which
# Sphinx doesn't auto-link. Anchors are read off core_hal_intrin + the symbol
# maps so coverage tracks the docs (undocumented symbols stay plain).
_UI_MM = "../../../main_modules/"    # page is 3 dirs deep


def _univ_intrin_link_map(out_dir: pathlib.Path) -> dict:
    m = {"cv::hfloat": _UI_MM + "classcv_1_1hfloat.html",
         "cv::bfloat": _UI_MM + "classcv_1_1bfloat.html",
         "CV_16F": _UI_MM + "core_hal_interface.html#cv-16f",
         "CV_16BF": _UI_MM + "core_hal_interface.html#cv-16bf"}
    # Intrinsic functions, from the anchors on the rendered group page.
    hp = out_dir / "main_modules" / "core_hal_intrin.html"
    if hp.is_file():
        html = hp.read_text(encoding="utf-8")
        for a in set(re.findall(r'id="(cv-v-[a-z0-9-]+)"', html)):
            m.setdefault("v_" + a[len("cv-v-"):].replace("-", "_"),
                         _UI_MM + "core_hal_intrin.html#" + a)
    # Register typedefs (v_float32, …) and the FP16/BF16 classes, from the maps.
    for sym, url in {**_LOCAL_TYPEDEF_URL, **_LOCAL_CLASS_URL}.items():
        if sym.startswith("v_") or sym in ("hfloat", "bfloat"):
            m.setdefault(sym, _UI_MM + url)
    # Symbols with no local page but in the Doxygen tag (v_exp, v_log, …) link
    # to the official docs; those absent here too have no target and stay plain.
    for sym, url in _CV_SYMBOL_URL.items():
        if sym.startswith(("v_", "vx_")) and sym not in m:
            m[sym] = url
    return m


def _linkify_univ_intrin(out_dir: pathlib.Path) -> None:
    """Link every documented intrinsic/type on the univ_intrin tutorial (inline
    code and highlighted blocks). Idempotent; undocumented symbols stay plain."""
    page = out_dir / "tutorials" / "core" / "univ_intrin" / "univ_intrin.html"
    if not page.is_file():
        return
    text = page.read_text(encoding="utf-8")
    if _UI_MM + "classcv_1_1hfloat.html" in text:       # already linkified
        return
    links = _univ_intrin_link_map(out_dir)

    def _inline(m: "re.Match") -> str:
        txt = m.group(1).strip()
        href = links.get(txt)
        if not href:                                    # e.g. "v_exp(x)" -> v_exp
            lead = re.match(r"[A-Za-z_:][\w:]*", txt)
            href = links.get(lead.group(0)) if lead else None
        return (f'<code class="docutils literal notranslate">'
                f'<a class="reference internal" href="{href}">'
                f'<span class="pre">{m.group(1)}</span></a></code>'
                ) if href else m.group(0)

    def _token(m: "re.Match") -> str:
        href = links.get(m.group(2))
        return (f'<a class="reference internal" href="{href}">'
                f'<span class="{m.group(1)}">{m.group(2)}</span></a>'
                ) if href else m.group(0)

    text = _INLINE_CODE_SPAN_RE.sub(_inline, text)
    # Token-link only outside existing <a>…</a> spans (typedefs/classes are
    # already linked by _linkify_code_blocks) so we never nest anchors.
    parts = re.split(r"(<a\b[^>]*>.*?</a>)", text, flags=re.DOTALL)
    text = "".join(p if i % 2 else _PYG_TOKEN_SPAN_RE.sub(_token, p)
                   for i, p in enumerate(parts))
    page.write_text(text, encoding="utf-8")


def _inline_coll_graphs_on_finish(app, exception):
    """build-finished entry point."""
    if exception is not None:
        return
    out = pathlib.Path(app.outdir)
    for _api in ("main_modules", "extra_modules"):
        _inline_collaboration_svgs(out / _api, out / "_images")
        _strip_breathe_class_clutter(out / _api)
        _linkify_code_blocks(out)
    _linkify_inline_code(out)
    _localize_doxygen_links(out)
    _drop_moved_stub_search_entries()
    _copy_js_tryit_files(out)
    _fix_gapi_images(out)
    _generate_search_map(out)
    _repair_dangling_toc_anchors(out)
    _redirect_orphan_duplicates(app, out)
    _linkify_dnn_engine_selection(out)
    _linkify_hal_page(out)
    _linkify_univ_intrin(out)
    _inject_sidebar_autoscroll(out)
