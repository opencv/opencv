# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""API-reference stub writers. Entry point: ``_generate_api_stubs``."""
from __future__ import annotations
import pathlib, os as _os, re, shutil as _shutil, textwrap as _textwrap
from .state import *
from .xml_render import *
from .examples import (
    _find_examples_for_class, _render_examples_block, _generate_example_pages,
)


# xphoto TonemapDurand function documentation enhancements
_XPHOTO_DOCS = {
    "getContrast": {
        "detailed": "Retrieves the resulting contrast on logarithmic scale, computed as log(max / min), where max and min are maximum and minimum luminance values of the resulting image.",
        "returns": "resulting contrast on logarithmic scale, i. e. log(max / min), where max and min are maximum and minimum luminance values of the resulting image.",
        "sig_prefix": "virtual ", "sig_suffix": " const = 0",
    },
    "setContrast": {
        "detailed": "Sets the resulting contrast on logarithmic scale for the tone mapping algorithm.",
        "params": [("contrast", "resulting contrast on logarithmic scale, i. e. log(max / min), where max and min are maximum and minimum luminance values of the resulting image.")],
        "sig_prefix": "virtual ", "sig_suffix": " = 0",
    },
    "getSaturation": {
        "detailed": "Retrieves the current saturation enhancement value. See createTonemapDrago for details.",
        "returns": "saturation enhancement value. See createTonemapDrago",
        "sig_prefix": "virtual ", "sig_suffix": " const = 0",
    },
    "setSaturation": {
        "detailed": "Sets the saturation enhancement value for the tone mapping algorithm.",
        "params": [("saturation", "saturation enhancement value. See createTonemapDrago")],
        "sig_prefix": "virtual ", "sig_suffix": " = 0",
    },
    "getSigmaSpace": {
        "detailed": "Retrieves the sigma parameter for the bilateral filter in coordinate space.",
        "returns": "bilateral filter sigma in coordinate space",
        "sig_prefix": "virtual ", "sig_suffix": " const = 0",
    },
    "setSigmaSpace": {
        "detailed": "Sets the sigma parameter for the bilateral filter in coordinate space.",
        "params": [("sigma_space", "bilateral filter sigma in coordinate space")],
        "sig_prefix": "virtual ", "sig_suffix": " = 0",
    },
    "getSigmaColor": {
        "detailed": "Retrieves the sigma parameter for the bilateral filter in color space.",
        "returns": "bilateral filter sigma in color space",
        "sig_prefix": "virtual ", "sig_suffix": " const = 0",
    },
    "setSigmaColor": {
        "detailed": "Sets the sigma parameter for the bilateral filter in color space.",
        "params": [("sigma_color", "bilateral filter sigma in color space")],
        "sig_prefix": "virtual ", "sig_suffix": " = 0",
    },
}


def _enhance_xphoto_member(m: dict, class_name: str = "") -> dict:
    """Enhance xphoto TonemapDurand function documentation with detailed descriptions
    and full C++ qualifiers (virtual, const = 0) injected at stub generation time.

    Works for both class page (class_name provided) and module page (derive class
    from the definition field when qualifiedname is missing in the group XML)."""
    # Derive class name from definition when qualifiedname is absent (group XML)
    if not class_name:
        defn = m.get("definition") or ""
        # definition looks like: "virtual float cv::xphoto::TonemapDurand::getContrast"
        # strip return type tokens to get the qualified function name
        parts = defn.split("::")
        if len(parts) >= 3:
            func_part = parts[-1]  # e.g. "getContrast"
            class_name = "::".join(parts[:-1]).split()[-1] + "::" + "::".join(parts[1:-1])
            # Simpler: extract class from definition by splitting on "::"
            # "virtual float cv::xphoto::TonemapDurand::getContrast" -> "cv::xphoto::TonemapDurand"
            qualified_parts = defn.rsplit("::", 1)
            if len(qualified_parts) == 2:
                class_name = qualified_parts[0].split()[-1]  # last token before ::name

    if class_name != "cv::xphoto::TonemapDurand":
        return m

    func_name = m.get("name", "")
    if func_name not in _XPHOTO_DOCS:
        return m

    m = dict(m)
    docs = _XPHOTO_DOCS[func_name]

    if "detailed" in docs and not m.get("detailed"):
        m["detailed"] = docs["detailed"]
    if "returns" in docs and not m.get("returns"):
        m["returns"] = docs["returns"]
    if "params" in docs and not m.get("params"):
        m["params"] = docs["params"]
    if "sig_prefix" in docs:
        m["sig_prefix"] = docs["sig_prefix"]
    if "sig_suffix" in docs:
        m["sig_suffix"] = docs["sig_suffix"]

    # For module page: qualified name is missing, use definition to set full_name
    if not m.get("qualified") or "::" not in m.get("qualified", ""):
        defn = m.get("definition") or ""
        if "::" in defn:
            # "virtual float cv::xphoto::TonemapDurand::getContrast" -> "cv::xphoto::TonemapDurand::getContrast"
            m["qualified"] = defn.split()[-1]

    return m


# Drives write-if-changed and the stale-file sweep.
_stub_written: set[pathlib.Path] = set()

# "Shell" groups (<=2 classes, no own members) merged with their classes; see
# _write_api_stub. _MERGED_GROUPS -> [(out_path, intro_lines, classes)];
# _MERGED_CLASS_TO_GROUP -> {class_refid: group_docname} for cross-refs/redirect.
_MERGED_GROUPS: list = []
_MERGED_CLASS_TO_GROUP: dict[str, str] = {}


def _stub_write(path: pathlib.Path, content: str) -> None:
    """Write only if changed; mark path live for this run."""
    if not (path.is_file() and path.read_text(encoding="utf-8") == content):
        path.write_text(content, encoding="utf-8")
    _stub_written.add(path)

_DOXY_HTML_ROOT: pathlib.Path | None = None
_API_OUT_DIR: pathlib.Path | None = None


def _include_page_href(inc: str) -> str | None:
    """Link a `#include <...>` to the SPHINX file-reference page generated for
    that header (`_write_file_ref_stubs`), which inlines the include-dependency
    graph — a NEW Sphinx-format diagram page, never a Doxygen page. The page is
    a sibling in the same api dir (named after the Doxygen file stem, e.g.
    `ios_8h.html`). Returns None when the file isn't in the tag file."""
    rel = _FILE_URL.get(inc)
    return rel.rsplit("/", 1)[-1] if rel else None


_MEMBER_ANCHOR_RE = re.compile(r"_1[a-z][A-Za-z0-9]*$")
_SIG_WORD = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_:")
# Module dir of the page currently being written (set by _generate_api_stubs)
# so a signature `<ref>` link can be made relative to it.
_CUR_MODULE_DIR = ""


def _rel_doc_url(refid: str):
    """Relative `.html` URL from the current module page to a signature
    `<ref>`'s page: the compound's own page, or — for a member ref (enum value,
    method) — its enclosing compound's page. Handles cross-module links."""
    dn = _ANCHOR_TO_DOC.get(refid)
    if not dn:
        dn = _ANCHOR_TO_DOC.get(_MEMBER_ANCHOR_RE.sub("", refid))
    if not dn:
        return None
    tgt_dir, _, tgt_file = dn.rpartition("/")
    if not tgt_dir or tgt_dir == _CUR_MODULE_DIR:
        return f"{tgt_file or dn}.html"
    return f"../{tgt_dir}/{tgt_file}.html"


def _sig_html_with_links(line: str, url_by_text: dict):
    """HTML for a signature line, each known ref token wrapped in a link.
    Matches only at token boundaries (so a return-type token isn't re-linked
    inside the function's own `::`-qualified name). Returns None when no token
    matched — the caller then keeps a plain markdown code span."""
    import html as _html
    spans: list = []
    for text in sorted(url_by_text, key=len, reverse=True):
        start = 0
        while True:
            i = line.find(text, start)
            if i < 0:
                break
            j = i + len(text)
            start = j
            if (i == 0 or line[i - 1] not in _SIG_WORD) and \
               (j >= len(line) or line[j] not in _SIG_WORD) and \
               not any(s < j and i < e for s, e, _ in spans):
                spans.append((i, j, url_by_text[text]))
    if not spans:
        return None
    spans.sort()
    out, last = [], 0
    for s, e, url in spans:
        out.append(_html.escape(line[last:s]))
        out.append(f'<a class="reference internal opencv-sig-link" '
                   f'href="{_html.escape(url, quote=True)}">'
                   f'{_html.escape(line[s:e])}</a>')
        last = e
    out.append(_html.escape(line[last:]))
    return "".join(out)


_INCBY_CACHE: dict = {}


def _included_by_map(xml_dir: pathlib.Path) -> tuple:
    """Reverse every file compound's `<includes>` into an included-by map.

    The Doxygen XML here was generated with INCLUDED_BY_GRAPH=NO, so it carries
    no `<invincdepgraph>`; derive the relationship from `<includes>` instead.
    Returns (rev, stem_path): rev[stem] = set of stems that `#include` it;
    stem_path[stem] = its display include-path. Cached per xml_dir."""
    import xml.etree.ElementTree as _ET
    key = str(xml_dir)
    if key in _INCBY_CACHE:
        return _INCBY_CACHE[key]
    rev: dict = {}
    stem_path: dict = {}
    for fx in xml_dir.glob("*_8*.xml"):          # `_8` encodes the `.` of a file
        try:
            cd = _ET.parse(str(fx)).getroot().find("compounddef")
        except _ET.ParseError:
            continue
        if cd is None or cd.get("kind") != "file":
            continue
        f_stem = cd.get("id") or fx.stem
        loc = cd.find("location")
        fpath = (loc.get("file") if loc is not None else "") or ""
        j = fpath.find("opencv2/")
        stem_path[f_stem] = (fpath[j:] if j >= 0
                             else (cd.findtext("compoundname") or f_stem))
        for inc in cd.findall("includes"):
            iid = inc.get("refid")
            if iid:
                rev.setdefault(iid, set()).add(f_stem)
                stem_path.setdefault(iid, (inc.text or "").strip())
    _INCBY_CACHE[key] = (rev, stem_path)
    return rev, stem_path


def _generate_dep_graph_svg(stem: str, rev: dict, stem_path: dict,
                            out_dir: pathlib.Path,
                            max_nodes: int = 48):
    """Build the included-by graph for `stem` with graphviz `dot`, since Doxygen
    emitted no `*__dep.svg`. Walks UP the reverse-include graph (capped), writes
    `{stem}__dep.svg` to out_dir in a Doxygen-like style, and returns it — or
    None when there are no includers / dot is unavailable. Box-links point at
    the sibling Sphinx file pages and are rewritten by the build-finished step
    exactly like the include graph's links."""
    import subprocess, hashlib as _hl
    if not rev.get(stem):
        return None
    nodes = {stem}
    edges: set = set()
    frontier = [stem]
    while frontier and len(nodes) <= max_nodes:
        x = frontier.pop(0)
        for inc in sorted(rev.get(x, ())):
            edges.add((inc, x))
            if inc not in nodes:
                nodes.add(inc)
                frontier.append(inc)
    if len(nodes) <= 1:
        return None

    def _nid(s: str) -> str:
        return "n" + _hl.md5(s.encode("utf-8")).hexdigest()[:10]
    dot = [
        'digraph "incby" {',
        '  bgcolor="transparent";',
        '  edge [color="#1868b4", arrowsize="0.7"];',
        '  node [shape=box, style=filled, fillcolor="white", color="#3f3f3f",'
        ' fontname="Helvetica", fontsize="10", height="0.2",'
        ' margin="0.11,0.04"];',
        '  rankdir="BT";',                       # file on top, includers below
    ]
    for n in sorted(nodes):
        label = stem_path.get(n, n).replace('"', "")
        if n == stem:
            attrs = f'label="{label}", fillcolor="#bfbfbf"'   # the file itself
        else:
            attrs = f'label="{label}", URL="{n}.html"'
        dot.append(f"  {_nid(n)} [{attrs}];")
    for a, b in sorted(edges):
        dot.append(f"  {_nid(a)} -> {_nid(b)};")
    dot.append("}")
    try:
        res = subprocess.run(["dot", "-Tsvg"],
                             input="\n".join(dot).encode("utf-8"),
                             capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0 or not res.stdout:
        return None
    svg_path = out_dir / f"{stem}__dep.svg"
    try:
        svg_path.write_bytes(res.stdout)
    except OSError:
        return None
    return svg_path


def _write_file_ref_stubs(out_dir: pathlib.Path, xml_dir: pathlib.Path) -> None:
    """One Sphinx file-reference page per header (title + its include-dependency
    graph), so `#include` links land on a Sphinx diagram page. The graph SVG is
    taken from the Doxygen build and inlined via the SAME machinery as class
    collaboration diagrams (`_diagram_svg_lines` + the build-finished
    `_inline_collaboration_svgs`, which also rewrites the graph's box-links to
    local pages). Generated for every header in the tag file so the graph's
    boxes resolve to sibling file pages."""
    import xml.etree.ElementTree as _ET
    html_root = xml_dir.parent / "html"
    if not (html_root.is_dir() and _FILE_URL):
        return
    # Index both graph SVGs once (stem -> path) — avoids per-file globbing.
    incl: dict[str, pathlib.Path] = {}
    dep: dict[str, pathlib.Path] = {}
    for svg in html_root.rglob("*__incl.svg"):
        incl[svg.name[: -len("__incl.svg")]] = svg
    for svg in html_root.rglob("*__dep.svg"):
        dep[svg.name[: -len("__dep.svg")]] = svg
    # Doxygen emitted no included-by graphs here, so reconstruct them ourselves.
    _incby_rev, _incby_path = _included_by_map(xml_dir)
    # Enums/functions by their declaring header — the file XML doesn't list them
    # (Doxygen aggregates members by their <location file>), so do the same here.
    # Capture the full detail Doxygen shows on its file page: every enumerator
    # (name + value) for enums, and the return type + signature for functions,
    # plus each member's brief.
    def _norm(el) -> str:
        return " ".join(_itertext(el).split()) if el is not None else ""
    member_index: dict[str, list[dict]] = {}
    for gx in xml_dir.glob("group__*.xml"):
        try:
            groot = _ET.parse(str(gx)).getroot()
        except _ET.ParseError:
            continue
        for md in groot.iter("memberdef"):
            k = md.get("kind")
            if k not in ("enum", "function"):
                continue
            loc = md.find("location")
            f = loc.get("file", "") if loc is not None else ""
            j = f.find("opencv2/")
            if j < 0:
                continue
            name = (md.findtext("name") or "").strip()
            qual = (md.findtext("qualifiedname") or "").strip() or name
            entry = {"kind": k, "name": name, "qual": qual,
                     "id": md.get("id", ""), "brief": _norm(md.find("briefdescription"))}
            if k == "enum":
                # Bare enumerator names (IMREAD_UNCHANGED), rendered inside a
                # code-block box like the class page's "Public Types" enums.
                entry["values"] = [
                    ((ev.findtext("name") or "").strip(),
                     _norm(ev.find("initializer")))
                    for ev in md.findall("enumvalue")]
            else:
                entry["type"] = _norm(md.find("type"))
                entry["args"] = (md.findtext("argsstring") or "").strip()
                entry["static"] = md.get("static") == "yes"
            member_index.setdefault(f[j:], []).append(entry)

    def _own_or_parent_doc(refid: str) -> str | None:
        """Docname of refid's own page, or — for a nested type with no standalone
        page (e.g. `cv::ImageCollection::iterator`) — its enclosing class page."""
        dn = _ANCHOR_TO_DOC.get(refid)
        if dn:
            return dn
        body = refid
        for pref in ("class", "struct", "union"):
            if body.startswith(pref):
                body = body[len(pref):]
                break
        while "_1_1" in body:
            body = body.rsplit("_1_1", 1)[0]
            for pref in ("class", "struct", "union"):
                d = _ANCHOR_TO_DOC.get(pref + body)
                if d:
                    return d
        return None

    def _xref(refid: str, name: str) -> str:
        """`[name](/docname.md)` (own or enclosing page), else plain text.

        Plain link text (no code span) so names render as blue links like the
        original Doxygen file page — not monospace chips."""
        dn = _own_or_parent_doc(refid)
        return f"[{name}](/{dn}.md)" if dn else name

    def _ple(s: str) -> str:
        """Backslash-escape markup chars for `parsed-literal` content, so a
        signature's `<`/`>`/`*`/`_`/`[`/`]`/`` ` `` render literally while the
        block still parses the one real link (the member name) we embed."""
        for ch in "\\`*_<>[]":
            s = s.replace(ch, "\\" + ch)
        return s

    def _decl_box(decl_lines: list) -> list:
        """A `parsed-literal` box: looks like a code block (monospace, keeps
        line breaks) but PARSES inline links — so the member name inside the box
        is clickable, like the original Doxygen file page (no separate header)."""
        return [":::{parsed-literal}", ":class: opencv-decl", ""] \
            + decl_lines + [":::", ""]

    for inc, rel in _FILE_URL.items():
        stem = rel.rsplit("/", 1)[-1]
        if stem.endswith(".html"):
            stem = stem[:-5]                          # e.g. ios_8h
        base = inc.rsplit("/", 1)[-1]                 # ios.h
        # Mirror the Doxygen file page: #include directives, the include graph,
        # the included-by graph, Classes and Namespaces (members are documented
        # on their module/group pages and linked from there).
        includes, classes, namespaces = [], [], []
        fxml = xml_dir / f"{stem}.xml"
        if fxml.is_file():
            try:
                cd = _ET.parse(str(fxml)).getroot().find("compounddef")
            except _ET.ParseError:
                cd = None
            if cd is not None:
                includes = [(_i.text or "").strip() for _i in cd.findall("includes")]
                classes = [(_c.get("refid", ""), (_c.text or "").strip())
                           for _c in cd.findall("innerclass")]
                namespaces = [(_n.get("refid", ""), (_n.text or "").strip())
                              for _n in cd.findall("innernamespace")]
        # orphan: reached via #include links, not the nav toctree.
        lines = ["---", "orphan: true", "---", "", f"# {inc}", ""]
        if includes:
            lines += ["```cpp"] + [f"#include <{i}>" for i in includes] + ["```", ""]
        svg = incl.get(stem)
        if svg is not None:
            lines += _diagram_svg_lines(
                svg, out_dir, f"Include dependency graph for {base}",
                f"Include dependency graph for {base}:")
        dsvg = dep.get(stem)
        if dsvg is None:                          # Doxygen didn't make one — build it
            dsvg = _generate_dep_graph_svg(stem, _incby_rev, _incby_path, out_dir)
        if dsvg is not None:
            _dep_lines = _diagram_svg_lines(
                dsvg, out_dir, f"Files that include {base}",
                f"This graph shows which files directly or indirectly "
                f"include {base}:")
            # Our generated raw SVG is intermediate (the hashed theme variants
            # are what the page references); drop it. Never touch Doxygen's own.
            if dsvg.parent == out_dir:
                try:
                    dsvg.unlink()
                except OSError:
                    pass
            lines += _dep_lines
        if classes:
            lines += ["## Classes", ""]
            for rid, nm in classes:
                kind = ("struct" if rid.startswith("struct")
                        else "union" if rid.startswith("union") else "class")
                own = _ANCHOR_TO_DOC.get(rid)            # own page (for "More…")
                brief = " ".join(_read_class_brief(rid, xml_dir).split())
                # Clickable name inside the box; brief (+ More…) below it.
                lines += _decl_box([f"{kind} {_xref(rid, nm)}"])
                if brief:
                    more = (f" [More…](/{own}.md#detailed-description)"
                            if own else "")
                    lines += [f"{brief}{more}", ""]
            lines += [""]
        if namespaces:
            lines += ["## Namespaces", ""]
            for rid, nm in namespaces:
                # Namespace pages carry a `{#api_ns_<slug>}` heading label.
                slug = nm.replace("::", "__")
                lines += _decl_box([f"namespace [{nm}](#api_ns_{slug})"])
            lines += [""]
        # Enumerations / Functions declared in this header — each links (via its
        # `(refid)=` MyST label) to its detail on the module/group page.
        _members = member_index.get(inc, [])

        def _uniq(kind: str) -> list[dict]:
            seen, out = set(), []
            for e in _members:
                r = e.get("id")
                if e["kind"] == kind and r and r not in seen:
                    seen.add(r)
                    out.append(e)
            return out

        def _name_link(e: dict) -> str:
            return f"[{e['qual']}](#{e['id']})" if e.get("id") else e["qual"]
        enums, funcs = _uniq("enum"), _uniq("function")
        if enums:
            lines += ["## Enumerations", ""]
            for e in enums:
                # One box per enum: the enum NAME and EVERY enumerator are
                # clickable (all resolve to the enum's detail via its `(refid)=`
                # label). Brief (+ More…) below — like the original file page.
                def _ev(vn, vi):
                    nm = (f"[{_ple(vn)}](#{e['id']})" if e.get("id")
                          else _ple(vn))
                    return f"    {nm}{(' ' + _ple(vi)) if vi else ''}"
                _evs = [_ev(vn, vi) for vn, vi in e["values"]]
                _evs = [v + ("," if i < len(_evs) - 1 else "")  # no trailing comma
                        for i, v in enumerate(_evs)]
                lines += _decl_box([f"enum {_name_link(e)} {{"] + _evs + ["}"])
                if e["brief"]:
                    lines += [f"{e['brief']} [More…](#{e['id']})"
                              if e.get("id") else e["brief"], ""]
            lines += [""]
        if funcs:
            lines += ["## Functions", ""]
            for fn in funcs:
                # Signature box with the function NAME clickable inside it
                # (return type + args around it as literal text); brief below.
                pre = "static " if fn.get("static") else ""
                rty = f"{_ple(fn['type'])} " if fn["type"] else ""
                decl = f"{pre}{rty}{_name_link(fn)} {_ple(fn['args'])}".rstrip()
                lines += _decl_box([decl])
                if fn["brief"]:
                    lines += [fn["brief"], ""]
            lines += [""]
        if not (includes or svg or classes or namespaces or enums or funcs):
            lines += [f"Defined in header `{inc}`.", ""]
        _stub_write(out_dir / f"{stem}.md", "\n".join(lines))
        _ANCHOR_TO_DOC[stem] = f"{out_dir.name}/{stem}"


def _diagram_svg_lines(svg_path: pathlib.Path, out_dir: pathlib.Path,
                       alt: str, intro: str, extra_class: str = "") -> list[str]:
    """Write theme-aware variants of a Doxygen graph SVG; return its MyST block.

    `.opencv-coll-graph` is the class the build-finished step keys on to inline
    the SVG, so call/caller graphs reuse it and add `extra_class` for styling.
    Content-hashed filenames bust browser caches. Returns [] if unreadable."""
    import hashlib as _hashlib
    try:
        raw = svg_path.read_text(encoding="utf-8")
    except OSError:
        return []
    light_txt = _svg_make_transparent(raw)
    dark_txt = _svg_dark_variant(raw)
    lh = _hashlib.md5(light_txt.encode("utf-8")).hexdigest()[:10]
    dh = _hashlib.md5(dark_txt.encode("utf-8")).hexdigest()[:10]
    light_name = f"{svg_path.stem}.{lh}.svg"
    dark_name = f"{svg_path.stem}.{dh}.dark.svg"
    try:
        (out_dir / light_name).write_text(light_txt, encoding="utf-8")
        (out_dir / dark_name).write_text(dark_txt, encoding="utf-8")
    except OSError:
        return []
    _stub_written.add(out_dir / light_name)
    _stub_written.add(out_dir / dark_name)
    base = ["opencv-coll-graph"] + ([extra_class] if extra_class else [])

    def _attr(variant: str) -> str:
        return "{" + " ".join(f".{c}" for c in base + [variant]) + "}"
    return [
        intro, "",
        f"![{alt}]({light_name}){_attr('only-light')}", "",
        f"![{alt}]({dark_name}){_attr('only-dark')}", "",
    ]


def _call_graph_lines(member: dict) -> list[str]:
    """Embedded call/caller graphs for a function member detail block (or [])."""
    if _DOXY_HTML_ROOT is None or _API_OUT_DIR is None:
        return []
    out: list[str] = []
    for svg, intro, alt in _find_call_graph_svgs(member, _DOXY_HTML_ROOT):
        out += _diagram_svg_lines(svg, _API_OUT_DIR, alt, intro,
                                  extra_class="opencv-call-graph")
    return out


def _group_by_section_header(items: list[dict]) -> list[tuple[str, list[dict]]]:
    """Split members into contiguous runs sharing a Doxygen `@name` header."""
    groups: list[tuple[str, list[dict]]] = []
    for m in items:
        hdr = m.get("section_header") or ""
        if not groups or groups[-1][0] != hdr:
            groups.append((hdr, []))
        groups[-1][1].append(m)
    return groups


def _collect_all_group_names(node: dict) -> list[str]:
    """Flatten group hierarchy to every group's `name`."""
    return [node["name"]] + [n for c in node["children"]
                             for n in _collect_all_group_names(c)]


def _namespaces_section(entries: list) -> list[str]:
    """`## Namespaces` block of `@subpage` entries; `entries` is `(ns_name, anchor)`."""
    lines = ["## Namespaces", ""]
    for _ns_name, anchor in entries:
        lines.append(f"- @subpage {anchor}")
    lines.append("")
    return lines


def _write_namespace_stub(ns: dict, out_dir: pathlib.Path,
                          xml_dir: pathlib.Path,
                          ns_group_map: dict | None = None,
                          group_info: dict | None = None,
                          classes_seen: dict | None = None) -> tuple[str, str]:
    """Write namespace_<slug>.md under out_dir. Returns (anchor, fname)."""
    import xml.etree.ElementTree as _ET
    slug = ns["name"].replace("::", "__")
    anchor = f"api_ns_{slug}"
    fname = f"namespace_{slug}.md"
    lines = [f"# {ns['name']} namespace {{#{anchor}}}", ""]

    # Breadcrumbs: link back to the group pages that contain this namespace.
    if ns_group_map and group_info:
        crumbs: list[str] = []
        for grp in sorted(ns_group_map.get(ns["name"], set())):
            chain: list[str] = []
            cur: str | None = grp
            while cur and cur in group_info:
                chain.append(cur)
                cur = group_info[cur]["parent"]
            chain.reverse()
            parts = [f"[{group_info[g]['title']}]({g}.md)" for g in chain]
            if parts:
                crumbs.append(" » ".join(parts))
        if crumbs:
            lines += [" | ".join(crumbs), ""]

    if ns.get("brief"):
        lines += [ns["brief"], ""]

    # Read member sections from patched XML (has inlined group memberdefs).
    ns_sections: dict[str, list[dict]] = {}
    ns_xml_path = _PATCHED_XML_DIR / f"{ns['refid']}.xml" if ns.get("refid") else None
    if ns_xml_path and ns_xml_path.is_file():
        try:
            cd_ns = _ET.parse(ns_xml_path).getroot().find("compounddef")
            if cd_ns is not None:
                ns_pfx = ns["name"] + "::"
                for sd in cd_ns.findall("sectiondef"):
                    for md in sd.findall("memberdef"):
                        kind = md.get("kind", "")
                        section_title = dict(_MEMBERDEF_SECTIONS).get(kind)
                        if not section_title:
                            continue
                        qualified = (md.findtext("qualifiedname") or "").strip() or \
                                    (md.findtext("name") or "").strip()
                        # Skip class methods / sub-namespace members.
                        if qualified.startswith(ns_pfx) and "::" in qualified[len(ns_pfx):]:
                            continue
                        def _pt(p) -> str:
                            t = _itertext(p.find("type"))
                            arr = (p.findtext("array") or "").strip()
                            return (t + arr) if arr else t
                        enum_values = []
                        if kind == "enum":
                            for ev in md.findall("enumvalue"):
                                enum_values.append({
                                    "name":        (ev.findtext("name") or "").strip(),
                                    "initializer": (ev.findtext("initializer") or "").strip(),
                                    "brief":       _enum_value_desc(ev),
                                })
                        ns_sections.setdefault(section_title, []).append({
                            "id":          md.get("id", ""),
                            "kind":        kind,
                            "name":        (md.findtext("name") or "").strip(),
                            "qualified":   qualified,
                            "type":        _itertext(md.find("type")),
                            "type_elem":   md.find("type"),
                            "static":      md.get("static") == "yes",
                            "args":        (md.findtext("argsstring") or "").strip(),
                            "param_types": [_pt(p) for p in md.findall("param")],
                            "params_sig":  [(_pt(p),
                                             (p.findtext("declname") or "").strip(),
                                             _itertext(p.find("defval")))
                                            for p in md.findall("param")],
                            "brief":       _itertext(md.find("briefdescription")),
                            "enum_values": enum_values,
                            "strong":      md.get("strong", "no") == "yes",
                        })
        except _ET.ParseError:
            pass

    # Sub-namespaces listed in this namespace's XML.
    ns_prefix = ns["name"] + "::"
    innernamespaces = []
    if ns_xml_path and ns_xml_path.is_file():
        try:
            cd2 = _ET.parse(ns_xml_path).getroot().find("compounddef")
            if cd2 is not None:
                for inn in cd2.findall("innernamespace"):
                    iname = (inn.text or "").strip()
                    irefid = inn.get("refid", "")
                    if iname:
                        innernamespaces.append((iname, irefid))
        except _ET.ParseError:
            pass

    def _ns_has_content(refid: str) -> bool:
        f = xml_dir / f"{refid}.xml"
        if not f.is_file():
            return True
        try:
            cd3 = _ET.parse(f).getroot().find("compounddef")
            return cd3 is not None and bool(
                cd3.findall("sectiondef") or cd3.findall("innerclass") or
                cd3.findall("innernamespace"))
        except _ET.ParseError:
            return True

    nonempty_ns = [(n, r) for n, r in innernamespaces if _ns_has_content(r)]
    if nonempty_ns:
        lines += ["## Namespaces", "", "| Namespace |", "|---|"]
        for iname, _ in sorted(nonempty_ns, key=lambda x: x[0].lower()):
            short = iname[len(ns_prefix):] if iname.startswith(ns_prefix) else iname
            islug = iname.replace("::", "__")
            lines.append(f"| [namespace {short}](namespace_{islug}.md) |")
        lines.append("")

    # Classes directly in this namespace.
    refid_prefix = ns["name"].replace("::", "_1_1") + "_1_1"
    innerclasses = []
    for kind in ("struct", "class"):
        for xml_file in sorted(xml_dir.glob(f"{kind}{refid_prefix}*.xml")):
            try:
                cd2 = _ET.parse(xml_file).getroot().find("compounddef")
                if cd2 is None:
                    continue
                cname = (cd2.findtext("compoundname") or "").strip()
                # Skip classes in a sub-namespace; allow template specializations
                # whose parameters may contain qualified names (e.g. cv::Affine3<_Tp>).
                bare = cname[len(ns_prefix):].split("<")[0]
                if "::" in bare:
                    continue
                brief = _itertext(cd2.find("briefdescription"))
                innerclasses.append((xml_file.stem, cname, kind, brief))
            except _ET.ParseError:
                continue
    if innerclasses:
        lines += ["## Classes", "", "| Name |", "|---|"]
        for ic_refid, ic_name, ic_kind, ic_brief in innerclasses:
            page = _class_page_name(ic_refid)
            short_name = ic_name[len(ns_prefix):]
            # Emit the `class`/`struct` keyword as a SEPARATE, non-
            # clickable code chip — only the qualified short name is
            # the link target. Mirrors the api-group Classes table
            # post-rewrite (translate.py step 8e) so namespace-stub
            # rows read the same way: `class` plain, name clickable.
            lines.append(f"| `{ic_kind}` [`{short_name}`]({page}.md) |")
            # Surface orphan namespace classes (not in any group) to the
            # caller so their stubs get written — otherwise links like
            # `class Node` 404. Group classes are still added by
            # `_write_api_stub`; the dedupe keys off `refid`.
            if classes_seen is not None and ic_refid not in classes_seen:
                classes_seen[ic_refid] = {
                    "refid":     ic_refid,
                    "name":      ic_name,
                    "qualified": ic_name,
                    "kind":      ic_kind,
                    "brief":     ic_brief,
                }
        lines.append("")

    # Member summary tables.
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = ns_sections.get(section_title, [])
        if not items:
            continue
        # Drop the redundant summary when the detail loop covers every member
        # (all but template specializations); keep Enumerations always.
        if section_title != "Enumerations" and all(
                "<" not in (m.get("name") or "") for m in items):
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        if section_title in ("Typedefs", "Variables"):
            for m in items:
                lines.append("```cpp")
                if section_title == "Typedefs":
                    # Append <argsstring> so function-pointer typedefs (the name
                    # lives inside the "void(*" type) don't truncate; empty for
                    # plain typedefs.
                    _args = (m.get("args") or "").strip()
                    lines.append(f"typedef {m['type']} {m['name']}{_args}")
                else:
                    lines.append(f"{m['type']} {m['name']}")
                lines.append("```")
                lines.append("")
            continue
        elif section_title == "Enumerations":
            for m in items:
                if m["brief"]:
                    lines.append(_md_escape_cell(m["brief"]))
                    lines.append("")
                lines.append("```cpp")
                lines.extend(_enum_synopsis_lines(m))
                lines.append("```")
                lines.append("")
            continue
        else:
            lines += ["| Name | Description |", "|---|---|"]
            for m in items:
                lines.append(f"| [`{m['name']}`](#{m['id']}) | {_md_escape_cell(m['brief'])} |")
        lines.append("")

    # Detailed description or doxygennamespace fallback.
    if not ns_sections and not innerclasses:
        lines += ["## Detailed Description", "",
                  f"```{{doxygennamespace}} {ns['name']}", ":project: opencv", "```", ""]
    elif ns.get("detailed"):
        lines += ["## Detailed Description", "", ns["detailed"], ""]

    # Per-member detail blocks.
    seen_define_names: set[str] = set()
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = ns_sections.get(section_title, [])
        if not items:
            continue
        if section_title == "Enumerations":
            enum_items = [m for m in items if "<" not in (m.get("name") or "")]
            if enum_items:
                lines.append(f"## {_MEMBER_DETAIL_SECTION[section_title]}")
                lines.append("")
                for m in enum_items:
                    qualified = m["qualified"] or m["name"]
                    keyword = "enum class" if m.get("strong") else "enum"
                    lines.append(f"({m['id']})=")
                    lines.append(f"### {m['name']}")
                    lines.append("")
                    lines += [f"`{keyword} {qualified}`", ""]
                    if m.get("brief"):
                        lines += [_md_escape_cell(m["brief"]), ""]
                    vals = m.get("enum_values") or []
                    if vals:
                        lines += _enumerator_list_table(
                            vals, qualified, bool(m.get("strong")))
            continue
        directive = _MEMBER_DIRECTIVE.get(kind_key)
        if not directive:
            continue
        rendered = []
        for m in items:
            if "<" in (m.get("name") or ""):
                continue
            qualified = m["qualified"] or m["name"]
            if m["kind"] == "function":
                spec = qualified + _function_signature(m)
            elif m["kind"] == "define":
                if m["name"] in seen_define_names:
                    continue
                seen_define_names.add(m["name"])
                spec = m["name"]
            else:
                spec = qualified
            rendered.append((spec, directive))
        if not rendered:
            continue
        lines.append(f"## {_MEMBER_DETAIL_SECTION[section_title]}")
        lines.append("")
        for spec, dname in rendered:
            short = spec.split("(")[0].split("::")[-1]
            suffix = "()" if dname == "doxygenfunction" else ""
            lines += [f"### {short}{suffix}", "",
                      f"```{{{dname}}} {spec}", ":project: opencv", "```", ""]

    _stub_write(out_dir / fname, "\n".join(lines) + "\n")
    return anchor, fname


def _read_gapi_full_content() -> str:
    """Read and convert full content from gapi 00-root.markdown for gapi_ref stub."""
    import re as _re
    candidates = [
        CONTRIB_ROOT / "gapi" / "doc" / "00-root.markdown",
        CONTRIB_ROOT / "modules" / "gapi" / "doc" / "00-root.markdown",
    ]
    root_md = next((p for p in candidates if p.is_file()), None)
    if root_md is None:
        return ""

    # Find sample file for API Example
    sample_candidates = [
        CONTRIB_ROOT / "gapi" / "samples" / "api_example.cpp",
        CONTRIB_ROOT / "modules" / "gapi" / "samples" / "api_example.cpp",
    ]
    sample_file = next((p for p in sample_candidates if p.is_file()), None)

    text = root_md.read_text(encoding="utf-8", errors="ignore")

    # Strip top-level title — gapi_ref stub has its own title
    text = _re.sub(r"^# Graph API[^\n]*\n+", "", text)

    # Downgrade # headings to ## (since gapi_ref stub already has a # title)
    text = _re.sub(r"^# ([^\n]+)", r"## \1", text, flags=_re.MULTILINE)

    # Strip Doxygen heading anchors like {#gapi_root_intro}
    text = _re.sub(r"\s*\{#\w+\}", "", text)

    # Convert @note to MyST admonition (three colons required)
    def _note(m):
        body = m.group(1).strip()
        return f"\n:::{{note}}\n{body}\n:::\n"
    text = _re.sub(r"@note\s+(.*?)(?=\n##|\n- |\Z)", _note, text, flags=_re.DOTALL)

    # Chapter subpages: link to Doxygen HTML pages in contrib_modules
    _chapter_pages = {
        "gapi_purposes":   ("Why Graph API?",               "../contrib_modules/gapi/doc/01-background.html"),
        "gapi_hld":        ("High-level design overview",   "../contrib_modules/gapi/doc/10-hld-overview.html"),
        "gapi_kernel_api": ("Kernel API",                   "../contrib_modules/gapi/doc/20-kernel-api.html"),
        "gapi_impl":       ("Implementation details",       "../contrib_modules/gapi/doc/30-implementation.html"),
    }

    # API group subpages → links to extra_modules pages
    _api_groups = {
        "gapi_ref":     ("G-API framework",                            "gapi_ref"),
        "gapi_core":    ("G-API Core functionality",                   "gapi_core"),
        "gapi_imgproc": ("G-API Image processing functionality",       "gapi_imgproc"),
        "gapi_video":   ("G-API Video processing functionality",       "gapi_video"),
        "gapi_draw":    ("G-API Drawing and composition functionality", "gapi_draw"),
    }

    def _subpage(m):
        name = m.group(1)
        if name in _chapter_pages:
            label, url = _chapter_pages[name]
            return f"[{label}]({url})"
        if name in _api_groups:
            label, page = _api_groups[name]
            return f"[{label}](../extra_modules/{page}.md)"
        return name
    text = _re.sub(r"@subpage\s+(\w+)", _subpage, text)

    # Convert @ref tutorial_table_of_content_gapi → link to gapi tutorial page
    text = _re.sub(
        r"@ref\s+tutorial_table_of_content_gapi",
        "[tutorials and porting examples](../tutorials_contrib/gapi/gapi.md)",
        text)

    # Replace @include with fenced code block
    def _include(m):
        rel = m.group(1).strip()
        if sample_file and sample_file.is_file():
            code = sample_file.read_text(encoding="utf-8", errors="ignore")
            return f"\n```cpp\n{code.rstrip()}\n```\n"
        return f"\n`{rel}`\n"
    text = _re.sub(r"@include\s+(\S+)", _include, text)

    # Rewrite relative image paths to point to contrib_modules location
    text = _re.sub(
        r"!\[([^\]]*)\]\(pics/([^)]+)\)",
        r"![\1](../contrib_modules/gapi/doc/pics/\2)",
        text)

    # Strip HTML comments
    text = _re.sub(r"<!--.*?-->", "", text, flags=_re.DOTALL)

    return text.strip()


def _write_api_stub(node: dict, out_dir: pathlib.Path,
                    classes_seen: dict, ns_map: dict | None = None) -> None:
    """Write one .md per group node, recursing into children.

    Parent groups → @subpage index pages; leaf groups → summary tables + detail
    blocks. Inner classes populate `classes_seen` for later page emission."""
    name = node["name"]
    title = node["title"]
    out = out_dir / f"{name}.md"

    lines = [f"# {title} {{#api_{name}}}", ""]

    # Fully hand-rendered fallback page (module with no Doxygen XML): emit the
    # prepared body verbatim, then write its topic/class subpages (the body's
    # links + hidden toctree point at them). No class/subgroup XML to parse.
    if node.get("body_md"):
        lines += [node["body_md"], ""]
        _stub_write(out, "\n".join(lines) + "\n")
        for _cn, _md in node.get("child_pages", []):
            _stub_write(out_dir / f"{_cn}.md", _md + "\n")
        return

    # Classes-only shell group (<=2 classes, no own members/subgroups): defer the
    # class content to the seeded pass (_generate_api_stubs), which hosts it here
    # and redirects the standalone class pages.
    if (1 <= len(node["innerclasses"]) <= 2 and not node["sections"]
            and not node["children"]):
        _intro = list(lines)
        _txt = "\n\n".join(x for x in (
            (node.get("brief") or "").strip(),
            (node.get("detailed") or "").strip()) if x)
        if _txt:
            _intro += [_txt, ""]
        for c in node["innerclasses"]:
            classes_seen.setdefault(c["refid"], c)
            _MERGED_CLASS_TO_GROUP[c["refid"]] = f"{out_dir.name}/{name}"
        _MERGED_GROUPS.append((out, _intro, node["innerclasses"]))
        return

    # Brief + "View details" under the title (mirrors the Doxygen group page).
    _brief = node.get("brief") or ""
    if _brief:
        _more = " [View details](#detailed-description)" if node["detailed"] else ""
        lines += [f"{_brief}{_more}", ""]

    if node["children"]:
        lines += ["## Topics", ""]
        for child in node["children"]:
            lines.append(f"- @subpage api_{child['name']}")
        lines.append("")

    # Detailed Description heading — shown even when empty (ayush's layout).
    if node["detailed"]:
        _dd = f"{_brief}\n\n{node['detailed']}" if _brief else node["detailed"]
        lines += ["## Detailed Description", "", _dd, ""]
    elif node["innerclasses"] or node["sections"] or node["children"]:
        lines += ["## Detailed Description", ""]
    if ns_map and ns_map.get(name):
        lines += _namespaces_section(ns_map[name])

    if node["innerclasses"]:
        lines += ["## Classes", "", "{.api-reference-table}",
                  "| Name | Description |", "|---|---|"]
        for c in node["innerclasses"]:
            classes_seen.setdefault(c["refid"], c)
            page = _class_page_name(c["refid"])
            link = f"[`{c['kind']} {c['name']}`]({page}.md)"
            lines.append(f"| {link} | {_md_escape_cell(c['brief'])} |")
        lines.append("")

    class_qualifieds = {c.get("qualified") for c in classes_seen.values()
                        if c.get("qualified")}

    def _is_class_member(m: dict) -> bool:
        q = m.get("qualified") or ""
        if "::" not in q:
            return False
        parent = q.rsplit("::", 1)[0]
        return parent in class_qualifieds

    def _is_template_spec(m: dict) -> bool:
        name = m.get("name") or ""
        if name.startswith("operator"):
            return False
        return "<" in name

    def _member_anchor_link(m: dict, label: str, raw: bool = False) -> str:
        text = label if raw else f"`{label}`"
        if _is_class_member(m):
            q = m["qualified"]
            parent_qualified = q.rsplit("::", 1)[0]
            for c in classes_seen.values():
                if c.get("qualified") == parent_qualified:
                    return f"[{text}]({_class_page_name(c['refid'])}.md)"
        return f"[{text}](#{m['id']})"

    def _member_anchor_target(m: dict) -> str:
        """URL (page or in-page anchor) that the function name in a row
        should link to. Same resolution as `_member_anchor_link`, but
        returns the bare target string instead of a wrapped markdown
        link — so the caller can put the link ONLY around the function
        name and leave the parameter types/names as separate spans.

        On Core-functionality pages where `_render_core_basic_func`
        emits the function detail block, the anchor on the page is
        `#cv-<func_slug>` (from `_func_slug(name)`), NOT the
        slug-normalized refid `#group-core-…-1ga<hex>`. Without this
        branch the summary-table row links to a `group-…` anchor that
        doesn't exist on the page → click goes nowhere. Functions on
        non-core pages still fall back to the refid-slug form (which
        matches the MyST `({refid})=` target Sphinx emits when no
        explicit anchor is used)."""
        if _is_class_member(m):
            q = m["qualified"]
            parent_qualified = q.rsplit("::", 1)[0]
            for c in classes_seen.values():
                if c.get("qualified") == parent_qualified:
                    # Used in a raw-HTML <a href> (see _func_row_split_md); MyST
                    # only rewrites .md->.html for Markdown []() links, not raw
                    # HTML, so point at .html directly or the href 404s.
                    return f"{_class_page_name(c['refid'])}.html"
        # Functions on core pages: target the `_func_slug`-based anchor
        # that `_render_core_basic_func` actually emits.
        if _is_core_page and m.get("kind") == "function" and m.get("name"):
            return f"#{_func_slug(m['name'])}"
        import re as _re
        return f"#{_re.sub(r'_+', '-', m['id'])}"

    def _func_row_split_md(m: dict) -> str:
        """Function summary-row Name cell as ONE continuous inline-code
        block (no separate chips). The function name (with `cv::`
        prefix) is wrapped in an `<a>` inside the `<code>` and points
        at the detail anchor. Parameter types are still individually
        clickable — step 8g's pass 1 walks the inner HTML of this
        `<code>`, skips the embedded `<a>`, and linkifies any
        recognized type tokens it finds in the rest of the signature.
        Parameter names + default values stay plain (no link)."""
        from html import escape as _esc_html
        target = _member_anchor_target(m)
        # `::` is HTML-entity-encoded so the later `_linkify_cv_symbols`
        # text-level pass doesn't see `cv::Name` and nest a second
        # anchor inside ours. Browsers decode back to `:` on render.
        name_text = f"cv::{m['name']}".replace("::", "&#58;&#58;")
        name_html = (f'<a class="reference internal" '
                     f'href="{target}">{name_text}</a>')
        params_sig = m.get("params_sig") or []
        # Pipe escaping: a literal `|` inside the cell would split the
        # markdown-table row, so swap to its HTML entity.
        def _esc(s: str) -> str:
            return _esc_html(s).replace("|", "&#124;")
        if not params_sig:
            inner = f"{name_html}()"
        elif len(params_sig) == 1:
            t, nm, dv = params_sig[0]
            decl = nm + (f" = {dv}" if dv else "")
            inner = f"{name_html}({_esc(t)} {_esc(decl)})"
        else:
            # Multi-line: `<br>` inside the code block gives one param
            # per line; CSS already handles `<code>` line breaks for
            # the existing detail blocks.
            last_i = len(params_sig) - 1
            lines = [f"{name_html}("]
            for i, (t, nm, dv) in enumerate(params_sig):
                tail = " )" if i == last_i else ","
                decl = nm + (f" = {dv}" if dv else "")
                lines.append(f"    {_esc(t)} {_esc(decl)}{tail}")
            inner = "<br>".join(lines)
        return f'<code class="docutils literal notranslate">{inner}</code>'

    # Renders the summary table (or enum synopsis) for one member kind given a
    # list of members — used both for the standard per-kind sections and for
    # the @name-group sections appended afterwards. Returns markdown lines
    # (no `## heading`); enum output already carries its own trailing blanks.
    #
    # `_is_core_page` extends the original `core_basic`-only treatment to
    # every Core-functionality group page (core_array, core_cluster,
    # core_utils, …). The user's spec: "apply the same clickability +
    # redirection rules to all other pages in core functionality" — that
    # means the clickable HTML enum synopsis, the "More..." link from
    # summary to detail, the per-page hand-rolled function detail blocks,
    # the rich return-type cell, and the per-enum detail section all flip
    # on together for every `core_*` group. Any page whose group name
    # starts with `core` qualifies (top-level group is "core"; children
    # are "core_basic", "core_array", …).
    _is_core_page = name.startswith("core")
    _rich_return = _is_core_page

    def _summary_block(section_title: str, members: list) -> list[str]:
        out: list[str] = []
        if section_title == "Functions":
            out += ["{.api-reference-table .api-function-table}",
                    "| Return | Name | Description |", "|---|---|---|"]
            for m in members:
                ret_type = _md_escape_cell(m["type"])
                sig_link = _func_row_split_md(m)
                if not ret_type:
                    ret = "\u00a0"  # ctor/dtor: blank cell, never backticked
                elif _rich_return:
                    storage = "static " if m.get("static") else ""
                    if m.get("template"):
                        ret = f"`{m['template']}`<br>`{storage}{ret_type}`"
                    else:
                        ret = f"`{storage}{ret_type}`"
                else:
                    ret = f"`{ret_type}`"
                out.append(f"| {ret} | {sig_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Typedefs":
            # Type cell as a code span so steps 8b/8g linkify its tokens.
            out += ["{.api-typedef-table}",
                    "| Type | Name | Description |", "|---|---|---|"]
            for m in members:
                t = _md_escape_cell(m["type"])
                t_cell = f"`{t}`" if t else "\u00a0"
                # Function-pointer typedefs (e.g. cv::dnn::ActivationFunc) store
                # the name inside the type ("void(*") with the params in
                # <argsstring> (")(const void *...)"). Split the type: keep only
                # the return type ("void") in the Type column and move the
                # "(*<qualified>)(args)" declarator into the Name column, so the
                # row reads naturally as "void | (*cv::dnn::ActivationFunc)(...)"
                # rather than the disjoint "void(*" / "cv::ActivationFunc)(...)".
                # Plain typedefs have no argsstring, so they keep "cv::<name>".
                fp_args = (m.get("args") or "").strip()
                if fp_args and "(" in (m.get("type") or ""):
                    ret, _sep, rest = (m.get("type") or "").strip().partition("(")
                    t_cell = f"`{_md_escape_cell(ret.strip())}`" if ret.strip() else "\u00a0"
                    label = "(" + rest + (m.get("qualified") or f"cv::{m['name']}") + fp_args
                else:
                    label = f"cv::{m['name']}"
                name_link = _member_anchor_link(m, label.replace("|", "\\|"))
                out.append(f"| {t_cell} | {name_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Variables":
            out += ["{.api-reference-table}",
                    "| Type | Name | Description |", "|---|---|---|"]
            for m in members:
                t = _md_escape_cell(m["type"])
                t_cell = f"`{t}`" if t else "\u00a0"
                name_link = _member_anchor_link(m, m["name"])
                out.append(f"| {t_cell} | {name_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Enumerations":
            # Code-style synopsis (Doxygen layout) instead of name/desc table.
            # On non-core group pages we emit the synopsis only — the
            # per-value initializer list is already self-explanatory. On
            # every Core-functionality page we additionally append a
            # "More..." link, inline at the end of the brief description
            # (or alone below the synopsis when no brief exists),
            # pointing to that enum's detail block in the "Enumeration
            # Type Documentation" section emitted by the detail loop
            # below.
            _enum_more_link = True
            # On every Core-functionality page the synopsis is emitted
            # as raw HTML (NOT a ```cpp code fence) so every `cv::…`
            # token becomes its own `<a>` linking to the enum's detail
            # block. We hand-roll Pygments-style spans (`k`, `n`, `p`)
            # so the existing `.highlight pre` styling kicks in and the
            # synopsis still looks like a code block.
            _clickable_synopsis = True
            import html as _html_mod
            # Encode `::` so translate's cv-linkifier skips it.
            def _safe(s: str) -> str:
                return _html_mod.escape(s).replace("::", "&#58;&#58;")
            for m in members:
                # Named enums anchor on their `### Name` heading slug. Anonymous
                # enums anchor on the detail block's MyST `({id})=` target, whose
                # slug normalizes `_`-runs to `-`; a raw `#<id>` with underscores
                # does NOT resolve in MyST, so match the normalized form here.
                _enum_anchor = (m["name"].lower() if m.get("name")
                                else re.sub(r"_+", "-", m["id"]))
                _more = ""
                if _enum_more_link:
                    _more = f"[View details](#{_enum_anchor})"
                if _clickable_synopsis:
                    _qual = m["qualified"] or m["name"]
                    _is_strong = bool(m.get("strong"))
                    _keyword = "enum struct" if _is_strong else "enum"
                    # Enumerator-name prefix: scoped → "cv::EnumName::",
                    # unscoped → the enum's parent scope (so values
                    # render as `cv::ACCESS_READ`).
                    if _is_strong:
                        _val_prefix = _qual + "::"
                    elif "::" in _qual:
                        _val_prefix = _qual.rsplit("::", 1)[0] + "::"
                    else:
                        _val_prefix = ""
                    _href = f"#{_enum_anchor}"
                    out.append(
                        '<div class="highlight-cpp notranslate '
                        'opencv-enum-clickable"><div class="highlight"><pre>'
                    )
                    # Anonymous enums have no name to link; emit a bare `enum {`.
                    _name_html = (
                        f'<a class="reference internal opencv-enum-link" href="{_href}">'
                        f'<span class="n">{_safe(_qual)}</span></a> ' if _qual else "")
                    out.append(
                        f'<span class="k">{_html_mod.escape(_keyword)}</span> '
                        f'{_name_html}<span class="p">{{</span>'
                    )
                    _vals = m.get("enum_values") or []
                    for _i, _v in enumerate(_vals):
                        _comma = ('<span class="p">,</span>'
                                  if _i < len(_vals) - 1 else '')
                        _init = (' ' + _html_mod.escape(_v["initializer"])
                                 if _v.get("initializer") else '')
                        _full = _val_prefix + _v["name"]
                        # All enumerators link to the enum's detail section via
                        # its `#<name>` heading anchor (which exists on the
                        # page). Per-value `_CPPv4…` ids are NOT emitted in the
                        # detail block, so a per-value href would dead-link.
                        out.append(
                            f'    <a class="reference internal opencv-enum-link" href="{_href}">'
                            f'<span class="n">{_safe(_full)}</span></a>'
                            f'{_init}{_comma}'
                        )
                    out.append('<span class="p">}</span></pre></div></div>')
                    # Blank line closes the raw-HTML block (CommonMark rule 7).
                    out.append("")
                else:
                    out.append("```cpp")
                    out.extend(_enum_synopsis_lines(m))
                    out.append("```")
                # Brief + the inline "More..." link (when generated).
                if m["brief"]:
                    line = _md_escape_cell(m["brief"])
                    if _more:
                        line = f"{line} {_more}"
                        _more = ""
                    out.append(line)
                if _more:
                    out.append(_more)
                out.append("")
        else:  # Macros
            out += ["{.api-reference-table}", "| Name | Description |", "|---|---|"]
            for m in members:
                name_link = _member_anchor_link(m, m["name"])
                out.append(f"| {name_link} | {_md_escape_cell(m['brief'])} |")
        return out

    _named_groups: list[tuple[str, str, list]] = []   # (header, section_title, members)
    def _grp_detail_covers(m, kind_key):
        # Must mirror the detail loop's skips below: a summary member lacks a
        # detail block only if it's a class member or a template specialization.
        if kind_key in ("function", "variable") and _is_class_member(m):
            return False
        if _is_template_spec(m):
            return False
        return True
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        ungrouped = [m for m in items if not (m.get("section_header") or "")]
        for _hdr, _members in _group_by_section_header(
                [m for m in items if (m.get("section_header") or "")]):
            _named_groups.append((_hdr, section_title, _members))
        # Drop the redundant summary only when the detail loop covers every
        # ungrouped member; keep Enumerations always.
        if ungrouped:
            _covered = section_title != "Enumerations" and all(
                _grp_detail_covers(m, kind_key) for m in ungrouped)
            if not _covered:
                lines.append(f"## {section_title}")
                lines.append("")
                lines += _summary_block(section_title, ungrouped)
                lines.append("")
    for _hdr, section_title, _members in _named_groups:
        lines.append(f"## {_hdr}")
        lines.append("")
        lines += _summary_block(section_title, _members)
        lines.append("")

    # Detail blocks via `_render_member_detail` (breathe chokes); macros keep
    # `{doxygendefine}`; enum detail is hand-rolled (every core_* page).
    seen_define_names: set[str] = set()
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        _core_basic_funcs = (_is_core_page and kind_key == "function")
        _ov_total: dict[str, int] = {}
        _ov_idx: dict[str, int] = {}
        _slug_seen: set[str] = set()
        if _core_basic_funcs:
            for m in items:
                if _is_class_member(m) or _is_template_spec(m):
                    continue
                _ov_total[m["name"]] = _ov_total.get(m["name"], 0) + 1
        blocks: list[list[str]] = []
        for m in items:
            # Class members render on their own class page; skip on the group.
            if kind_key in ("function", "variable") and _is_class_member(m):
                continue
            # Template specializations carry `<…>`; summary table still lists them.
            if _is_template_spec(m):
                continue
            if _core_basic_funcs:
                short = m["name"]
                _ov_idx[short] = _ov_idx.get(short, 0) + 1
                slug = _func_slug(short)
                emit_anchor = slug not in _slug_seen
                _slug_seen.add(slug)
                blocks.append(_render_core_basic_func(
                    m, _ov_idx[short], _ov_total.get(short, 1), emit_anchor))
                continue
            if kind_key == "enum":
                # Every core_* page (gated by the early `continue` above).
                # Hand-rolled in place of `{doxygenenum}` — breathe's directive
                # drops every enumerator's initializer and `briefdescription`
                # (renders the `<dd>` empty), so the live page's per-value
                # `=1<<24` constants and one-line descriptions vanished.
                # Pulling from the XML metadata ourselves restores them.
                _qual = m["qualified"] or m["name"]
                _is_strong = bool(m.get("strong"))
                _keyword = "enum class" if _is_strong else "enum"
                _qual_safe = _qual.replace("::", "&#58;&#58;")
                if m.get("name"):
                    _enum_href = f"#{m['name'].lower()}"
                    blk: list[str] = [
                        f"({_sphinx_cpp_v4_id(_qual)})=",
                        f"### {m['name']}",
                        "",
                        f'<code class="docutils literal notranslate opencv-enum-sig">'
                        f'{_keyword} <a class="reference internal" '
                        f'href="{_enum_href}">{_qual_safe}</a></code>',
                        "",
                    ]
                else:
                    # Anonymous enum has no heading slug; anchor it with a MyST
                    # `({id})=` target (what `_enum_anchor` links to). A raw
                    # `<h3 id=…>` is not a MyST reference target and dead-links.
                    blk = [
                        f"({m['id']})=",
                        f"### {_keyword}",
                        "",
                    ]
                if m.get("include_file"):
                    _einc = m["include_file"]
                    _ehref = _include_page_href(_einc)
                    if _ehref:
                        blk += [
                            "{.opencv-api-include}",
                            f'<code class="docutils literal notranslate">'
                            f'#include &lt;<a class="reference external '
                            f'opencv-include-link" '
                            f'href="{_ehref}">'
                            f'{_einc}</a>&gt;</code>',
                            "",
                        ]
                    else:
                        blk += ["{.opencv-api-include}",
                                f"`#include <{_einc}>`", ""]
                if m.get("brief"):
                    blk += [m["brief"], ""]
                if m.get("detailed"):
                    blk += [m["detailed"], ""]
                _vals = m.get("enum_values") or []
                if _vals:
                    blk += ["**Enumerator:**", ""]
                    blk += _enumerator_list_table(_vals, _qual, _is_strong)
                blocks.append(blk)
                continue
            if kind_key == "define":
                # Macros aren't namespaced; dedupe arity-overloaded ones.
                if m["name"] in seen_define_names:
                    continue
                seen_define_names.add(m["name"])
            # Hand-rolled block (breathe's {doxygendefine} drops the #include and
            # the macro's Value); `_render_member_detail` keeps `#define NAME(…)`,
            # the include row and the Value, and emits the `(id)=` cross-ref anchor.
            m = _enhance_xphoto_member(m)
            _full_name = m["qualified"] or m["name"]
            blocks.append(
                _render_member_detail(m, _full_name))
        if not blocks:
            continue
        lines.append(f"## {_MEMBER_DETAIL_SECTION[section_title]}")
        lines.append("")
        for b in blocks:
            lines += b

    # Hidden toctree registers per-class pages in the sidebar.
    if node["innerclasses"]:
        lines += ["```{toctree}", ":hidden:", ":maxdepth: 1", ""]
        for c in node["innerclasses"]:
            lines.append(_class_page_name(c["refid"]))
        lines += ["```", ""]

    _stub_write(out, "\n".join(lines) + "\n")
    # Subgroup pages: recurse (no-op for leaf groups).
    for child in node["children"]:
        _write_api_stub(child, out_dir, classes_seen, ns_map)


def _collect_base_chain(data: dict, xml_dir: pathlib.Path) -> list:
    """Transitive base classes in inheritance order, each with its loaded class
    data — for rendering Doxygen's "inherited from <base>" member sections.
    Deduped and cycle-safe."""
    out: list = []
    seen: set = set()
    queue = list(data.get("bases", []))
    while queue:
        refid, name, _prot = queue.pop(0)
        if not refid or refid in seen:
            continue
        seen.add(refid)
        bdata = _read_class_data(refid, xml_dir)
        if bdata is None:
            continue
        out.append((refid, bdata.get("name") or name, bdata))
        queue.extend(bdata.get("bases", []))
    return out


def _inherited_section(summary_title: str, base_qual: str, base_refid: str,
                       items: list) -> list[str]:
    """A collapsible "<title> inherited from <base>" block (sphinx-design
    dropdown). The base name and every member name are clickable links; the
    return-type column stays plain — only the links are blue."""
    # Absolute (root-relative) docname so the base link resolves even when the
    # base class lives in a different module dir (e.g. cv::Algorithm in
    # main_modules linked from an extra_modules page).
    _dn = _ANCHOR_TO_DOC.get(base_refid)
    base_link = f"[{base_qual}](/{_dn}.md)" if _dn else base_qual
    out = [f":::{{dropdown}} {summary_title} inherited from {base_link}",
           ":animate: fade-in", "",
           "{.api-reference-table .api-function-table}",
           "| Return | Name | Description |", "|---|---|---|"]
    for m in items:
        ret = _md_escape_cell(m["type"])
        if ret and m.get("static"):
            ret = "static " + ret
        ret_cell = f"`{ret}`" if ret else " "
        if m["kind"] == "function":
            label = _func_sig_md(m["name"], m.get("params_sig"))
            sig_link = f"[{label}](#{m['id']})"
        else:
            sig = f"{m['name']}{_md_escape_cell(m['args'])}"
            sig_link = f"[`{sig}`](#{m['id']})"
        out.append(f"| {ret_cell} | {sig_link} | {_md_escape_cell(m['brief'])} |")
    out += ["", ":::", ""]
    return out


# sectiondef kind → summary heading, in Doxygen order.
_CLASS_SUMMARY_SECTIONS = [
    ("public-type",             "Public Types"),
    ("public-func",             "Public Member Functions"),
    ("public-static-func",      "Static Public Member Functions"),
    ("public-attrib",           "Public Attributes"),
    ("public-static-attrib",    "Static Public Attributes"),
    ("protected-type",          "Protected Types"),
    ("protected-func",          "Protected Member Functions"),
    ("protected-static-func",   "Static Protected Member Functions"),
    ("protected-attrib",        "Protected Attributes"),
    ("protected-static-attrib", "Static Protected Attributes"),
    ("friend",                  "Friends"),
]


def _param_item_lines(nm: str, desc: str) -> list[str]:
    """Render one `**Parameters**` entry, indenting any multi-line / bulleted
    description so it nests under the param bullet. Without this a description
    carrying its own list (e.g. calibration `flags`) collapses into a run-on
    blob or breaks out past the card boundary as a flat sibling list.

    Param NAME renders as a monospace box (`code.opencv-param-name`, styled
    for light + dark in custom.css)."""
    import html as _html
    chip = f'<code class="opencv-param-name">{_html.escape(nm)}</code>'
    if not desc:
        return [f"- {chip}"]
    lines = desc.split("\n")
    out = [f"- {chip} — {lines[0]}"]
    # Continuation lines align with the bullet's content column (2 spaces);
    # blank lines stay empty so the nested list/paragraphs render loosely.
    out += [f"  {ln}" if ln.strip() else "" for ln in lines[1:]]
    return out


def _enumerator_list_table(values: list[dict], enum_qualified: str,
                           is_strong: bool, with_anchors: bool = False) -> list[str]:
    """Render an enum's values as a MyST `{list-table}`.

    Unlike a Markdown pipe table, list-table cells hold block content, so a
    value's description keeps its `@note` admonition, lists and links instead of
    being flattened (see `_enum_value_desc`). `with_anchors` adds a per-value
    `<span id>` so the clickable class-synopsis links resolve to each row."""
    if not values:
        return []
    has_desc = any((v.get("brief") or "").strip() for v in values)
    # Lead with a blank line: the class-member enum path emits a raw `<h3>` (and
    # optional `<p>` brief) immediately before this table. Without the separator
    # CommonMark folds the `​```{list-table}` opener and its `:option:` lines into
    # that HTML block (HTML blocks run until a blank line), leaving the closing
    # ``` orphaned — it then opens a stray code block that swallows the rest of
    # the page (Member Function docs, etc.) as literal text.
    out = ["", "```{list-table}", ":header-rows: 0",
           f":widths: {'30 70' if has_desc else '100'}",
           ":class: opencv-enum-table", ""]
    for v in values:
        nm = v["name"]
        cell = ""
        if with_anchors:
            cell = f'<span id="{_sphinx_cpp_v4_id(f"{enum_qualified}::{nm}")}"></span>'
        cell += f"`{nm}`"
        py = _python_enum_name(enum_qualified, nm, is_strong)
        if py:
            cell += f"<br>Python: `{py}`"
        out.append(f"* - {cell}")
        if has_desc:
            desc = (v.get("brief") or "").strip()
            if desc:
                _dl = desc.split("\n")
                out.append(f"  - {_dl[0]}")
                out += [("    " + ln) if ln.strip() else "" for ln in _dl[1:]]
            else:
                out.append("  -")
    out += ["```", ""]
    return out


def _signature_lines(head: str, params_sig: list) -> list[str]:
    """Doxygen-style declaration split across lines, one parameter per line
    (`type name`, single-spaced — no column padding).

    `head` is everything up to the `(` (e.g. ``double cv::calibrateCamera``).
    Returns plain strings; the caller wraps each as inline code. A 0/1-param
    declaration stays on one line — the wrapping only helps long lists."""
    if not params_sig:
        return [f"{head}()"]
    def _decl(nm: str, dv: str) -> str:
        return (nm + (f" = {dv}" if dv else "")).strip()
    if len(params_sig) == 1:
        t, nm, dv = params_sig[0]
        inner = f"{t} {_decl(nm, dv)}".strip()
        return [f"{head}({inner})"]
    lines = [f"{head}("]
    last = len(params_sig) - 1
    for i, (t, nm, dv) in enumerate(params_sig):
        tail = " )" if i == last else ","
        # Single space between type and name (no column padding).
        lines.append(f"    {t} {_decl(nm, dv)}{tail}".rstrip())
    return lines


def _func_sig_md(name: str, params_sig: list) -> str:
    """Multi-line signature for a summary-table cell, matching the detail block.

    Each line from `_signature_lines` becomes its own inline-code span joined
    by `<br>`, so the padded type column survives (CSS gives these spans
    `white-space: pre-wrap`) and the whole signature stays one clickable link.
    Pipes are escaped per line so an `A|B` default can't break the table cell."""
    return "<br>".join(
        f"`{ln.replace('|', chr(0x5c) + '|')}`"
        for ln in _signature_lines(name, params_sig or []))


def _render_member_detail(m: dict, full_name: str) -> list[str]:
    """Render one member's detail block from XML (no breathe; it chokes).

    `full_name` is the declaration name; `(id)=` keeps `#refid` links working."""
    short = m["name"]
    kind = m["kind"]
    # Heading is just `name()` for functions; full signature is in the block below.
    head = f"{short}()" if kind == "function" else short
    if m.get("id"):
        _LOCAL_MEMBER_IDS.add(m["id"])
    out = [f"({m['id']})=", f"### {head}".rstrip(), ""]

    # Declaration (template line, if any, then the C++ signature).
    tmpl = m.get("template") or ""
    prefix = "static " if m.get("static") else ""
    typ = (m.get("type") or "").strip()
    if kind == "function":
        # One parameter per line, type column padded so names align.
        # sig_prefix/sig_suffix allow callers to inject qualifiers (e.g. virtual/const = 0).
        sig_prefix = m.get("sig_prefix") or ""
        sig_suffix = m.get("sig_suffix") or ""
        head = f"{sig_prefix}{prefix}{typ + ' ' if typ else ''}{full_name}"
        sig_lines = _signature_lines(head, m.get("params_sig") or [])
        if sig_suffix:
            sig_lines[-1] = sig_lines[-1] + sig_suffix
    elif kind == "define":
        # `#define NAME(args)` — macro params carry only a name, no type.
        mp = m.get("macro_params") or []
        params = f"({', '.join(mp)})" if mp else ""
        sig_lines = [f"#define {short}{params}"]
    elif kind == "typedef":
        # Function-pointer typedefs (e.g. cv::dnn::ActivationFunc) are split by
        # Doxygen across <type>/<name>/<argsstring>: type="void(*", the name sits
        # in the middle, and argsstring=")(const void *input, …)". Append the
        # args or the decl truncates to "typedef void(* cv::dnn::ActivationFunc".
        # Plain typedefs carry an empty argsstring, so this is a no-op for them.
        args = (m.get("args") or "").strip()
        sig_lines = [f"typedef {typ} {full_name}{args}".strip()]
    else:  # variable / attribute — append the `= value` initializer if present.
        decl = f"{prefix}{typ + ' ' if typ else ''}{full_name}".strip()
        init = (m.get("initializer") or "").strip()
        sig_lines = [f"{decl} {init}".strip() if init else decl]
    # Template clause + declaration as inline code (keeps token-linkifier
    # active); `{.opencv-api-sig}` lets the CSS preserve the alignment spaces.
    # Lines carrying a `<ref>` (a type/typedef/enum value Doxygen hyperlinks)
    # are emitted as an HTML <code> with explicit, refid-resolved links — same
    # element as a markdown code span, so the box format is unchanged.
    _url_by_text: dict = {}
    for _t, _rid in (m.get("sig_refs") or []):
        if _t and _t not in _url_by_text:
            _u = _rel_doc_url(_rid)
            if _u:
                _url_by_text[_t] = _u

    import html as _html_pkg
    def _code(ln: str) -> str:
        if _url_by_text:
            _h = _sig_html_with_links(ln, _url_by_text)
            if _h is not None:
                return f'<code class="docutils literal notranslate">{_h}</code>'
        return (f'<code class="docutils literal notranslate">'
                f'{_html_pkg.escape(ln)}</code>')
    _sig = ([f"`{tmpl}`"] if tmpl else []) + [_code(ln) for ln in sig_lines]
    out += ["{.opencv-api-sig}", "\\\n".join(_sig), ""]

    inc = (m.get("include_file") or "").strip()
    if inc:
        _href = _include_page_href(inc)
        if _href:
            out += [
                "{.opencv-api-include}",
                f'<code class="docutils literal notranslate">'
                f'#include &lt;<a class="reference external '
                f'opencv-include-link" href="{_href}">{inc}</a>&gt;</code>',
                "",
            ]
        else:
            out += ["{.opencv-api-include}", f"`#include <{inc}>`", ""]

    # Macro body, shown as docs.opencv.org's "Value:" row.
    if kind == "define":
        val = (m.get("initializer") or "").strip()
        if val:
            out += ["**Value:**", "", "```cpp", val, "```", ""]

    # Python binding signature(s) from pyopencv_signatures.json (dormant until built).
    if kind == "function":
        py_entries = (_PY_SIGNATURES.get(full_name)
                      or _PY_SIGNATURES.get(f"cv::{full_name}")
                      or [])
        if py_entries:
            out += ["**Python:**", ""]
            for e in py_entries:
                py_name = e.get("name", "")
                if not py_name:
                    continue
                py_sig = f"{py_name}({e.get('arg', '')})"
                py_ret = e.get("ret", "")
                if py_ret and py_ret not in ("None", ""):
                    py_sig += f" -> {py_ret}"
                out += ["```python", py_sig, "```", ""]

    if m.get("brief"):
        out += [m["brief"], ""]
    if m.get("detailed"):
        out += [m["detailed"], ""]
    if m.get("params"):
        out += ["**Parameters**", ""]
        for nm, desc in m["params"]:
            out += _param_item_lines(nm, desc)
        out.append("")
    if m.get("returns"):
        out += ["**Returns**", "", m["returns"], ""]
    out += _call_graph_lines(m)
    return out


def _render_core_basic_func(m: dict, idx: int, total: int,
                            emit_anchor: bool) -> list[str]:
    """Hand-rolled Function block for core_basic (breathe can't parse it).

    Signature is inline code for token-linkifier (translate step 8g); heading
    `{#cv-slug}` anchor (first overload) is the Functions-table target (step 8i)."""
    short = m["name"]
    slug = _func_slug(short)
    suffix = f" [{idx}/{total}]" if total > 1 else ""
    head = f"### {short}(){suffix}"
    out = [f"{head} {{#{slug}}}" if emit_anchor else head, ""]
    # Template clause + signature as inline code (keeps token-linkifier active).
    ret = m.get("type") or ""
    storage = ("static " if m.get("static") else "") \
        + ("inline " if m.get("inline") else "")
    qname = m["qualified"] or m["name"]
    head = f"{storage}{ret} {qname}".strip()
    sig_lines = _signature_lines(head, m.get("params_sig") or [])
    _url_by_text: dict = {}
    for _t, _rid in (m.get("sig_refs") or []):
        if _t and _t not in _url_by_text:
            _u = _rel_doc_url(_rid)
            if _u:
                _url_by_text[_t] = _u

    import html as _html_pkg
    def _code(ln: str) -> str:
        if _url_by_text:
            _h = _sig_html_with_links(ln, _url_by_text)
            if _h is not None:
                return f'<code class="docutils literal notranslate">{_h}</code>'
        return (f'<code class="docutils literal notranslate">'
                f'{_html_pkg.escape(ln)}</code>')
    _sig = ([f"`{m['template']}`"] if m.get("template") else []) + \
        [_code(ln) for ln in sig_lines]
    out += ["{.opencv-api-sig}", "\\\n".join(_sig), ""]
    if m.get("include_file"):
        _ipath = m["include_file"]
        _href = _include_page_href(_ipath)
        if _href:
            out += [
                "{.opencv-api-include}",
                f'<code class="docutils literal notranslate">'
                f'#include &lt;<a class="reference external '
                f'opencv-include-link" href="{_href}">{_ipath}</a>&gt;</code>',
                "",
            ]
        else:
            out += ["{.opencv-api-include}", f"`#include <{_ipath}>`", ""]
    if m.get("brief"):
        out += [m["brief"], ""]
    if m.get("detailed"):
        out += [m["detailed"], ""]
    if m.get("params"):
        out += ["**Parameters**", ""]
        for nm, desc in m["params"]:
            out += _param_item_lines(nm, desc)
        out.append("")
    if m.get("returns"):
        out += [f"**Returns** — {m['returns']}", ""]
    out += _call_graph_lines(m)
    return out


def _write_class_stub(cls: dict, out_dir: pathlib.Path,
                      xml_dir: pathlib.Path, inline: bool = False):
    """One .md per inner class, mirroring Doxygen's class-page layout.

    Falls back to `{doxygenclass}`/`{doxygenstruct}` if class XML can't be read.
    When `inline=True`, the body lines are returned (no title, not written) so a
    single-class "shell" group page can host the class content directly."""
    page = _class_page_name(cls["refid"])
    out = out_dir / f"{page}.md"
    qualified = cls["qualified"] or cls["name"]
    kind_label = cls["kind"].title()
    title = f"{kind_label} {qualified}"
    # No `{#refid}` anchor; `_generate_api_stubs` seeds `_ANCHOR_TO_DOC` instead.
    lines = [] if inline else [f"# {title}", ""]

    # Class-page header: brief + `View details` + `#include` line.
    _header_data = _read_class_data(cls["refid"], xml_dir)
    if _header_data is not None:
        import html as _html_pkg
        _brief = (_header_data.get("brief") or "").strip()
        if _brief:
            # Link only when there's a detailed description to jump to; skip when
            # inlined, where the #detailed-description anchor collides with a
            # sibling class and the detail is right below anyway.
            _more = (
                ' <a class="opencv-class-more" href="#detailed-description">View details</a>'
                if _header_data.get("detailed") and not inline else ""
            )
            lines.append(
                f'<p class="opencv-class-brief">'
                f'{_html_pkg.escape(_brief)}{_more}</p>'
            )
            lines.append("")
        _inc = (_header_data.get("include") or "").strip()
        if _inc:
            _chref = _include_page_href(_inc)
            if _chref:
                _inc_code = (
                    f'#include &lt;<a class="reference external '
                    f'opencv-include-link" '
                    f'href="{_chref}">'
                    f'{_html_pkg.escape(_inc)}</a>&gt;')
            else:
                _inc_code = f'#include &lt;{_html_pkg.escape(_inc)}&gt;'
            lines.append(
                f'<div class="opencv-class-include"><code>{_inc_code}</code></div>'
            )
            lines.append("")

    _svg = _find_collaboration_svg(cls["refid"], xml_dir.parent / "html")
    if _svg is not None:
        lines += _diagram_svg_lines(
            _svg, out_dir,
            f"Collaboration diagram for {qualified}",
            f"Collaboration diagram for {qualified}:")

    data = _read_class_data(cls["refid"], xml_dir)
    if data is None:  # missing XML
        directive = "doxygenstruct" if cls["kind"] == "struct" else "doxygenclass"
        lines += [
            f"```{{{directive}}} {qualified}",
            ":project: opencv",
            ":members:",
            ":protected-members:",
            ":undoc-members:",
            "```",
            "",
        ]
        _stub_write(out, "\n".join(lines))
        return

    # 1) Summary tables in Doxygen's order, each followed by the same-kind
    #    members inherited from base classes (collapsible, like Doxygen).
    base_chain = _collect_base_chain(data, xml_dir)
    _additional: list = []   # protected/private inherited -> trailing group
    for sd_kind, summary_title in _CLASS_SUMMARY_SECTIONS:
        items = data["sections"].get(sd_kind, [])
        inherited = [(rid, qual, bdata["sections"].get(sd_kind, []))
                     for rid, qual, bdata in base_chain
                     if bdata["sections"].get(sd_kind, [])]
        # Public inherited members sit inline under their section; inherited
        # protected/private members go in a trailing "Additional Inherited
        # Members" group, matching Doxygen's layout.
        if sd_kind.startswith("public"):
            inline_inherited = inherited
        else:
            inline_inherited = []
            _additional += [(summary_title, rid, qual, bi)
                            for rid, qual, bi in inherited]
        non_enum_items = [m for m in items if m["kind"] != "enum"]
        enum_items = [m for m in items if m["kind"] == "enum"]
        # Own typedef/function/variable get a detail block below, so drop their
        # summary rows; keep kinds without one (e.g. friends) and enums.
        kept_rows = [m for m in non_enum_items
                     if m["kind"] not in ("typedef", "function", "variable")]
        if not kept_rows and not enum_items and not inline_inherited:
            continue
        lines.append(f"## {summary_title}")
        lines.append("")
        if kept_rows:
            lines += ["{.api-reference-table .api-function-table}",
                      "| Return | Name | Description |", "|---|---|---|"]
            for m in kept_rows:
                ret = _md_escape_cell(m["type"])
                if ret and m["static"]:
                    ret = "static " + ret
                ret_cell = f"`{ret}`" if ret else "\u00a0"
                if m["kind"] == "function":
                    sig_link = f"[{_func_sig_md(m['name'], m.get('params_sig'))}](#{m['id']})"
                else:
                    sig = f"{m['name']}{_md_escape_cell(m['args'])}"
                    sig_link = f"[`{sig}`](#{m['id']})"
                lines.append(
                    f"| {ret_cell} | {sig_link} | {_md_escape_cell(m['brief'])} |")
            lines.append("")
        for m in enum_items:
            if m["brief"]:
                lines.append(_md_escape_cell(m["brief"]))
                lines.append("")
            # HTML synopsis: `<a>` ids match the `_CPPv4…` detail anchors.
            lines.extend(_enum_synopsis_html(m, strip_scope=qualified))
            lines.append("")
        # Public members of this kind inherited from each base class (inline).
        for _b_refid, _b_qual, _b_items in inline_inherited:
            lines += _inherited_section(summary_title, _b_qual, _b_refid, _b_items)

    # Inherited protected/private members, grouped like Doxygen.
    if _additional:
        lines += ["## Additional Inherited Members", ""]
        for _title, _b_refid, _b_qual, _b_items in _additional:
            lines += _inherited_section(_title, _b_qual, _b_refid, _b_items)

    _directive = "doxygenstruct" if cls["kind"] == "struct" else "doxygenclass"
    examples = _find_examples_for_class(qualified.rsplit("::", 1)[-1])
    if data["detailed"]:
        lines += [
            "## Detailed Description",
            "",
            f"```{{{_directive}}} {qualified}",
            ":project: opencv",
            "```",
            "",
        ]
        lines += _render_examples_block(examples)
    elif (data.get("brief") or "").strip():
        # No separate detailed text — show the brief under a Detailed
        # Description heading so the section (and its `#detailed-description`
        # anchor, which "More…" links target) exists, like the original docs.
        lines += ["## Detailed Description", "", data["brief"].strip(), ""]
        lines += _render_examples_block(examples)
    elif examples:
        lines += ["## Examples", ""]
        lines += _render_examples_block(examples)

    # 3) Per-member detail blocks (ctor/dtor split from other functions).
    class_simple = qualified.rsplit("::", 1)[-1]

    typedef_items: list[dict] = []
    enum_items_all: list[dict] = []
    ctor_dtor_items: list[dict] = []
    func_items: list[dict] = []
    var_items: list[dict] = []
    for sd_items in data["sections"].values():
        for m in sd_items:
            if m["kind"] == "typedef":
                typedef_items.append(m)
            elif m["kind"] == "enum":
                enum_items_all.append(m)
            elif m["kind"] == "function":
                if m["name"] == class_simple or m["name"] == f"~{class_simple}":
                    ctor_dtor_items.append(m)
                else:
                    func_items.append(m)
            elif m["kind"] == "variable":
                var_items.append(m)

    if typedef_items:
        lines += ["## Member Typedef Documentation", ""]
        for m in typedef_items:
            lines += _render_member_detail(m, f"{qualified}::{m['name']}")

    if enum_items_all:
        # Target of the Public Types synopsis links.
        import html as _html
        lines += ["## Member Enumeration Documentation", ""]
        for m in enum_items_all:
            lines.append(f"({m['id']})=")  # legacy MyST anchor for old @ref
            enum_qualified = m.get("qualified") or m["name"]
            enum_id = _sphinx_cpp_v4_id(enum_qualified)
            enum_short = enum_qualified.rsplit("::", 1)[-1]
            lines.append(
                f'<h3 class="opencv-enum-heading" id="{enum_id}">'
                f'enum <span class="opencv-enum-name">{_html.escape(enum_short)}</span></h3>'
            )
            if m["brief"]:
                lines.append(f"<p>{_html.escape(_md_escape_cell(m['brief']))}</p>")
            lines += _enumerator_list_table(
                m.get("enum_values") or [], enum_qualified,
                bool(m.get("strong")), with_anchors=True)

    # Dedupe by refid (a memberdef can span sectiondefs).
    def _dedupe(items: list[dict]) -> list[dict]:
        seen, out = set(), []
        for m in items:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            out.append(m)
        return out

    if ctor_dtor_items:
        lines += ["## Constructor & Destructor Documentation", ""]
        for m in _dedupe(ctor_dtor_items):
            lines += _render_member_detail(m, f"{qualified}::{m['name']}")

    if func_items:
        lines += ["## Member Function Documentation", ""]
        for m in _dedupe(func_items):
            # Enhance xphoto documentation with detailed descriptions
            m = _enhance_xphoto_member(m, qualified)
            lines += _render_member_detail(m, f"{qualified}::{m['name']}")

    if var_items:
        lines += ["## Member Data Documentation", ""]
        for m in _dedupe(var_items):
            lines += _render_member_detail(m, f"{qualified}::{m['name']}")

    _src_inc = (_header_data.get("include") or "").strip() if _header_data else ""
    if _src_inc:
        import html as _html_pkg2
        _kind_word = "struct" if cls["kind"] == "struct" else "class"
        _dir = _src_inc.rsplit("/", 1)[0] + "/" if "/" in _src_inc else ""
        _base = _src_inc.rsplit("/", 1)[-1]
        _fhref = _include_page_href(_src_inc)
        if _fhref:
            _flink = (f'{_html_pkg2.escape(_dir)}<a class="reference external '
                      f'opencv-include-link" '
                      f'href="{_fhref}">'
                      f'{_html_pkg2.escape(_base)}</a>')
        else:
            _flink = _html_pkg2.escape(_src_inc)
        lines += [
            "",
            "{.opencv-class-files}",
            "## Source file",
            "",
            f"The documentation for this {_kind_word} was generated from the "
            "following file:",
            "",
            f"- {_flink}",
            "",
        ]

    if inline:
        return lines
    _stub_write(out, "\n".join(lines))


def _write_placeholder_stubs(out_dir: pathlib.Path,
                             xml_dir: pathlib.Path) -> None:
    """Stub pages for Doxygen's bare template-param classes (`_Tp`, `float_type`).
    Doxygen renders near-empty `class…` pages for these (title + collaboration
    diagram); mirror that so diagram cross-links resolve instead of 404ing.
    Marked `orphan` since nothing toctrees them."""
    html_root = xml_dir.parent / "html"
    for _doxy_file, (_display, _page) in _PLACEHOLDER_STUBS.items():
        stem = _page[:-5] if _page.endswith(".html") else _page
        refid = _doxy_file[:-5] if _doxy_file.endswith(".html") else _doxy_file
        lines = ["---", "orphan: true", "---", "",
                 f"# {_display} Class Reference", ""]
        _svg = _find_collaboration_svg(refid, html_root)
        if _svg is not None:
            lines += _diagram_svg_lines(
                _svg, out_dir,
                f"Collaboration diagram for {_display}",
                f"Collaboration diagram for {_display}:")
        lines += ["", "The documentation for this class was generated from the "
                  "following files:", ""]
        _stub_write(out_dir / f"{stem}.md", "\n".join(lines))
        _ANCHOR_TO_DOC[stem] = f"{out_dir.name}/{stem}"


# Feature-complete content for optional modules whose Doxygen group XML is
# absent from this build (disabled / hardware-gated — e.g. cannops needs the
# Ascend SDK). When the XML is missing we render a full page from this data —
# Topics, Classes, Functions and Function Documentation — mirroring the official
# group pages. Links are either working `#include` file pages or in-page anchors
# (no dependency on class/subgroup pages that don't exist here → no 404s).
# Each module: title, include (umbrella, top of page), fn_include (shown in
# Function Documentation), description (markdown), topics, classes
# (kind, qualified, brief), functions (return, qualified, args, brief).
_FALLBACK_MODULE_DATA: dict = {
    "datasets": {
        "title": "Framework for working with different datasets",
        "include": "opencv2/datasets/dataset.hpp",
        "fn_include": "opencv2/datasets/util.hpp",
        "description":
            "The datasets module includes classes for working with different "
            "datasets: load data, evaluate different algorithms on them, "
            "contains benchmarks, etc.\n\n"
            "It is planned to have:\n\n"
            "- *basic*: loading code for all datasets to help start work with "
            "them.\n"
            "- *next stage*: quick benchmarks for all datasets to show how to "
            "solve them using OpenCV and implement evaluation code.\n"
            "- *finally*: implement on OpenCV state-of-the-art algorithms, "
            "which solve these tasks.",
        "topics": ["Action Recognition", "Face Recognition",
                   "Gesture Recognition", "Human Pose Estimation",
                   "Image Registration", "Image Segmentation",
                   "Multiview Stereo Matching", "Object Recognition",
                   "Pedestrian Detection", "SLAM", "Super Resolution",
                   "Text Recognition", "Tracking"],
        "classes": [
            ("class",  "cv::datasets::Dataset", ""),
            ("struct", "cv::datasets::Object", ""),
        ],
        "functions": [
            ("void", "cv::datasets::createDirectory",
             "(const std::string &path)",
             "Create a directory at the given path."),
            ("void", "cv::datasets::getDirList",
             "(const std::string &dirName, "
             "std::vector< std::string > &fileNames)",
             "List the file names in a directory."),
            ("void", "cv::datasets::split",
             "(const std::string &s, std::vector< std::string > &elems, "
             "char delim)",
             "Split a string into tokens on a delimiter."),
        ],
    },
    "dnn_objdetect": {
        "title": "DNN used for object detection",
        "include": "opencv2/core_detect.hpp",
        "fn_include": "opencv2/core_detect.hpp",
        "description":
            "The dnn_objdetect module includes deep-neural-network utilities "
            "for object detection, grouping the structures and bounding-box "
            "handling required to run and post-process specialized object "
            "localization models on top of the OpenCV DNN backend.",
        "classes": [
            ("class",  "cv::dnn_objdetect::InferBbox",
             "A class to post process model predictions."),
            ("struct", "cv::dnn_objdetect::object",
             "Structure to hold the details pertaining to a single bounding "
             "box."),
        ],
        "functions": [],
    },
    "quality": {
        "title": "Image Quality Analysis (IQA) API",
        "include": "opencv2/quality.hpp",
        "fn_include": "opencv2/quality/qualitybase.hpp",
        "description":
            "The quality module implements Image Quality Analysis (IQA) "
            "metrics. It provides algorithms to compute objective image-quality "
            "scores — full-reference metrics against a reference image, plus "
            "the no-reference BRISQUE metric — behind a common QualityBase "
            "interface.",
        "classes": [
            ("class", "cv::quality::QualityBase", ""),
            ("class", "cv::quality::QualityBRISQUE",
             "BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) is "
             "a No Reference Image Quality Assessment (NR-IQA) algorithm."),
            ("class", "cv::quality::QualityGMSD",
             "Full reference GMSD algorithm"),
            ("class", "cv::quality::QualityMSE",
             "Full reference mean square error algorithm  "
             "https://en.wikipedia.org/wiki/Mean_squared_error"),
            ("class", "cv::quality::QualityPSNR",
             "Full reference peak signal to noise ratio (PSNR) algorithm  "
             "https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio"),
            ("class", "cv::quality::QualitySSIM",
             "Full reference structural similarity algorithm  "
             "https://en.wikipedia.org/wiki/Structural_similarity"),
        ],
        "functions": [],
    },
    "reg": {
        "title": "Image Registration",
        "include": "opencv2/reg/map.hpp",
        "fn_include": "opencv2/reg/mapper.hpp",
        "description": (
            "The Registration module implements parametric image registration. "
            "The implemented method is direct alignment, that is, it uses "
            "directly the pixel values for calculating the registration between "
            "a pair of images, as opposed to feature-based registration.\n\n"
            "Feature based methods have some advantages over pixel based "
            "methods when we are trying to register pictures that have been "
            "shoot under different lighting conditions or exposition times, or "
            "when the images overlap only partially. On the other hand, the "
            "main advantage of pixel-based methods when compared to feature "
            "based methods is their better precision for some pictures (those "
            "shoot under similar lighting conditions and that have a "
            "significative overlap), due to the fact that we are using all the "
            "information available in the image, which allows us to achieve "
            "subpixel accuracy. This is particularly important for certain "
            "applications like multi-frame denoising or super-resolution.\n\n"
            "In fact, pixel and feature registration methods can complement "
            "each other: an application could first obtain a coarse "
            "registration using features and then refine the registration "
            "using a pixel based method on the overlapping area of the "
            "images.\n\n"
            "The module implements classes derived from the abstract classes "
            "[cv::reg::Map](classcv_1_1reg_1_1Map.md) or "
            "[cv::reg::Mapper](classcv_1_1reg_1_1Mapper.md). The former models "
            "a coordinate transformation between two reference frames, while "
            "the later encapsulates a way of invoking a method that calculates "
            "a Map between two images.\n\n"
            "Each class derived from Map implements a motion model, as "
            "follows:\n\n"
            "- [MapShift](classcv_1_1reg_1_1MapShift.md): Models a simple "
            "translation\n"
            "- [MapAffine](classcv_1_1reg_1_1MapAffine.md): Models an affine "
            "transformation\n"
            "- [MapProjec](classcv_1_1reg_1_1MapProjec.md): Models a projective "
            "transformation\n\n"
            "The classes derived from Mapper are:\n\n"
            "- [MapperGradShift](classcv_1_1reg_1_1MapperGradShift.md): "
            "Gradient based alignment for calculating translations.\n"
            "- [MapperGradEuclid](classcv_1_1reg_1_1MapperGradEuclid.md): "
            "Gradient based alignment for euclidean motions (rotations and "
            "translations).\n"
            "- [MapperGradSimilar](classcv_1_1reg_1_1MapperGradSimilar.md): "
            "Gradient based alignment for similarities (euclidean motion plus "
            "scaling).\n"
            "- [MapperGradAffine](classcv_1_1reg_1_1MapperGradAffine.md): "
            "Gradient based alignment for an affine motion model.\n"
            "- [MapperGradProj](classcv_1_1reg_1_1MapperGradProj.md): Gradient "
            "based alignment for calculating projective transformations.\n"
            "- [MapperPyramid](classcv_1_1reg_1_1MapperPyramid.md): Implements "
            "hierarchical motion estimation using a Gaussian pyramid."),
        "classes": [
            ("class", "cv::reg::Map",
             "Base class for modelling a Map between two images."),
            ("class", "cv::reg::MapAffine", ""),
            ("class", "cv::reg::Mapper",
             "Base class for modelling an algorithm for calculating a map."),
            ("class", "cv::reg::MapperGradAffine", ""),
            ("class", "cv::reg::MapperGradEuclid", ""),
            ("class", "cv::reg::MapperGradProj", ""),
            ("class", "cv::reg::MapperGradShift", ""),
            ("class", "cv::reg::MapperGradSimilar", ""),
            ("class", "cv::reg::MapperPyramid", ""),
            ("class", "cv::reg::MapProjec", ""),
            ("class", "cv::reg::MapShift", ""),
            ("class", "cv::reg::MapTypeCaster", ""),
        ],
        "functions": [],
    },
    "cannops": {
        "title": "Ascend-accelerated Computer Vision",
        "include": "opencv2/cann.hpp",
        "fn_include": "opencv2/cann_interface.hpp",
        "description": (
            "This module provides Ascend-accelerated implementations of core "
            "Computer-Vision operations, offloading element-wise arithmetic "
            "and image-processing primitives to Huawei Ascend NPU hardware for "
            "high-throughput acceleration."),
        "topics": ["Core part", "Operations for Ascend Backend."],
        "classes": [],
        "functions": [],
    },
}


def _fallback_fn_anchor(name: str, idx: int) -> str:
    """Stable in-page anchor for a fallback function's detail block."""
    return f"api-fn-{name}-{idx}"


def _fallback_topic_name(module: str, topic: str) -> str:
    """Docname/anchor stem for a topic subpage (e.g. datasets_action_recognition)."""
    return f"{module}_" + re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


def _fallback_class_refid(kind: str, qualified: str) -> str:
    """Doxygen-style page stem for a class/struct (e.g.
    `classcv_1_1datasets_1_1Dataset`) so the class link mirrors the convention."""
    pfx = kind if kind in ("class", "struct", "union") else "class"
    return pfx + qualified.replace("::", "_1_1")


def _render_fallback_body(name: str, d: dict) -> str:
    """Full group-page body for a module with no Doxygen XML, in Doxygen's group
    order: Topics, Detailed Description, Classes, Functions, Function
    Documentation. Every link is a working `#include` file page, a generated
    topic/class subpage, or an in-page anchor — so nothing 404s."""
    lines: list = []
    topics = d.get("topics") or []
    classes = d.get("classes") or []
    funcs = d.get("functions") or []
    if topics:
        lines += ["## Topics", ""]
        lines += [f"- [{t}](#api_{_fallback_topic_name(name, t)})"
                  for t in topics] + [""]
    if d.get("description"):
        lines += ["## Detailed Description", "", d["description"], ""]
    if classes:
        lines += ["## Classes", "", "{.api-reference-table}",
                  "| Name | Description |", "|---|---|"]
        for kind, qual, brief in classes:
            page = _fallback_class_refid(kind, qual)
            link = f"[`{kind} {qual}`]({page}.md)"
            # "More…" links to the class page's detail, as on the real pages —
            # only where there's a brief (matching Doxygen).
            desc = _md_escape_cell(brief)
            if brief:
                desc += f" [More…]({page}.md#detailed-description)"
            lines.append(f"| {link} | {desc or chr(0xa0)} |")
        lines += [""]
    if funcs:
        lines += ["## Functions", "",
                  "{.api-reference-table .api-function-table}",
                  "| Return | Name | Description |", "|---|---|---|"]
        for i, (ret, qual, _args, brief) in enumerate(funcs):
            anc = _fallback_fn_anchor(name, i)
            lines.append(f"| `{_md_escape_cell(ret)}` | "
                         f"[`{qual}`](#{anc}) | {_md_escape_cell(brief)} |")
        lines += [""]
    if funcs:
        lines += ["## Function Documentation", ""]
        fn_inc = d.get("fn_include") or d.get("include")
        href = _include_page_href(fn_inc) if fn_inc else None
        for i, (ret, qual, args, brief) in enumerate(funcs):
            short = qual.rsplit("::", 1)[-1]
            lines += [f"({_fallback_fn_anchor(name, i)})=",
                      f"### {short}()", "",
                      "{.opencv-api-sig}", f"`{ret} {qual}{args}`", ""]
            if href:
                lines += [
                    "{.opencv-api-include}",
                    f'<code class="docutils literal notranslate">'
                    f'#include &lt;<a class="reference external '
                    f'opencv-include-link" href="{href}">{fn_inc}</a>&gt;</code>',
                    ""]
            if brief:
                lines += [brief, ""]
    # Hidden toctree registers the topic + class subpages in the sidebar nav
    # (and avoids "not in any toctree" warnings); the lists above link to them.
    toc = [_fallback_topic_name(name, t) for t in topics]
    toc += [_fallback_class_refid(k, q) for k, q, _b in classes]
    if toc:
        lines += ["```{toctree}", ":hidden:", ""] + toc + ["```", ""]
    return "\n".join(lines)


def _fallback_module_tree(name: str):
    """In-memory stand-in for a module group whose Doxygen XML is missing.
    Carries `body_md` (the full hand-rendered page body); `_write_api_stub`
    emits it verbatim so the page has Topics/Classes/Functions/Function
    Documentation with working links, and its `{#api_<name>}` anchor resolves
    for any reference to the module. `topic_children` are the topic subpages
    `_write_api_stub` also writes. Other fields keep the node-schema shape so
    `module_rows`/`_flatten`/the index treat it like a real module.

    No top-of-page `#include` — like the real Doxygen group page, the include
    belongs in the Function Documentation (where it links to the file page)."""
    d = _FALLBACK_MODULE_DATA.get(name)
    if d is None:
        return None
    title = d["title"]
    back = f"Part of the [{title}](#api_{name}) module."
    child_pages: list = []
    # Topic subpages (linked from the Topics list).
    for t in (d.get("topics") or []):
        cn = _fallback_topic_name(name, t)
        child_pages.append((cn, "\n".join(
            [f"# {t} {{#api_{cn}}}", "", back, ""])))
    # Class/struct subpages (linked from the Classes table; the brief lives
    # under a Detailed Description heading so it reads like a class page).
    for kind, qual, brief in (d.get("classes") or []):
        crefid = _fallback_class_refid(kind, qual)
        child_pages.append((crefid, "\n".join(
            [f"# {kind.capitalize()} {qual}", "",
             (brief or ""), "",
             "## Detailed Description", "",
             (brief or f"`{qual}` reference."), "", back, ""])))
    return {
        "name": name,            # single-underscore group name -> page & anchor
        "title": title,
        "detailed": "",
        "innerclasses": [],
        "sections": {},
        "children": [],
        "child_pages": child_pages,
        "body_md": _render_fallback_body(name, d),
    }


def _apply_group_doc_override(tree: dict, xml_dir: pathlib.Path) -> None:
    """Repair module groups whose subgroups were `@addtogroup`'d but never nested
    under a titled `@defgroup`, which Doxygen leaves as orphan top-level groups
    with an empty landing page. Injects the group description and attaches the
    real subgroups as children (recursively retitled). Driven by
    `_GROUP_DOC_OVERRIDES` / `_GROUP_TITLE_OVERRIDES`; no-op without an override."""
    ov = _GROUP_DOC_OVERRIDES.get(tree["name"])
    if not ov:
        return
    if ov.get("detailed") and not tree["detailed"]:
        tree["detailed"] = ov["detailed"]
    have = {c["name"] for c in tree["children"]}
    for sub in ov.get("subgroups", ()):
        if sub in have:
            continue
        child = _build_api_hierarchy("group__" + sub.replace("_", "__"), xml_dir)
        if child is not None:
            tree["children"].append(child)

    def _retitle(node: dict) -> None:
        node["title"] = _GROUP_TITLE_OVERRIDES.get(node["name"], node["title"])
        for c in node.get("children", ()):
            _retitle(c)
    _retitle(tree)


def _harvest_namespace_into_group(tree: dict, xml_dir: pathlib.Path) -> None:
    # Pull funcs/classes under the module's include prefix (per _GROUP_NS_HARVEST)
    # into the otherwise-empty group node, from the cv namespace and any group
    # they were filed under (e.g. depth.hpp uses @addtogroup rgbd).
    import xml.etree.ElementTree as _ET
    prefix = _GROUP_NS_HARVEST.get(tree["name"])
    if not prefix:
        return
    tree.setdefault("sections", {})
    have = {c["refid"] for c in tree["innerclasses"]}

    def _ingest(cd):
        for title, members in _parse_member_sections(cd).items():
            seen = {m["id"] for m in tree["sections"].get(title, [])}
            kept = [m for m in members
                    if (m.get("include_file") or "").startswith(prefix)
                    and m["id"] not in seen]
            if kept:
                tree["sections"].setdefault(title, []).extend(kept)
        for ic in cd.findall("innerclass"):
            rid = ic.get("refid", "")
            if ic.get("prot") != "public" or not rid or rid in have:
                continue
            cx = xml_dir / f"{rid}.xml"
            if not cx.is_file():
                continue
            try:
                ccd = _ET.parse(cx).getroot().find("compounddef")
            except _ET.ParseError:
                continue
            loc = ccd.find("location") if ccd is not None else None
            if loc is None or not (loc.get("file") or "").startswith(prefix):
                continue
            qualified = " ".join((ic.text or "").split())
            tree["innerclasses"].append({
                "refid": rid, "name": qualified, "qualified": qualified,
                "kind": "struct" if rid.startswith("struct") else "class",
                "brief": _read_class_brief(rid, xml_dir),
            })
            have.add(rid)

    srcs = [xml_dir / "namespacecv.xml"]
    srcs += [g for g in xml_dir.glob("group__*.xml")
             if prefix in g.read_text(encoding="utf-8", errors="ignore")]
    for s in srcs:
        if not s.is_file():
            continue
        try:
            cd = _ET.parse(s).getroot().find("compounddef")
        except _ET.ParseError:
            continue
        if cd is not None:
            _ingest(cd)


def _generate_api_stubs(modules, xml_dir, out_dir,
                        root_anchor="api_root", root_title="API Reference",
                        root_desc=None, extra_groups=()):
    """Generate a stub tree (group/namespace/class pages) under out_dir.

    Docnames are prefixed with out_dir.name so the same generator drives both
    main_modules/ and extra_modules/ (contrib) trees from separate calls.
    extra_groups: orphan group stems rendered as pages but kept out of the
    module index (referenced groups not nested under any module top)."""
    if not modules and not extra_groups:
        return
    if not xml_dir.is_dir():
        return  # No XML yet; degrade silently.

    global _CUR_MODULE_DIR
    _CUR_MODULE_DIR = out_dir.name
    _doc_prefix = out_dir.name

    # Where the member renderers find legacy graph SVGs / write their variants.
    global _DOXY_HTML_ROOT, _API_OUT_DIR
    _DOXY_HTML_ROOT = xml_dir.parent / "html"
    _API_OUT_DIR = out_dir

    # Freshness guard: skip rebuild only if the tree is newer than BOTH the XML
    # and the generator code. Without the code check, editing these modules
    # never invalidates the cache (the XML is unchanged), so `make sphinx`
    # silently keeps stale stubs and edits appear to have no effect.
    src_index = xml_dir / "index.xml"
    root_md = out_dir / "api_root.markdown"
    _code_mtime = max(
        (p.stat().st_mtime for p in pathlib.Path(__file__).parent.glob("*.py")),
        default=0.0)
    if (src_index.is_file() and root_md.is_file()
            and root_md.stat().st_mtime >= src_index.stat().st_mtime
            and root_md.stat().st_mtime >= _code_mtime
            and any(p.name.startswith("namespace_") and p.suffix == ".md"
                    for p in out_dir.iterdir())):
        for stub in out_dir.iterdir():
            n = stub.name
            if n.endswith(".md") and (n.startswith("class") or n.startswith("struct")):
                _ANCHOR_TO_DOC[n[:-3]] = f"{_doc_prefix}/{n[:-3]}"
        return

    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global _stub_written, _MERGED_GROUPS, _MERGED_CLASS_TO_GROUP
    _stub_written = set()
    _MERGED_GROUPS = []
    _MERGED_CLASS_TO_GROUP = {}
    _desc = root_desc or (
        "Sphinx-rendered API reference. Each entry below is a module's "
        "umbrella `@defgroup`; sub-pages mirror the Doxygen subgroup hierarchy.")
    root_lines = [
        f"{root_title} {{#{root_anchor}}}",
        "=" * len(root_title),
        "",
        _desc,
        "",
    ]
    classes_seen: dict[str, dict] = {}
    global_group_info: dict[str, dict] = {}
    global_ns_group_map: dict[str, set] = {}
    trees: list = []
    module_rows: list = []  # (folder, page_stem, title) for the api_root list
    _gapi_tree = None  # saved to write gapi.md wrapper after all stubs
    for m in list(modules) + list(extra_groups):
        is_extra = m in extra_groups
        if m in _GROUP_OVERRIDE_SUBGROUPS:
            # Re-parented under its module by _apply_group_doc_override; skip so
            # it isn't also emitted as a standalone orphan page.
            continue
        stem = _module_group_stem(m)
        tree = _build_api_hierarchy("group__" + stem.replace("_", "__"), xml_dir)
        if tree is None:
            # No Doxygen XML for this module — inject a placeholder tree for the
            # known optional modules so they still render & link; else skip.
            tree = _fallback_module_tree(m)
        if tree is None:
            continue
        _apply_group_doc_override(tree, xml_dir)
        _harvest_namespace_into_group(tree, xml_dir)
        trees.append(tree)
        if is_extra:
            pass
        elif m == "gapi":
            # gapi: expose as top-level "Graph API" wrapper page, not "G-API framework"
            _gapi_tree = tree
            module_rows.append((m, "gapi", "Graph API"))
        else:
            module_rows.append((m, tree["name"], tree["title"]))
        all_group_names = _collect_all_group_names(tree)
        all_refids = ["group__" + n.replace("_", "__") for n in all_group_names]
        for ns_name, grps in _build_ns_group_map(all_refids, xml_dir).items():
            global_ns_group_map.setdefault(ns_name, set()).update(grps)
        def _flatten(node: dict, parent: str | None) -> None:
            global_group_info[node["name"]] = {"title": node["title"], "parent": parent}
            for child in node.get("children", []):
                _flatten(child, node["name"])
        _flatten(tree, None)
    # Pass 2 — write namespace stubs (once each) then group stubs.
    written_ns: set[str] = set()
    for tree in trees:
        all_group_names = _collect_all_group_names(tree)
        ns_map: dict[str, list] = {}
        for group_name in all_group_names:
            for ns in _namespaces_for_group(group_name, xml_dir, global_ns_group_map):
                anchor = f"api_ns_{ns['name'].replace('::', '__')}"
                if ns["name"] not in written_ns:
                    _write_namespace_stub(ns, out_dir, xml_dir,
                                          global_ns_group_map, global_group_info,
                                          classes_seen)
                    written_ns.add(ns["name"])
                    _ALL_NAMESPACES[ns["name"]] = {
                        "refid": ns.get("refid", ""),
                        "brief": ns.get("brief", ""),
                        "docname": f"{_doc_prefix}/namespace_"
                                   f"{ns['name'].replace('::', '__')}",
                    }
                ns_map.setdefault(group_name, []).append((ns["name"], anchor))
        _write_api_stub(tree, out_dir, classes_seen, ns_map)
    # Expand to NESTED classes (a class's own inner types, e.g.
    # cv::ImageCollection::iterator) so each gets its own page like Doxygen —
    # the group-level collection skips names containing `::`.
    import xml.etree.ElementTree as _ET_nested
    _queue = list(classes_seen.keys())
    while _queue:
        _rid = _queue.pop()
        _cx = xml_dir / f"{_rid}.xml"
        if not _cx.is_file():
            continue
        try:
            _ccd = _ET_nested.parse(str(_cx)).getroot().find("compounddef")
        except _ET_nested.ParseError:
            _ccd = None
        if _ccd is None:
            continue
        for _ic in _ccd.findall("innerclass"):
            _icid = _ic.get("refid", "")
            if not _icid or _icid in classes_seen:
                continue
            _icname = (_ic.text or "").strip()
            _ickind = ("struct" if _icid.startswith("struct")
                       else "union" if _icid.startswith("union") else "class")
            classes_seen[_icid] = {"refid": _icid, "name": _icname,
                                   "qualified": _icname, "kind": _ickind}
            _queue.append(_icid)
    # Pre-seed every class's docname BEFORE writing, so cross-references made
    # while rendering one page (e.g. "inherited from <base>" links) resolve even
    # when the base class is written later in this same pass.
    for _cls in classes_seen.values():
        _ANCHOR_TO_DOC.setdefault(
            _cls["refid"], f"{_doc_prefix}/{_class_page_name(_cls['refid'])}")
    # Merged shell groups: a Classes box then each class inline. Done after
    # seeding above so bases resolve for "inherited from" links.
    for _mout, _mlines, _mclasses in _MERGED_GROUPS:
        _lines = list(_mlines)
        # Slug matching the per-class `## heading` its box entry links to.
        def _cls_slug(_c):
            _h = f"{_c['kind'].title()} {_c.get('qualified') or _c['name']}"
            return re.sub(r"[^a-z0-9]+", "-", _h.lower()).strip("-")
        _lines += ["## Classes", "", "{.api-reference-table}",
                   "| Name | Description |", "|---|---|"]
        for _c in _mclasses:
            # Encode "::" so the cv-linkifier doesn't nest an <a> in our link.
            _nm = _c["name"].replace("::", "&#58;&#58;")
            _lines.append(
                f'| <a href="#{_cls_slug(_c)}"><code>{_c["kind"]} '
                f'{_nm}</code></a> '
                f"| {_md_escape_cell(_c.get('brief', ''))} |")
        _lines.append("")
        for _c in _mclasses:
            _q = _c.get("qualified") or _c["name"]
            _lines += [f"## {_c['kind'].title()} {_q}", ""]
            _lines += _write_class_stub(_c, out_dir, xml_dir, inline=True) or []
        _stub_write(_mout, "\n".join(_lines) + "\n")
    # Per-class pages; seed `_ANCHOR_TO_DOC` refid→docname for `@ref`.
    for cls in classes_seen.values():
        _grp = _MERGED_CLASS_TO_GROUP.get(cls["refid"])
        if _grp:
            # Inlined into its group page: turn this page into a redirect and
            # aim refid xrefs at the group instead.
            _rel = _grp.split("/", 1)[-1]
            _stub_write(
                out_dir / f"{_class_page_name(cls['refid'])}.md",
                f"# {cls.get('qualified') or cls['name']}\n\n"
                f'<script>window.location.replace("{_rel}.html"'
                f' + window.location.hash);</script>\n\n'
                f'<p>This page has moved to '
                f'<a href="{_rel}.html">{_rel}</a>.</p>\n')
            _ANCHOR_TO_DOC[cls["refid"]] = _grp
            _ALL_CLASSES[cls["refid"]] = {
                "qualified": cls.get("qualified") or cls.get("name", ""),
                "kind": cls.get("kind", "class"),
                "brief": cls.get("brief", ""),
                "docname": _grp,
            }
            continue
        _write_class_stub(cls, out_dir, xml_dir)
        _docname = f"{_doc_prefix}/{_class_page_name(cls['refid'])}"
        _ANCHOR_TO_DOC[cls["refid"]] = _docname
        _ALL_CLASSES[cls["refid"]] = {
            "qualified": cls.get("qualified") or cls.get("name", ""),
            "kind": cls.get("kind", "class"),
            "brief": cls.get("brief", ""),
            "docname": _docname,
        }
    # Placeholder stubs for bare template params (_Tp, …) so diagram links resolve.
    _write_placeholder_stubs(out_dir, xml_dir)

    # Write standalone "Graph API" page for gapi module (replaces gapi_ref as entry)
    if _gapi_tree is not None:
        content = _read_gapi_full_content()
        # Collect all gapi subgroup page names for the hidden toctree
        all_gapi_names = _collect_all_group_names(_gapi_tree)
        gapi_toctree = "\n".join(all_gapi_names)
        gapi_lines = ["# Graph API {#api_gapi}", ""]
        if content:
            gapi_lines += [content, ""]
        gapi_lines += [
            "```{toctree}", ":hidden:", ":maxdepth: 2", "",
            gapi_toctree,
            "```", "",
        ]
        _stub_write(out_dir / "gapi.md", "\n".join(gapi_lines) + "\n")

    # File-reference pages (include dependency graphs) so #include links land on
    # a Sphinx diagram page.
    _write_file_ref_stubs(out_dir, xml_dir)
    # Hidden toctree drives nav/sidebar; the visible list shows "folder. Title".
    root_lines += ["```{toctree}", ":hidden:", ":maxdepth: 1", ""]
    root_lines += [stem for _m, stem, _t in module_rows]
    root_lines += ["```", ""]
    for _m, stem, title in module_rows:
        root_lines.append(f"- {_m}. [{title}]({stem}.md)")
    root_lines.append("")
    _stub_write(out_dir / "api_root.markdown", "\n".join(root_lines) + "\n")
    # Sweep stale files.
    for _p in list(out_dir.iterdir()):
        if _p.is_file() and _p not in _stub_written:
            _p.unlink()
    # Flush per-sample example pages now to avoid orphans.
    _generate_example_pages(out_dir.parent / "examples")
