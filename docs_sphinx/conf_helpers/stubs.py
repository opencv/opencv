# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""API-reference stub writers. Entry point: ``_generate_api_stubs``."""
from __future__ import annotations
import pathlib, os as _os, shutil as _shutil, textwrap as _textwrap
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


def _stub_write(path: pathlib.Path, content: str) -> None:
    """Write only if changed; mark path live for this run."""
    if not (path.is_file() and path.read_text(encoding="utf-8") == content):
        path.write_text(content, encoding="utf-8")
    _stub_written.add(path)

_DOXY_HTML_ROOT: pathlib.Path | None = None
_API_OUT_DIR: pathlib.Path | None = None


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
                          group_info: dict | None = None) -> tuple[str, str]:
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
            lines.append(f"| [`{ic_kind} {short_name}`]({page}.md) |")
        lines.append("")

    # Member summary tables.
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = ns_sections.get(section_title, [])
        if not items:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        if section_title == "Functions":
            lines += ["{.api-function-table}", "| Return | Name |", "|---|---|"]
            for m in items:
                ret_md = _type_to_md(m.get("type_elem"))
                if not ret_md:
                    ret_md = _md_escape_cell(m["type"]) or "\u00a0"
                if m.get("static"):
                    ret_md = "static " + ret_md
                # Multi-line, one-param-per-line signature (matching the detail
                # block); return type stays in its own cell, so head = name.
                label = _func_sig_md(m["name"], m.get("params_sig"))
                lines.append(f"| {ret_md} | [{label}](#{m['id']}) |")
        elif section_title in ("Typedefs", "Variables"):
            for m in items:
                lines.append("```cpp")
                lines.append(f"typedef {m['type']} {m['name']}" if section_title == "Typedefs"
                              else f"{m['type']} {m['name']}")
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


def _write_api_stub(node: dict, out_dir: pathlib.Path,
                    classes_seen: dict, ns_map: dict | None = None) -> None:
    """Write one .md per group node, recursing into children.

    Parent groups → @subpage index pages; leaf groups → summary tables + detail
    blocks. Inner classes populate `classes_seen` for later page emission."""
    name = node["name"]
    title = node["title"]
    out = out_dir / f"{name}.md"

    lines = [f"# {title} {{#api_{name}}}", ""]

    if node["children"]:
        lines += ["## Topics", ""]
        for child in node["children"]:
            lines.append(f"- @subpage api_{child['name']}")
        lines.append("")

    # Detailed Description heading — shown even when empty (ayush's layout).
    if node["detailed"]:
        lines += ["## Detailed Description", "", node["detailed"], ""]
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

    _rich_return = (name == "core_basic")

    def _summary_block(section_title: str, members: list) -> list[str]:
        out: list[str] = []
        if section_title == "Functions":
            out += ["{.api-reference-table .api-function-table}",
                    "| Return | Name | Description |", "|---|---|---|"]
            for m in members:
                ret_type = _md_escape_cell(m["type"])
                label = _func_sig_md(m["name"], m.get("params_sig"))
                sig_link = _member_anchor_link(m, label, raw=True)
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
                name_link = _member_anchor_link(m, f"cv::{m['name']}")
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
            import html as _html_mod
            # Encode `::` so translate's cv-linkifier skips it.
            def _safe(s: str) -> str:
                return _html_mod.escape(s).replace("::", "&#58;&#58;")
            for m in members:
                _anchor = (m.get("name") or "").lower() or m["id"]
                _href = f"#{_anchor}"
                _qual = m["qualified"] or m["name"]
                _is_strong = bool(m.get("strong"))
                _keyword = "enum struct" if _is_strong else "enum"
                # Enumerator name prefix (scope).
                if _is_strong:
                    _val_prefix = _qual + "::"
                elif "::" in _qual:
                    _val_prefix = _qual.rsplit("::", 1)[0] + "::"
                else:
                    _val_prefix = ""
                out.append(
                    '<div class="highlight-cpp notranslate '
                    'opencv-enum-clickable"><div class="highlight"><pre>'
                )
                # Anonymous enums have no name to link; emit a bare `enum {`.
                _name_html = (
                    f'<a class="reference internal" href="{_href}">'
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
                    out.append(
                        f'    <a class="reference internal" href="{_href}">'
                        f'<span class="n">{_safe(_full)}</span></a>'
                        f'{_init}{_comma}'
                    )
                out.append('<span class="p">}</span></pre></div></div>')
                # Blank line closes the raw-HTML block (CommonMark rule 7).
                out.append("")
                _details = (f'<a class="reference internal" '
                            f'href="{_href}">View details</a>')
                if m["brief"]:
                    out.append(f'{_md_escape_cell(m["brief"])} {_details}')
                else:
                    out.append(_details)
                out.append("")
        else:  # Macros
            out += ["{.api-reference-table}", "| Name | Description |", "|---|---|"]
            for m in members:
                name_link = _member_anchor_link(m, m["name"])
                out.append(f"| {name_link} | {_md_escape_cell(m['brief'])} |")
        return out

    _named_groups: list[tuple[str, str, list]] = []   # (header, section_title, members)
    for _, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        ungrouped = [m for m in items if not (m.get("section_header") or "")]
        for _hdr, _members in _group_by_section_header(
                [m for m in items if (m.get("section_header") or "")]):
            _named_groups.append((_hdr, section_title, _members))
        if ungrouped:
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
    # `{doxygendefine}`; enum detail is hand-rolled (core_basic only).
    seen_define_names: set[str] = set()
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        _core_basic_funcs = (name == "core_basic" and kind_key == "function")
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
                # Hand-rolled (breathe's {doxygenenum} drops initializers/briefs).
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
                    blk = [f'<h3 id="{m["id"]}">{_keyword}</h3>', ""]
                if m.get("include_file"):
                    _einc = m["include_file"]
                    _eifile = _FILE_URL.get(_einc)
                    if _eifile:
                        blk += [
                            "{.opencv-api-include}",
                            f'<code class="docutils literal notranslate">'
                            f'#include &lt;<a class="reference external '
                            f'opencv-include-link" '
                            f'href="../../../doc/doxygen/html/{_eifile}">'
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
    blob or breaks out past the card boundary as a flat sibling list."""
    if not desc:
        return [f"- `{nm}`"]
    lines = desc.split("\n")
    out = [f"- `{nm}` — {lines[0]}"]
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
    out = ["```{list-table}", ":header-rows: 0",
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
        sig_lines = [f"typedef {typ} {full_name}".strip()]
    else:  # variable / attribute — append the `= value` initializer if present.
        decl = f"{prefix}{typ + ' ' if typ else ''}{full_name}".strip()
        init = (m.get("initializer") or "").strip()
        sig_lines = [f"{decl} {init}".strip() if init else decl]
    # Template clause + declaration as inline code (keeps token-linkifier
    # active); `{.opencv-api-sig}` lets the CSS preserve the alignment spaces.
    _sig = ([f"`{tmpl}`"] if tmpl else []) + [f"`{ln}`" for ln in sig_lines]
    out += ["{.opencv-api-sig}", "\\\n".join(_sig), ""]

    inc = (m.get("include_file") or "").strip()
    if inc:
        _ifile = _FILE_URL.get(inc)
        if _ifile:
            _href = f"../../../doc/doxygen/html/{_ifile}"
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
    _sig = ([f"`{m['template']}`"] if m.get("template") else []) + \
        [f"`{ln}`" for ln in sig_lines]
    out += ["{.opencv-api-sig}", "\\\n".join(_sig), ""]
    if m.get("include_file"):
        _ipath = m["include_file"]
        _ifile = _FILE_URL.get(_ipath)
        if _ifile:
            _href = f"../../../doc/doxygen/html/{_ifile}"
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
                      xml_dir: pathlib.Path) -> None:
    """One .md per inner class, mirroring Doxygen's class-page layout.

    Falls back to `{doxygenclass}`/`{doxygenstruct}` if class XML can't be read."""
    page = _class_page_name(cls["refid"])
    out = out_dir / f"{page}.md"
    qualified = cls["qualified"] or cls["name"]
    kind_label = cls["kind"].title()
    title = f"{kind_label} {qualified}"
    # No `{#refid}` anchor; `_generate_api_stubs` seeds `_ANCHOR_TO_DOC` instead.
    lines = [f"# {title}", ""]

    # Class-page header: brief + `View details` + `#include` line.
    _header_data = _read_class_data(cls["refid"], xml_dir)
    if _header_data is not None:
        import html as _html_pkg
        _brief = (_header_data.get("brief") or "").strip()
        if _brief:
            # Link only when there's a detailed description to jump to.
            _more = (
                ' <a class="opencv-class-more" href="#detailed-description">View details</a>'
                if _header_data.get("detailed") else ""
            )
            lines.append(
                f'<p class="opencv-class-brief">'
                f'{_html_pkg.escape(_brief)}{_more}</p>'
            )
            lines.append("")
        _inc = (_header_data.get("include") or "").strip()
        if _inc:
            _cifile = _FILE_URL.get(_inc)
            if _cifile:
                _inc_code = (
                    f'#include &lt;<a class="reference external '
                    f'opencv-include-link" '
                    f'href="../../../doc/doxygen/html/{_cifile}">'
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

    # 1) Summary tables in Doxygen's order.
    for sd_kind, summary_title in _CLASS_SUMMARY_SECTIONS:
        items = data["sections"].get(sd_kind, [])
        if not items:
            continue
        lines.append(f"## {summary_title}")
        lines.append("")
        non_enum_items = [m for m in items if m["kind"] != "enum"]
        enum_items = [m for m in items if m["kind"] == "enum"]
        if non_enum_items:
            lines += ["{.api-reference-table .api-function-table}",
                      "| Return | Name | Description |", "|---|---|---|"]
            for m in non_enum_items:
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
        _ifile = _FILE_URL.get(_src_inc)
        if _ifile:
            _flink = (f'{_html_pkg2.escape(_dir)}<a class="reference external '
                      f'opencv-include-link" '
                      f'href="../../../doc/doxygen/html/{_ifile}">'
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


def _generate_api_stubs(modules, xml_dir, out_dir,
                        root_anchor="api_root", root_title="API Reference",
                        root_desc=None):
    """Generate a stub tree (group/namespace/class pages) under out_dir.

    Docnames are prefixed with out_dir.name so the same generator drives both
    main_modules/ and extra_modules/ (contrib) trees from separate calls."""
    if not modules:
        return
    if not xml_dir.is_dir():
        return  # No XML yet; degrade silently.

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
    global _stub_written
    _stub_written = set()
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
    for m in modules:
        stem = _module_group_stem(m)
        tree = _build_api_hierarchy("group__" + stem.replace("_", "__"), xml_dir)
        if tree is None:
            continue
        trees.append(tree)
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
                                          global_ns_group_map, global_group_info)
                    written_ns.add(ns["name"])
                    _ALL_NAMESPACES[ns["name"]] = {
                        "refid": ns.get("refid", ""),
                        "brief": ns.get("brief", ""),
                        "docname": f"{_doc_prefix}/namespace_"
                                   f"{ns['name'].replace('::', '__')}",
                    }
                ns_map.setdefault(group_name, []).append((ns["name"], anchor))
        _write_api_stub(tree, out_dir, classes_seen, ns_map)
    # Per-class pages; seed `_ANCHOR_TO_DOC` refid→docname for `@ref`.
    for cls in classes_seen.values():
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
