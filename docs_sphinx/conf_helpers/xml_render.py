# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""Doxygen XML -> Markdown primitives for the API-reference stubs."""
from __future__ import annotations
import copy as _copy
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *


def _wrap_emphasis(inner: str, delim: str) -> str:
    """Wrap `inner` in a `*`/`**` run, hoisting any leading/trailing whitespace
    outside the markers. CommonMark needs the delimiters to hug the text
    (`**x**`, not `** x **`), but Doxygen keeps the spaces inside `<b> … </b>`."""
    stripped = inner.strip()
    if not stripped:
        return inner
    lead = inner[:len(inner) - len(inner.lstrip())]
    trail = inner[len(inner.rstrip()):]
    return f"{lead}{delim}{stripped}{delim}{trail}"

_AMS_BLOCK_ENVS = frozenset((
    "align", "align*", "alignat", "alignat*", "flalign", "flalign*",
    "gather", "gather*", "multline", "multline*",
    "equation", "equation*", "eqnarray", "eqnarray*",
))


def _render_formula(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("\\[") and s.endswith("\\]"):
        inner = s[2:-2].strip()
        m = re.match(r"^\\begin\{([a-zA-Z]+\*?)\}", inner)
        if (m and m.group(1) in _AMS_BLOCK_ENVS
                and inner.rstrip().endswith("\\end{%s}" % m.group(1))):
            return f"\n\n{inner}\n\n"
        return f"\n\n$$\n{inner}\n$$\n\n"
    return s


def _render_image(node) -> str:
    if node.get("type") != "html":
        return ""
    name = (node.get("name") or "").strip()
    caption = "".join(node.itertext()).strip()
    hit = _IMAGE_INDEX.get(name)
    if not hit:
        return ""
    return f"![{caption}](/{hit})"


def _itertext(el) -> str:
    if el is None:
        return ""
    parts: list[str] = []

    def _walk(node) -> None:
        if node.tag == "formula":
            parts.append(_render_formula(node.text or ""))
        elif node.tag == "image":
            parts.append(_render_image(node))   # caption text not recursed
        else:
            if node.text:
                parts.append(node.text)
            for child in node:
                _walk(child)
                if child.tail:
                    parts.append(child.tail)

    _walk(el)
    return "".join(parts).strip()


# memberdef@kind -> section title; order mirrors Doxygen group page.
_MEMBERDEF_SECTIONS = (
    ("typedef",  "Typedefs"),
    ("enum",     "Enumerations"),
    ("function", "Functions"),
    ("variable", "Variables"),
    ("define",   "Macros"),
)


def _read_class_brief(refid: str, xml_dir: pathlib.Path,
                      _cache: dict = {}) -> str:
    """Read brief description from a class/struct compound XML; cached."""
    if refid in _cache:
        return _cache[refid]
    import xml.etree.ElementTree as _ET
    xml_path = xml_dir / f"{refid}.xml"
    brief = ""
    if xml_path.is_file():
        try:
            ccd = _ET.parse(xml_path).getroot().find("compounddef")
            if ccd is not None:
                brief = _itertext(ccd.find("briefdescription"))
        except _ET.ParseError:
            pass
    _cache[refid] = brief
    return brief


def _member_template(md) -> str:
    """`template<...>` prefix for a memberdef."""
    tpl = md.find("templateparamlist")
    if tpl is None:
        return ""
    parts = []
    for p in tpl.findall("param"):
        t = _itertext(p.find("type"))
        d = (p.findtext("declname") or "").strip()
        parts.append(f"{t} {d}".strip() if d else t)
    return f"template<{', '.join(parts)}>" if parts else ""


def _member_detail_parts(md):
    de = md.find("detaileddescription")
    if de is None:
        return "", [], ""
    params, returns = [], ""
    for para in de.findall("para"):
        for pl in para.findall("parameterlist"):
            if pl.get("kind") in ("param", "templateparam"):
                for it in pl.findall("parameteritem"):
                    nm = ", ".join(
                        t for t in (_itertext(n) for n in
                                    it.findall(".//parametername")) if t)
                    # Block-aware: a description carrying an <itemizedlist>
                    # (e.g. calibration `flags`) keeps its bullets as real
                    # Markdown instead of collapsing into a run-on paragraph.
                    d = _doxygen_desc_to_md(it.find("parameterdescription"))
                    if nm:
                        params.append((nm, d))
        for ss in para.findall("simplesect"):
            if ss.get("kind") == "return":
                returns = _itertext(ss)
    # Prune the param/return chrome (rendered separately) then convert the rest
    # with full block support so lists and notes survive. Preserve tail text when
    # removing elements so descriptions after parameterlist are not lost.
    pruned = _copy.deepcopy(de)
    for para in pruned.findall("para"):
        for child in list(para):
            if child.tag == "parameterlist":
                # Preserve tail text (text after the element) by moving it to previous sibling
                if child.tail:
                    prev_idx = list(para).index(child) - 1
                    if prev_idx >= 0:
                        prev_sibling = para[prev_idx]
                        prev_sibling.tail = (prev_sibling.tail or "") + child.tail
                    else:
                        # No previous sibling, prepend to para text
                        para.text = (para.text or "") + child.tail
                para.remove(child)
            elif (child.tag == "simplesect"
                  and child.get("kind") in ("param", "templateparam", "return")):
                # Same tail preservation for simplesect
                if child.tail:
                    prev_idx = list(para).index(child) - 1
                    if prev_idx >= 0:
                        prev_sibling = para[prev_idx]
                        prev_sibling.tail = (prev_sibling.tail or "") + child.tail
                    else:
                        para.text = (para.text or "") + child.tail
                para.remove(child)
    return _doxygen_desc_to_md(pruned), params, returns


# I/O proxy types as a variable type = Doxygen phantom; drop it.
_PHANTOM_VAR_TYPE_RE = re.compile(
    r"\b(?:Input|Output|InputOutput)Array(?:OfArrays)?\b")


def _is_phantom_variable(member: dict) -> bool:
    return (member.get("kind") == "variable"
            and bool(_PHANTOM_VAR_TYPE_RE.search(member.get("type") or "")))


def _parse_member_sections(cd) -> dict[str, list[dict]]:
    """Parse compounddef memberdefs into `{section_title: [member dict]}`."""
    sections: dict[str, list[dict]] = {}
    _seen_member_ids: set[str] = set()
    for sd in cd.findall("sectiondef"):
        # Doxygen `@name` member-group title; carried onto each member.
        sd_header = (sd.findtext("header") or "").strip()
        for md in sd.findall("memberdef"):
            kind = md.get("kind", "")
            section_title = dict(_MEMBERDEF_SECTIONS).get(kind)
            if not section_title:
                continue
            _mid = md.get("id", "")
            if _mid and _mid in _seen_member_ids:
                continue
            _seen_member_ids.add(_mid)
            qualified = (md.findtext("qualifiedname") or "").strip()
            if not qualified:
                qualified = (md.findtext("name") or "").strip()
            # Merge <array> suffix into <type> or breathe drops it.
            def _param_type(p) -> str:
                t = _itertext(p.find("type"))
                arr = (p.findtext("array") or "").strip()
                return (t + arr) if arr else t
            param_types = [_param_type(p) for p in md.findall("param")]
            # (type, name, default) per parameter — drives the multi-line,
            # column-aligned function signature in the detail card.
            def _param_sig(p) -> tuple:
                return (_param_type(p),
                        (p.findtext("declname") or "").strip(),
                        _itertext(p.find("defval")))
            params_sig = [_param_sig(p) for p in md.findall("param")]
            # Function-like macro params carry only a <defname> (no type/declname).
            macro_params = ([(p.findtext("defname") or "").strip()
                             for p in md.findall("param")] if kind == "define"
                            else [])
            # `= value` for a variable; the macro body for a define.
            initializer = _itertext(md.find("initializer"))
            enum_values = []
            is_strong = md.get("strong", "no") == "yes"
            if kind == "enum":
                for ev in md.findall("enumvalue"):
                    enum_values.append({
                        "name":        (ev.findtext("name") or "").strip(),
                        "initializer": (ev.findtext("initializer") or "").strip(),
                        # Per-enumerator brief for the detail block.
                        "brief":       _enum_value_desc(ev),
                    })
            _dtl, _params, _returns = _member_detail_parts(md)
            _loc = md.find("location")
            _include_file = _normalize_include(
                (_loc.get("file") if _loc is not None else "") or "")
            member = {
                "id":          md.get("id", ""),
                "kind":        kind,
                "name":        (md.findtext("name") or "").strip(),
                "qualified":   qualified,
                "type":        _itertext(md.find("type")),
                "args":        (md.findtext("argsstring") or "").strip(),
                "param_types": param_types,
                "params_sig":  params_sig,
                "macro_params": macro_params,
                "initializer": initializer,
                "brief":       _itertext(md.find("briefdescription")),
                "enum_values": enum_values,
                "strong":      is_strong,
                "static":      md.get("static") == "yes",
                "inline":      md.get("inline") == "yes",
                "const":       md.get("const") == "yes",
                "template":    _member_template(md),
                "include_file": _include_file,
                "section_header": sd_header,
                "detailed":    _dtl,
                "params":      _params,
                "returns":     _returns,
            }
            if _is_phantom_variable(member):
                continue
            sections.setdefault(section_title, []).append(member)
    return sections


def _build_api_hierarchy(refid: str, xml_dir: pathlib.Path,
                         _seen: set | None = None) -> dict | None:
    """Walk a group XML's <innergroup> children recursively."""
    import xml.etree.ElementTree as _ET
    _seen = _seen if _seen is not None else set()
    if refid in _seen:
        return None
    _seen.add(refid)
    xml_path = xml_dir / f"{refid}.xml"
    if not xml_path.is_file():
        return None
    try:
        root = _ET.parse(xml_path).getroot()
    except _ET.ParseError:
        return None
    cd = root.find("compounddef")
    if cd is None:
        return None
    name = (cd.findtext("compoundname") or "").strip()
    title = (cd.findtext("title") or name).strip()
    detailed_el = cd.find("detaileddescription")
    detailed = _doxygen_desc_to_md(detailed_el) if detailed_el is not None else ""
    # Inner classes (public only). One read per class's XML for its brief.
    # `qualified` is what `{doxygenclass}` needs (e.g. cv::ocl::Context); the
    # innerclass element's text already carries that, but normalize spaces.
    innerclasses = []
    for ic in cd.findall("innerclass"):
        if ic.get("prot") != "public":
            continue
        ic_refid = ic.get("refid", "")
        qualified = " ".join((ic.text or "").split())
        innerclasses.append({
            "refid": ic_refid,
            "name": qualified,
            "qualified": qualified,
            "kind": "struct" if ic_refid.startswith("struct") else "class",
            "brief": _read_class_brief(ic_refid, xml_dir),
        })
    sections = _parse_member_sections(cd)
    children = []
    for ig in cd.findall("innergroup"):
        child = _build_api_hierarchy(ig.get("refid"), xml_dir, _seen)
        if child is not None:
            children.append(child)
    return {"name": name, "title": title, "detailed": detailed,
            "innerclasses": innerclasses, "sections": sections,
            "children": children}


def _type_to_md(type_elem) -> str:
    """Render <type> XML as Markdown; turns <ref> children into links."""
    if type_elem is None:
        return ""
    out = type_elem.text or ""
    for child in type_elem:
        if child.tag == "ref":
            word = (child.text or "").strip()
            refid = child.get("refid", "")
            kindref = child.get("kindref", "")
            if kindref == "compound" and refid in _ANCHOR_TO_DOC:
                fn = _ANCHOR_TO_DOC[refid].split("/")[-1]
                out += f"[{word}]({fn}.md)"
            elif refid:
                out += f"[{word}](#{refid})"
            else:
                out += word
        out += child.tail or ""
    return out.strip()


def _doxygen_desc_to_md(el, h_level: int = 3) -> str:
    """Convert a Doxygen <detaileddescription> element to Markdown."""
    if el is None:
        return ""

    def _hl_text(hl_node) -> str:
        parts = []
        if hl_node.text:
            parts.append(hl_node.text)
        for child in hl_node:
            if child.tag == "sp":
                parts.append(" ")
            else:
                parts.append("".join(child.itertext()))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    def _programlisting(node) -> str:
        lines = []
        for codeline in node.findall("codeline"):
            lines.append("".join(_hl_text(hl) for hl in codeline.findall("highlight")))
        return "```cpp\n" + "\n".join(lines) + "\n```"

    def _ref_link(refid: str, text: str) -> str:
        if not (refid and text):
            return f"`{text}`" if text else ""
        m = re.search(r'_1([a-z]{1,3}[0-9a-f]{20,})$', refid)
        if m:
            url = f"{DOXYGEN_BASE_URL}{refid[:m.start()]}.html#{m.group(1)}"
        else:
            url = f"{DOXYGEN_BASE_URL}{refid}.html"
        return f"[`{text}`]({url})"

    _formula_md = _render_formula

    _BLOCK_TAGS = {"orderedlist", "itemizedlist", "programlisting", "simplesect", "table"}

    def _emit_block(sub, result: list, level: int) -> None:
        t = sub.tag
        if t == "programlisting":
            result.append(_programlisting(sub))
        elif t == "orderedlist":
            for i, item in enumerate(sub.findall("listitem"), 1):
                result.append(f"{i}. {_listitem_text(item)}")
        elif t == "itemizedlist":
            for item in sub.findall("listitem"):
                result.append(f"- {_listitem_text(item)}")
        elif t == "simplesect":
            kind = sub.get("kind", "")
            admon = {"note": "note", "warning": "warning",
                     "attention": "warning", "remark": "note"}.get(kind)
            body = "\n\n".join(_blocks(sub, level))
            if admon:
                result.append(f":::{{{admon}}}\n{body}\n:::")
            elif body:
                result.append(body)
        elif t == "table":
            rows = sub.findall("row")
            if not rows:
                return
            md_rows = []
            for row in rows:
                cells = [" ".join(_blocks(e, level)).replace("|", "\\|").replace("\n", " ").strip()
                         for e in row.findall("entry")]
                md_rows.append("| " + " | ".join(cells) + " |")
            has_header = (rows[0].find("entry") is not None
                          and rows[0].find("entry").get("thead") == "yes")
            if has_header and md_rows:
                cols = len(rows[0].findall("entry"))
                table_lines = [md_rows[0], "| " + " | ".join(["----"] * cols) + " |"] + md_rows[1:]
            else:
                table_lines = md_rows
            result.append("\n".join(table_lines))

    def _listitem_text(item, indent: int = 0) -> str:
        """Render one <listitem>'s content, recursively including nested
        lists. Without this, sub-bullets (e.g. the Modern-Robotics
        link / det(R)=1 properties / Slerp item under the camera-calib
        Note) were silently dropped — `_inline` breaks at the first
        block tag so the nested `<itemizedlist>` inside the parent
        `<para>` was never recursed into. Doxygen puts the nested list
        BOTH inside `<para>` (the common case) AND occasionally as a
        direct child of `<listitem>`, so we check both spots."""
        first = ""
        extras: list[str] = []
        pad = " " * (indent + 2)

        def _emit_sublist(nl) -> None:
            if nl.tag == "itemizedlist":
                for sub in nl.findall("listitem"):
                    extras.append(
                        f"{pad}- {_listitem_text(sub, indent + 2)}")
            elif nl.tag == "orderedlist":
                for i, sub in enumerate(nl.findall("listitem"), 1):
                    extras.append(
                        f"{pad}{i}. {_listitem_text(sub, indent + 2)}")

        for child in item:
            if child.tag == "para":
                # Inline text up to (but not including) the first block tag.
                txt = _inline(child).strip()
                if txt:
                    if not first:
                        first = txt
                    else:
                        extras.append(pad + txt)
                # Nested lists living inside the <para> (Doxygen's
                # typical layout for the OpenCV @note bullets).
                for inner in child:
                    if inner.tag in ("itemizedlist", "orderedlist"):
                        _emit_sublist(inner)
            elif child.tag in ("itemizedlist", "orderedlist"):
                _emit_sublist(child)
        if extras:
            return first + "\n" + "\n".join(extras) if first else "\n".join(extras)
        return first

    def _inline(node) -> str:
        parts = []
        if node.text:
            parts.append(node.text)
        for child in node:
            t = child.tag
            if t in _BLOCK_TAGS:
                break
            inner = "".join(child.itertext())
            if t == "ulink":
                url = child.get("url", "")
                parts.append(f"[{inner}]({url})" if url else inner)
            elif t == "image":
                parts.append(_render_image(child))
            elif t == "ref":
                parts.append(_ref_link(child.get("refid", ""), inner))
            elif t == "computeroutput":
                parts.append(f"`{inner}`" if inner else "")
            elif t == "emphasis":
                parts.append(_wrap_emphasis(inner, "*") if inner else "")
            elif t in ("bold", "strong"):
                parts.append(_wrap_emphasis(inner, "**") if inner else "")
            elif t == "formula":
                parts.append(_formula_md(inner))
            elif t == "sp":
                parts.append(" ")
            elif t == "linebreak":
                parts.append("\n")
            else:
                parts.append(inner)
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    def _blocks(node, level: int) -> list[str]:
        result = []
        children = list(node)
        i = 0
        while i < len(children):
            child = children[i]
            t = child.tag
            if t == "title":
                i += 1
                continue
            elif t == "simplesect":
                # Merge consecutive simplesects of the same kind into one admonition.
                kind = child.get("kind", "")
                admon = {"note": "note", "warning": "warning",
                         "attention": "warning", "remark": "note"}.get(kind)
                bodies = []
                while (i < len(children) and children[i].tag == "simplesect"
                       and children[i].get("kind", "") == kind):
                    bodies.append("\n\n".join(_blocks(children[i], level)))
                    i += 1
                body = "\n\n".join(b for b in bodies if b)
                if admon and body:
                    result.append(f":::{{{admon}}}\n{body}\n:::")
                elif body:
                    result.append(body)
                continue
            elif t == "para":
                pending: list[str] = []
                if child.text:
                    pending.append(child.text)
                for sub in child:
                    if sub.tag in _BLOCK_TAGS:
                        text = "".join(pending).strip()
                        if text:
                            result.append(text)
                        pending = []
                        _emit_block(sub, result, level)
                        if sub.tail and sub.tail.strip():
                            pending.append(sub.tail)
                    else:
                        inner = "".join(sub.itertext())
                        st = sub.tag
                        if st == "ulink":
                            url = sub.get("url", "")
                            pending.append(f"[{inner}]({url})" if url else inner)
                        elif st == "image":
                            pending.append(_render_image(sub))
                        elif st == "ref":
                            pending.append(_ref_link(sub.get("refid", ""), inner))
                        elif st == "computeroutput":
                            pending.append(f"`{inner}`" if inner else "")
                        elif st == "emphasis":
                            pending.append(_wrap_emphasis(inner, "*") if inner else "")
                        elif st in ("bold", "strong"):
                            pending.append(_wrap_emphasis(inner, "**") if inner else "")
                        elif st == "formula":
                            pending.append(_formula_md(inner))
                        elif st == "sp":
                            pending.append(" ")
                        elif st == "linebreak":
                            pending.append("\n")
                        else:
                            pending.append(inner)
                        if sub.tail:
                            pending.append(sub.tail)
                text = "".join(pending).strip()
                if text:
                    result.append(text)
            elif t in ("sect1", "sect2", "sect3"):
                title_text = child.findtext("title") or ""
                if title_text:
                    result.append(f"{'#' * level} {title_text}")
                    result.extend(_blocks(child, level + 1))
                else:
                    result.extend(_blocks(child, level))
            elif t in _BLOCK_TAGS:
                _emit_block(child, result, level)
            i += 1
        return result

    # Merge consecutive same-kind admonitions (e.g. two note paras → one box).
    raw = [b for b in _blocks(el, h_level) if b.strip()]
    merged: list[str] = []
    for block in raw:
        if merged and block.startswith(":::") and merged[-1].startswith(":::"):
            prev_kind = merged[-1].split("\n", 1)[0]
            cur_kind = block.split("\n", 1)[0]
            if prev_kind == cur_kind:
                inner_prev = merged[-1][len(prev_kind)+1:-3].strip()
                inner_cur = block[len(cur_kind)+1:-3].strip()
                merged[-1] = f"{prev_kind}\n{inner_prev}\n\n{inner_cur}\n:::"
                continue
        merged.append(block)
    return "\n\n".join(merged)


def _normalize_include(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    marker = "/include/"
    i = p.rfind(marker)
    return p[i + len(marker):] if i >= 0 else p


def _enum_value_desc(ev) -> str:
    if ev is None:
        return ""
    parts = [_doxygen_desc_to_md(ev.find(t)).strip()
             for t in ("briefdescription", "detaileddescription")]
    return "\n\n".join(p for p in parts if p)


def _md_escape_cell(text: str) -> str:
    """Make `text` safe for a single Markdown table cell."""
    return (text or "").replace("\n", " ").replace("\r", " ") \
                       .replace("|", "\\|").strip()


# memberdef@kind -> per-member breathe directive.
_MEMBER_DIRECTIVE = {
    "enum":     "doxygenenum",
    "function": "doxygenfunction",
    "typedef":  "doxygentypedef",
    "variable": "doxygenvariable",
    "define":   "doxygendefine",
}
# section title -> detail-block header.
_MEMBER_DETAIL_SECTION = {
    "Typedefs":     "Typedef Documentation",
    "Enumerations": "Enumeration Type Documentation",
    "Functions":    "Function Documentation",
    "Variables":    "Variable Documentation",
    "Macros":       "Macro Definition Documentation",
}


def _sphinx_cpp_v4_id(qualified_name: str) -> str:
    """Sphinx C++ v4 anchor id; mangling must match Sphinx's own."""
    parts = qualified_name.split("::")
    mangled = "".join(f"{len(p)}{p}" for p in parts)
    return f"_CPPv4N{mangled}E"


def _enum_synopsis_html(m: dict, strip_scope: str = "") -> list[str]:
    """Render an enum as an HTML block with clickable names.

    Span classes mirror Pygments' cpp-lexer output. `strip_scope` drops the
    redundant prefix from displayed names but never from the mangled id."""
    import html as _html
    qualified = m.get("qualified") or m["name"]
    is_strong = bool(m.get("strong"))
    keyword = "enum class" if is_strong else "enum"
    if is_strong:
        prefix = qualified + "::"
    elif "::" in qualified:
        prefix = qualified.rsplit("::", 1)[0] + "::"
    else:
        prefix = ""
    display_enum = qualified
    if strip_scope and qualified.startswith(strip_scope + "::"):
        display_enum = qualified[len(strip_scope) + 2:]
    display_prefix = prefix
    if strip_scope and prefix.startswith(strip_scope + "::"):
        display_prefix = prefix[len(strip_scope) + 2:]

    enum_id = _sphinx_cpp_v4_id(qualified)
    out = [
        '<div class="highlight-cpp notranslate opencv-enum-synopsis">'
        '<div class="highlight"><pre>'
    ]
    # href only, no id; anchors live in the detail loop.
    enum_link = (
        f'<a class="opencv-enum-link" href="#{enum_id}">'
        f'<span class="n">{_html.escape(display_enum)}</span></a>'
    )
    out.append(
        f'<span class="k">{_html.escape(keyword)}</span> '
        f'{enum_link} <span class="p">{{</span>'
    )
    vals = m.get("enum_values", []) or []
    for i, v in enumerate(vals):
        comma = '<span class="p">,</span>' if i < len(vals) - 1 else ""
        val_id = _sphinx_cpp_v4_id(f"{qualified}::{v['name']}")
        val_link = (
            f'<a class="opencv-enum-link" href="#{val_id}">'
            f'<span class="n">{_html.escape(v["name"])}</span></a>'
        )
        init_html = (" " + _html.escape(v["initializer"])) if v["initializer"] else ""
        prefix_html = ""
        if display_prefix:
            stripped = display_prefix.rstrip(":")
            prefix_html = (
                f'<span class="n">{_html.escape(stripped)}</span>'
                f'<span class="o">::</span>'
            )
        out.append(f"    {prefix_html}{val_link}{init_html}{comma}")
    out.append('<span class="p">}</span></pre></div></div>')
    return out


def _enum_synopsis_lines(m: dict, strip_scope: str = "") -> list[str]:
    """Render an enum as a Doxygen-style `enum {…}` code synopsis."""
    qualified = m.get("qualified") or m["name"]
    is_strong = bool(m.get("strong"))
    keyword = "enum class" if is_strong else "enum"
    if is_strong:
        prefix = qualified + "::"
    elif "::" in qualified:
        prefix = qualified.rsplit("::", 1)[0] + "::"
    else:
        prefix = ""
    enum_label = qualified
    if strip_scope and qualified.startswith(strip_scope + "::"):
        enum_label = qualified[len(strip_scope) + 2:]
    if strip_scope and prefix.startswith(strip_scope + "::"):
        prefix = prefix[len(strip_scope) + 2:]
    out = [f"{keyword} {enum_label} {{"]
    vals = m.get("enum_values", []) or []
    for i, v in enumerate(vals):
        comma = "," if i < len(vals) - 1 else ""
        init = (" " + v["initializer"]) if v["initializer"] else ""
        out.append(f"    {prefix}{v['name']}{init}{comma}")
    out.append("}")
    return out


def _function_signature(member: dict) -> str:
    """`(types)` disambiguator for `{doxygenfunction}`; trailing `const` needed for const methods."""
    types = ", ".join((t or "").strip() for t in member.get("param_types", []))
    sig = f"({types})"
    if member.get("const"):
        sig += " const"
    return sig


def _class_page_name(refid: str) -> str:
    """Per-class page filename: the Doxygen refid verbatim."""
    return refid


def _read_class_data(refid: str, xml_dir: pathlib.Path) -> dict | None:
    """Read a class/struct compound XML for the per-class page."""
    import xml.etree.ElementTree as _ET
    xml_path = xml_dir / f"{refid}.xml"
    if not xml_path.is_file():
        return None
    try:
        root = _ET.parse(xml_path).getroot()
    except _ET.ParseError:
        return None
    cd = root.find("compounddef")
    if cd is None:
        return None

    def _param_type(p) -> str:
        t = _itertext(p.find("type"))
        arr = (p.findtext("array") or "").strip()
        return (t + arr) if arr else t

    sections: dict[str, list[dict]] = {}
    for sd in cd.findall("sectiondef"):
        skind = sd.get("kind", "")
        items: list[dict] = []
        for md in sd.findall("memberdef"):
            mkind = md.get("kind", "")
            qualified = (md.findtext("qualifiedname") or "").strip()
            name = (md.findtext("name") or "").strip()
            enum_values = []
            if mkind == "enum":
                for ev in md.findall("enumvalue"):
                    enum_values.append({
                        "name":        (ev.findtext("name") or "").strip(),
                        "initializer": (ev.findtext("initializer") or "").strip(),
                        "brief":       _enum_value_desc(ev),
                    })
            _dtl, _params, _returns = _member_detail_parts(md)
            items.append({
                "id":          md.get("id", ""),
                "kind":        mkind,
                "name":        name,
                "qualified":   qualified or name,
                "type":        _itertext(md.find("type")),
                "args":        (md.findtext("argsstring") or "").strip(),
                "param_types": [_param_type(p) for p in md.findall("param")],
                "params_sig":  [(_param_type(p),
                                 (p.findtext("declname") or "").strip(),
                                 _itertext(p.find("defval")))
                                for p in md.findall("param")],
                "macro_params": ([(p.findtext("defname") or "").strip()
                                   for p in md.findall("param")]
                                  if mkind == "define" else []),
                "initializer": _itertext(md.find("initializer")),
                "brief":       _itertext(md.find("briefdescription")),
                "static":      md.get("static") == "yes",
                "virt":        md.get("virt", "non-virtual"),
                "const":       md.get("const") == "yes",
                "explicit":    md.get("explicit") == "yes",
                "enum_values": enum_values,
                "strong":      md.get("strong", "no") == "yes",
                "template":    _member_template(md),
                "detailed":    _dtl,
                "params":      _params,
                "returns":     _returns,
            })
        if items:
            sections[skind] = items

    # Boolean only; rendering delegated to breathe's :no-members:.
    detailed_el = cd.find("detaileddescription")
    has_detailed = bool(detailed_el is not None and any(
        _itertext(p).strip() for p in detailed_el.findall("para")
    ))
    include = _normalize_include(cd.findtext("includes") or "")
    return {
        "name":     (cd.findtext("compoundname") or "").strip(),
        "brief":    _itertext(cd.find("briefdescription")),
        "detailed": has_detailed,
        "sections": sections,
        "include":  include,
    }


# Filename -> path for every Doxygen `*graph.svg` (coll/call/caller). Built once
# per html_root: a single tree walk replaces an rglob per class/function, which
# matters now that thousands of function detail blocks each look up a graph.
_GRAPH_SVG_INDEX: dict[str, pathlib.Path] | None = None
_GRAPH_SVG_ROOT: pathlib.Path | None = None


def _graph_svg_index(html_root: pathlib.Path) -> dict[str, pathlib.Path]:
    global _GRAPH_SVG_INDEX, _GRAPH_SVG_ROOT
    if _GRAPH_SVG_INDEX is None or _GRAPH_SVG_ROOT != html_root:
        _GRAPH_SVG_ROOT = html_root
        _GRAPH_SVG_INDEX = {}
        if html_root and html_root.is_dir():
            for p in html_root.rglob("*graph.svg"):
                _GRAPH_SVG_INDEX.setdefault(p.name, p)
    return _GRAPH_SVG_INDEX


def _find_collaboration_svg(refid: str, html_root: pathlib.Path) -> pathlib.Path | None:
    """Locate the legacy Doxygen HTML collaboration SVG for a class."""
    return _graph_svg_index(html_root).get(f"{refid}__coll__graph.svg")


def _find_call_graph_svgs(
        member: dict,
        html_root: pathlib.Path) -> list[tuple[pathlib.Path, str, str]]:
    """Legacy call/caller-graph SVGs for a function member, as (path, intro, alt).

    Bridges the member's XML id to the HTML anchor via `_CALL_GRAPH_ANCHORS`
    (the SVGs are named after the HTML anchor, not the XML memberdef id)."""
    if member.get("kind") != "function":
        return []
    mid = member.get("id", "")
    if "_1" not in mid:
        return []
    compound = mid.rsplit("_1", 1)[0]
    name = member.get("name", "")
    if not (compound and name):
        return []
    anchor = _CALL_GRAPH_ANCHORS.get(
        (compound, name, _norm_args(member.get("args", ""))))
    if not anchor:
        return []
    index = _graph_svg_index(html_root)
    out: list[tuple[pathlib.Path, str, str]] = []
    for suffix, intro, kind in (
        ("cgraph", "Here is the call graph for this function:", "Call"),
        ("icgraph", "Here is the caller graph for this function:", "Caller"),
    ):
        svg = index.get(f"{compound}_{anchor}_{suffix}.svg")
        if svg is not None:
            out.append((svg, intro, f"{kind} graph for {name}"))
    return out


def _svg_make_transparent(text: str) -> str:
    """Light-mode: make only the full-canvas backdrop transparent."""
    return text.replace('fill="white" stroke="transparent"',
                        'fill="none" stroke="transparent"', 1)


def _svg_dark_variant(text: str) -> str:
    """Dark-mode: recolour the light Doxygen SVG into a dark diagram."""
    import re as _re
    text = _svg_make_transparent(text)              # backdrop first; order matters
    for _old in ('stroke="#666666"', 'stroke="black"', 'stroke="#404040"'):
        text = text.replace(_old, 'stroke="#c9d1d9"')
    text = text.replace('fill="#666666"', 'fill="#c9d1d9"')
    def _process_node(m):
        block = m.group(0)
        block = block.replace('fill="white"', 'fill="none"')
        block = block.replace('fill="grey"',  'fill="none"')
        block = block.replace('fill="#999999"', 'fill="#2d333b"')
        block = block.replace('fill="#bfbfbf"', 'fill="#2d333b"')
        block = block.replace('<text ', '<text fill="#ffffff" ')
        return block
    text = _re.sub(r'<g[^>]*class="node"[^>]*>.*?</g>',
                   _process_node, text, flags=_re.DOTALL)
    text = text.replace('fill="grey"',     'fill="none"')
    text = text.replace('fill="#999999"',  'fill="#2d333b"')
    text = text.replace('fill="#bfbfbf"',  'fill="#2d333b"')
    # Lookahead avoids double-prefixing per-node texts.
    text = _re.sub(r'<text (?!fill)', '<text fill="#ffffff" ', text)
    return text


_SPILLED_PROSE_RE = re.compile(r"\b(?:This|The)\s+(?:function|method)\b")


def _hoist_spilled_param_prose(cd) -> bool:
    import xml.etree.ElementTree as _ET
    changed = False
    for md in cd.iter("memberdef"):
        if md.get("kind") != "function":
            continue
        de = md.find("detaileddescription")
        if de is None:
            continue
        # param name -> declared type, to spot callback / function-pointer params.
        ptypes: dict = {}
        for p in md.findall("param"):
            dn = (p.findtext("declname") or "").strip()
            te = p.find("type")
            if dn:
                ptypes[dn] = "".join(te.itertext()) if te is not None else ""
        hoisted: list = []
        for it in de.iter("parameteritem"):
            pd = it.find("parameterdescription")
            para = pd.find("para") if pd is not None else None
            if para is None:
                continue
            ptype = " ".join(ptypes.get("".join(n.itertext()).strip(), "")
                             for n in it.findall(".//parametername"))
            if "Callback" in ptype or "(*" in ptype or "function<" in ptype:
                continue
            text = para.text or ""
            m = _SPILLED_PROSE_RE.search(text)
            if m is None or m.start() == 0:
                continue
            keep = text[:m.start()].rstrip()
            if not keep or re.search(r"\b(?:function|callback)\b", keep, re.I):
                continue
            spill = _ET.Element("para")
            spill.text = text[m.start():]
            for child in list(para):      # all children follow the leading text
                para.remove(child)
                spill.append(child)
            para.text = keep
            hoisted.append(spill)
            changed = True
        if hoisted:
            # Insert ahead of the <para> carrying the <parameterlist>, so the
            # prose reads as body text before the parameter table.
            kids = list(de)
            at = next((i for i, p in enumerate(kids)
                       if p.find("parameterlist") is not None), len(kids))
            for off, sp in enumerate(hoisted):
                de.insert(at + off, sp)
    return changed


def _patch_namespace_xml_for_breathe(xml_dir: pathlib.Path,
                                     out_dir: pathlib.Path) -> None:
    """Mirror `xml_dir` symlinks into `out_dir`, inlining group-only memberdefs.

    breathe's by-name lookup ignores bare `<member refid>`s, so @addtogroup
    members must be copied into the compound's sectiondef for it to resolve."""
    import xml.etree.ElementTree as _ET
    import os as _osmod, shutil as _shutil
    src_index = xml_dir / "index.xml"
    # Real stamp file, not dst_index (a symlink stat() would follow).
    stamp = out_dir / ".mirror_complete"
    if (src_index.is_file() and stamp.is_file()
            and stamp.stat().st_mtime >= src_index.stat().st_mtime):
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mirror xml_dir into out_dir as symlinks.
    for child in list(out_dir.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            _shutil.rmtree(child)
    for src in xml_dir.iterdir():
        dst = out_dir / src.name
        try:
            _osmod.symlink(src, dst)
        except (OSError, NotImplementedError):
            _shutil.copy2(src, dst)

    # 2) Cache for parsed group XMLs.
    _group_cache: dict[str, _ET.ElementTree] = {}

    def _load_group(group_id: str):
        if group_id in _group_cache:
            return _group_cache[group_id]
        gx = xml_dir / f"{group_id}.xml"
        if not gx.is_file():
            _group_cache[group_id] = None
            return None
        try:
            _group_cache[group_id] = _ET.parse(gx)
        except _ET.ParseError:
            _group_cache[group_id] = None
        return _group_cache[group_id]

    # 3) Patch group-memberdef `<member refid>`s in each non-group compound.
    _SKIP_PREFIXES = ("group", "index")
    for compound_file in xml_dir.glob("*.xml"):
        if any(compound_file.name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        try:
            tree = _ET.parse(compound_file)
        except _ET.ParseError:
            continue
        root = tree.getroot()
        cd = root.find("compounddef")
        if cd is None:
            continue
        existing_ids = {md.get("id") for md in cd.iter("memberdef")}
        patched = False
        for sd in cd.findall("sectiondef"):
            for member in list(sd.findall("member")):
                refid = member.get("refid", "")
                if not refid or refid in existing_ids:
                    continue
                # Compound id is the refid before "_1".
                if not refid.startswith("group__"):
                    continue
                sep = refid.find("_1")
                if sep < 0:
                    continue
                group_id = refid[:sep]
                gtree = _load_group(group_id)
                if gtree is None:
                    continue
                for md in gtree.getroot().iter("memberdef"):
                    if md.get("id") == refid:
                        sd.append(md)
                        existing_ids.add(refid)
                        patched = True
                        break
        if patched:
            out_file = out_dir / compound_file.name
            if out_file.is_symlink() or out_file.is_file():
                out_file.unlink()
            tree.write(out_file, encoding="utf-8", xml_declaration=True)

    # 3b) Hoist function-level prose that ran on into a @param description back
    #     into the function body. Runs over every mirrored compound (groups
    #     included) so the fix shows wherever breathe resolves the function.
    for compound_file in list(out_dir.glob("*.xml")):
        if compound_file.name == "index.xml":
            continue
        try:
            tree = _ET.parse(compound_file)
        except _ET.ParseError:
            continue
        cd = tree.getroot().find("compounddef")
        if cd is None:
            continue
        if _hoist_spilled_param_prose(cd):
            if compound_file.is_symlink() or compound_file.is_file():
                compound_file.unlink()
            tree.write(compound_file, encoding="utf-8", xml_declaration=True)

    # 4) Record completion.
    stamp.touch()


# -- Namespace pages ---------------------------------------------------------
# Doxygen 1.12 namespace XML lacks <innerclass>; these readers bridge it.

def _build_ns_group_map(all_refids: list[str],
                        xml_dir: pathlib.Path) -> dict[str, set]:
    """Return ``namespace_name -> set(group compound-name)``."""
    import xml.etree.ElementTree as _ET
    ns_to_groups: dict[str, set] = {}
    for refid in all_refids:
        xml_path = xml_dir / f"{refid}.xml"
        if not xml_path.is_file():
            continue
        try:
            cd = _ET.parse(xml_path).getroot().find("compounddef")
        except _ET.ParseError:
            continue
        if cd is None:
            continue
        cname = (cd.findtext("compoundname") or "").strip()
        for inn in cd.findall("innernamespace"):
            ns_xml = xml_dir / f"{inn.get('refid', '')}.xml"
            if not ns_xml.is_file():
                continue
            try:
                ns_cd = _ET.parse(ns_xml).getroot().find("compounddef")
            except _ET.ParseError:
                continue
            if ns_cd is None:
                continue
            ns_name = (ns_cd.findtext("compoundname") or "").strip()
            if not ns_name:
                continue
            if not (ns_cd.findall("sectiondef") or ns_cd.findall("innerclass")
                    or ns_cd.findall("innernamespace")):
                continue
            ns_to_groups.setdefault(ns_name, set()).add(cname)
    return ns_to_groups


def _namespaces_for_group(group_name: str, xml_dir: pathlib.Path,
                          ns_group_map: dict[str, set]) -> list[dict]:
    """Return ``{name, refid, brief, detailed}`` for ``group_name``'s namespaces."""
    import xml.etree.ElementTree as _ET, glob as _glob
    wanted = {ns for ns, grps in ns_group_map.items() if group_name in grps}
    out: list[dict] = []
    for ns_name in sorted(wanted):
        ns_file = xml_dir / ("namespace" + "_1_1".join(ns_name.split("::")) + ".xml")
        if not ns_file.is_file():
            for f in _glob.glob(str(xml_dir / "namespacecv*.xml")):
                try:
                    cd = _ET.parse(f).getroot().find("compounddef")
                except _ET.ParseError:
                    continue
                if cd is not None and (cd.findtext("compoundname") or "").strip() == ns_name:
                    ns_file = pathlib.Path(f)
                    break
        if not ns_file.is_file():
            continue
        try:
            cd = _ET.parse(ns_file).getroot().find("compounddef")
        except _ET.ParseError:
            continue
        if cd is None:
            continue
        brief = _itertext(cd.find("briefdescription"))
        detailed = _doxygen_desc_to_md(cd.find("detaileddescription"))
        out.append({"name": ns_name, "refid": cd.get("id", ""),
                    "brief": brief, "detailed": detailed})
    return out


def _read_namespace_member_sections(ns_refid: str,
                                    patched_xml_dir: pathlib.Path) -> dict:
    """Member sections for a namespace from the PATCHED XML."""
    import xml.etree.ElementTree as _ET
    if not ns_refid:
        return {}
    xml_path = patched_xml_dir / f"{ns_refid}.xml"
    if not xml_path.is_file():
        return {}
    try:
        cd = _ET.parse(xml_path).getroot().find("compounddef")
    except _ET.ParseError:
        return {}
    return _parse_member_sections(cd) if cd is not None else {}


def _namespace_innerclasses(ns_name: str, xml_dir: pathlib.Path) -> list[tuple]:
    """Return ``(refid, qualified_name, kind, brief)`` for classes directly in ``ns_name``."""
    import xml.etree.ElementTree as _ET
    ns_prefix = ns_name + "::"
    refid_prefix = ns_name.replace("::", "_1_1") + "_1_1"
    out: list[tuple] = []
    for kind in ("struct", "class"):
        for xml_file in sorted(xml_dir.glob(f"{kind}{refid_prefix}*.xml")):
            try:
                cd = _ET.parse(xml_file).getroot().find("compounddef")
            except _ET.ParseError:
                continue
            if cd is None:
                continue
            cname = (cd.findtext("compoundname") or "").strip()
            if "::" in cname[len(ns_prefix):]:   # in a sub-namespace
                continue
            out.append((xml_file.stem, cname, kind,
                        _itertext(cd.find("briefdescription"))))
    return out


__all__ = [
    "_itertext", "_type_to_md", "_doxygen_desc_to_md",
    "_enum_value_desc", "_normalize_include",
    "_MEMBERDEF_SECTIONS", "_read_class_brief",
    "_build_api_hierarchy", "_parse_member_sections", "_md_escape_cell",
    "_MEMBER_DIRECTIVE", "_MEMBER_DETAIL_SECTION", "_sphinx_cpp_v4_id",
    "_enum_synopsis_html", "_enum_synopsis_lines", "_function_signature",
    "_class_page_name", "_read_class_data", "_find_collaboration_svg",
    "_find_call_graph_svgs",
    "_svg_make_transparent", "_svg_dark_variant",
    "_patch_namespace_xml_for_breathe",
    "_build_ns_group_map", "_namespaces_for_group",
    "_read_namespace_member_sections", "_namespace_innerclasses",
]
