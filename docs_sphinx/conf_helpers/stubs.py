"""API-reference stub writers.

Emits one Markdown stub per Doxygen group / class, mirroring the legacy
Doxygen page layout (summary tables + per-member breathe directives).
``_generate_api_stubs`` is the entry point the build orchestrator calls.
Builds on the primitives in ``xml_render`` and the shared state in ``state``.
"""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *
from .xml_render import *
from .examples import (
    _find_examples_for_class, _render_examples_block, _generate_example_pages,
)


# Paths written during the current `_generate_api_stubs` run. Drives both the
# write-if-changed optimization (skip rewriting unchanged stubs so Sphinx's
# incremental build doesn't reprocess them) and the stale-file sweep at the
# end (anything in out_dir NOT written this run is from a removed module).
_stub_written: set[pathlib.Path] = set()


def _stub_write(path: pathlib.Path, content: str) -> None:
    """Write `content` to `path` only when it differs from what's on disk, and
    record the path as live for this run. Replaces the previous
    wipe-and-regenerate (`shutil.rmtree`) approach, which rewrote every stub
    each build and forced Sphinx to reprocess the whole api/ tree."""
    if not (path.is_file() and path.read_text(encoding="utf-8") == content):
        path.write_text(content, encoding="utf-8")
    _stub_written.add(path)


def _collect_all_group_names(node: dict) -> list[str]:
    """Flatten a group hierarchy to the list of every group's `name`."""
    return [node["name"]] + [n for c in node["children"]
                             for n in _collect_all_group_names(c)]


def _namespaces_section(entries: list) -> list[str]:
    """`## Namespaces` block listing each namespace as an `@subpage` (the
    `_subpage_list_to_toctree` translate rule turns these into a real
    toctree). `entries` is a list of `(ns_name, anchor)` tuples."""
    lines = ["## Namespaces", ""]
    for _ns_name, anchor in entries:
        lines.append(f"- @subpage {anchor}")
    lines.append("")
    return lines


def _write_namespace_stub(ns: dict, out_dir: pathlib.Path,
                          xml_dir: pathlib.Path) -> tuple[str, str]:
    """Write ``api/namespace_<slug>.md`` for one namespace. Returns
    ``(anchor, filename)``.

    The page is an index, not a second home for member detail: it lists the
    namespace's classes and a summary table per member kind, each row linking
    to the canonical ``#<refid>`` anchor that the owning *group* page already
    emits. Deliberately no per-member ``(id)=`` targets here — those live on
    the group pages, and duplicating them across docs would trip docutils'
    "Duplicate explicit target name". Enums render inline as a code synopsis
    (matching the group-page layout); the namespace's own prose, when any,
    comes from breathe's ``{doxygennamespace}``."""
    slug = ns["name"].replace("::", "__")
    anchor = f"api_ns_{slug}"
    fname = f"namespace_{slug}.md"
    lines = [f"# {ns['name']} namespace {{#{anchor}}}", ""]
    if ns.get("brief"):
        lines += [ns["brief"], ""]

    ns_prefix = ns["name"] + "::"
    innerclasses = _namespace_innerclasses(ns["name"], xml_dir)
    if innerclasses:
        lines += ["## Classes", "", "{.api-reference-table}",
                  "| Name | Description |", "|---|---|"]
        for ic_refid, ic_name, ic_kind, ic_brief in innerclasses:
            page = _class_page_name(ic_refid)
            short = ic_name[len(ns_prefix):] if ic_name.startswith(ns_prefix) else ic_name
            lines.append(
                f"| [`{ic_kind} {short}`]({page}.md) | {_md_escape_cell(ic_brief)} |")
        lines.append("")

    ns_sections = _read_namespace_member_sections(ns.get("refid", ""),
                                                  _PATCHED_XML_DIR)
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = ns_sections.get(section_title, [])
        if not items:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        if section_title == "Functions":
            lines += ["{.api-reference-table .api-function-table}",
                      "| Return | Name | Description |", "|---|---|---|"]
            for m in items:
                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                label = f"{m['name']}{_md_escape_cell(m['args'])}"
                lines.append(
                    f"| `{ret}` | [`{label}`](#{m['id']}) | {_md_escape_cell(m['brief'])} |")
        elif section_title in ("Typedefs", "Variables"):
            marker = ("{.api-typedef-table}" if section_title == "Typedefs"
                      else "{.api-reference-table}")
            lines += [marker, "| Type | Name | Description |", "|---|---|---|"]
            for m in items:
                t = _md_escape_cell(m["type"]) or "&nbsp;"
                lines.append(
                    f"| `{t}` | [`{m['name']}`](#{m['id']}) | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Enumerations":
            for m in items:
                if m["brief"]:
                    lines.append(_md_escape_cell(m["brief"]))
                    lines.append("")
                lines.append("```cpp")
                lines.extend(_enum_synopsis_lines(m))
                lines.append("```")
                lines.append("")
            continue   # already appended trailing blank
        else:  # Macros
            lines += ["{.api-reference-table}",
                      "| Name | Description |", "|---|---|"]
            for m in items:
                lines.append(
                    f"| [`{m['name']}`](#{m['id']}) | {_md_escape_cell(m['brief'])} |")
        lines.append("")

    # Namespace's own prose — emitted directly from the parsed XML text. We do
    # NOT use a bare `{doxygennamespace}` directive: breathe would re-dump every
    # member (duplicating the summary tables above) and register a second set of
    # member targets. The summary rows already deep-link into the group pages.
    if ns.get("detailed"):
        lines += ["## Detailed Description", "", ns["detailed"], ""]

    _stub_write(out_dir / fname, "\n".join(lines) + "\n")
    return anchor, fname


def _write_api_stub(node: dict, out_dir: pathlib.Path,
                    classes_seen: dict, ns_map: dict | None = None) -> None:
    """Write one .md per group node. Recurses into children.

    Parent groups (have <innergroup> children) → navigation index pages with
    @subpage toctrees. Leaf groups → Doxygen-style summary tables (Classes /
    Typedefs / Enumerations / Functions / Variables / Macros) at top, then
    a per-member detail block per kind (one breathe directive per member, not
    the recursive `{doxygengroup}` which inlines every nested class). Inner
    classes get their own pages — emitted later by `_generate_api_stubs` from
    `classes_seen`, which this fn populates."""
    name = node["name"]
    title = node["title"]
    out = out_dir / f"{name}.md"

    if node["children"]:
        # Navigation index page — list children as @subpage entries; the
        # existing _subpage_list_to_toctree rule converts them to a real
        # toctree at translate time.
        lines = [f"# {title} {{#api_{name}}}", ""]
        if node["detailed"]:
            lines += [node["detailed"], ""]
        if ns_map and ns_map.get(name):
            lines += _namespaces_section(ns_map[name])
        lines += ["## Topics", ""]
        for child in node["children"]:
            lines.append(f"- @subpage api_{child['name']}")
        _stub_write(out, "\n".join(lines) + "\n")
        for child in node["children"]:
            _write_api_stub(child, out_dir, classes_seen, ns_map)
        return

    # ---- Leaf page ----------------------------------------------------------
    lines = [f"# {title} {{#api_{name}}}", ""]
    if ns_map and ns_map.get(name):
        lines += _namespaces_section(ns_map[name])

    # Classes summary table — link to the per-class page that
    # `_generate_api_stubs` emits (one .md per refid, deduped across groups).
    if node["innerclasses"]:
        lines += ["## Classes", "", "{.api-reference-table}",
                  "| Name | Description |", "|---|---|"]
        for c in node["innerclasses"]:
            classes_seen.setdefault(c["refid"], c)
            page = _class_page_name(c["refid"])
            link = f"[`{c['kind']} {c['name']}`]({page}.md)"
            lines.append(f"| {link} | {_md_escape_cell(c['brief'])} |")
        lines.append("")

    # Build a fast lookup of class qualified names known so far — used to
    # detect when a group's "Functions"/"Variables" sectiondef is actually
    # listing a class member (Doxygen groups span class boundaries). Such
    # members get rendered on the class page, not as standalone breathe
    # directives.
    class_qualifieds = {c.get("qualified") for c in classes_seen.values()
                        if c.get("qualified")}

    def _is_class_member(m: dict) -> bool:
        q = m.get("qualified") or ""
        if "::" not in q:
            return False
        parent = q.rsplit("::", 1)[0]
        return parent in class_qualifieds

    def _is_template_spec(m: dict) -> bool:
        # Explicit template specializations carry `<…>` in the name (Doxygen
        # stores `cv::saturate_cast< unsigned >`). breathe's C++ parser
        # rejects this as a function-name argument, so we can't emit a
        # `doxygenfunction` directive for them — the summary table still
        # lists them; only the per-member detail block is skipped.
        return "<" in (m.get("name") or "")


    # Section summary tables in Doxygen's order. For class-member items the
    # in-page anchor breathe would have emitted doesn't exist (we skip the
    # per-member directive below) — point the link at the parent class page
    # instead so the table stays clickable.
    def _member_anchor_link(m: dict, label: str) -> str:
        if _is_class_member(m):
            q = m["qualified"]
            parent_qualified = q.rsplit("::", 1)[0]
            for c in classes_seen.values():
                if c.get("qualified") == parent_qualified:
                    return f"[`{label}`]({_class_page_name(c['refid'])}.md)"
        return f"[`{label}`](#{m['id']})"

    for _, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        if section_title == "Functions":
            lines += ["{.api-reference-table .api-function-table}",
                      "| Return | Name | Description |", "|---|---|---|"]
            for m in items:
                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                label = f"{m['name']}{_md_escape_cell(m['args'])}"
                sig_link = _member_anchor_link(m, label)
                lines.append(
                    f"| `{ret}` | {sig_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title in ("Typedefs", "Variables"):
            # Typedefs get their own table style; variables reuse the generic one.
            marker = ("{.api-typedef-table}" if section_title == "Typedefs"
                      else "{.api-reference-table}")
            lines += [marker, "| Type | Name | Description |", "|---|---|---|"]
            for m in items:
                t = _md_escape_cell(m["type"]) or "&nbsp;"
                name_link = _member_anchor_link(m, m["name"])
                lines.append(f"| `{t}` | {name_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Enumerations":
            # Code-style synopsis (Doxygen layout) instead of name/desc table.
            # Both summary and detail-block representations would duplicate
            # the same content — we only emit the synopsis here, and skip
            # enums in the detail-block loop below.
            for m in items:
                if m["brief"]:
                    lines.append(_md_escape_cell(m["brief"]))
                    lines.append("")
                lines.append("```cpp")
                lines.extend(_enum_synopsis_lines(m))
                lines.append("```")
                lines.append("")
            continue   # already appended trailing blank
        else:  # Macros
            lines += ["{.api-reference-table}",
                      "| Name | Description |", "|---|---|"]
            for m in items:
                name_link = _member_anchor_link(m, m["name"])
                lines.append(f"| {name_link} | {_md_escape_cell(m['brief'])} |")
        lines.append("")

    # Per-member detail blocks. Functions / typedefs / variables are rendered
    # from the Doxygen XML by `_render_member_detail` (NOT breathe): breathe's
    # overload resolver is unreliable for these — template-class members,
    # template-vs-non-template overload sets, and signatures carrying OpenCV
    # macros (`CV_OUT Point *`) all produce "Cannot find" / "Unable to resolve"
    # warnings. Macros keep breathe's `{doxygendefine}` (reliable, deduped by
    # name). Enums are rendered as a code synopsis in the summary section above.
    # Class members and template specializations are skipped here — see `_is_*`.
    seen_define_names: set[str] = set()
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items or kind_key == "enum":
            continue
        blocks: list[list[str]] = []
        for m in items:
            # Class members render on their own class page; skip on the group.
            if kind_key in ("function", "variable") and _is_class_member(m):
                continue
            # Explicit template specializations (`saturate_cast< unsigned >`)
            # carry `<…>` in the name — the summary table still lists them.
            if _is_template_spec(m):
                continue
            if kind_key == "define":
                # Macros aren't namespaced; dedupe arity-overloaded ones.
                if m["name"] in seen_define_names:
                    continue
                seen_define_names.add(m["name"])
                # No `(id)=` anchor here: breathe's {doxygendefine} already
                # registers the Doxygen refid as a target, and adding our own
                # would collide ("Duplicate explicit target name"). The summary
                # table's `#refid` link resolves to breathe's target.
                blocks.append([
                    f"```{{doxygendefine}} {m['name']}",
                    ":project: opencv",
                    "```",
                    "",
                ])
            else:
                blocks.append(
                    _render_member_detail(m, m["qualified"] or m["name"]))
        if not blocks:
            continue
        lines.append(f"## {_MEMBER_DETAIL_SECTION[section_title]}")
        lines.append("")
        for b in blocks:
            lines += b

    # Hidden toctree for per-class pages — needed so Sphinx knows these
    # files exist and so the left sidebar lists them under this group.
    if node["innerclasses"]:
        lines += ["```{toctree}", ":hidden:", ":maxdepth: 1", ""]
        for c in node["innerclasses"]:
            lines.append(_class_page_name(c["refid"]))
        lines += ["```", ""]

    _stub_write(out, "\n".join(lines) + "\n")


# Class-XML sectiondef kind → (summary heading, detail heading).
# Order in this mapping is the order Doxygen uses on a class page.
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


def _render_member_detail(m: dict, full_name: str) -> list[str]:
    """Render one member's detail block as Markdown straight from the Doxygen
    XML data — no breathe ``{doxygenfunction}`` / ``{doxygenvariable}`` /
    ``{doxygentypedef}`` directive.

    breathe's per-overload resolver is unreliable for template-class members
    (``cv::Mat_``/``Matx``/``Vec`` operators, template-vs-non-template overload
    sets like ``cv::Mat::ptr``) and for free functions whose Doxygen signature
    carries OpenCV macros (``CV_OUT Point *``) — it emits "Cannot find" /
    "Unable to resolve" warnings. We already parse the full member data, so we
    emit the declaration + brief/detailed + Parameters + Returns ourselves.
    ``full_name`` is the fully-qualified member name used in the declaration
    (e.g. ``cv::Mat_::operator()`` or ``cv::checkRange``); the ``(id)=`` anchor
    keeps the summary-table ``#refid`` links working."""
    short = m["name"]
    kind = m["kind"]
    head = short + (m.get("args", "") if kind == "function" else "")
    out = [f"({m['id']})=", f"### {head}".rstrip(), ""]

    # Declaration (template line, if any, then the C++ signature).
    tmpl = m.get("template") or ""
    prefix = "static " if m.get("static") else ""
    typ = (m.get("type") or "").strip()
    if kind == "typedef":
        decl = f"typedef {typ} {full_name}".strip()
    elif kind == "function":
        decl = (f"{prefix}{typ + ' ' if typ else ''}"
                f"{full_name}{m.get('args', '')}").strip()
    else:  # variable / attribute
        decl = f"{prefix}{typ + ' ' if typ else ''}{full_name}".strip()
    out += ["```cpp"] + ([tmpl] if tmpl else []) + [decl, "```", ""]

    if m.get("brief"):
        out += [m["brief"], ""]
    if m.get("detailed"):
        out += [m["detailed"], ""]
    if m.get("params"):
        out += ["**Parameters**", ""]
        for nm, desc in m["params"]:
            out.append(f"- `{nm}` — {desc}" if desc else f"- `{nm}`")
        out.append("")
    if m.get("returns"):
        out += ["**Returns**", "", m["returns"], ""]
    return out


def _write_class_stub(cls: dict, out_dir: pathlib.Path,
                      xml_dir: pathlib.Path) -> None:
    """One .md per inner class. Mirrors Doxygen's class-page layout:
      * Brief + detailed description for the class itself
      * Summary tables, one per sectiondef kind (Public Member Functions,
        Static Public Member Functions, Protected Attributes, etc.)
      * Detail blocks per member, grouped Doxygen-style into Constructor &
        Destructor Documentation / Member Function Documentation /
        Member Data Documentation / etc.

    Detail blocks are per-member breathe directives (`{doxygenfunction}` /
    `{doxygenvariable}` / `{doxygentypedef}`), not the recursive
    `{doxygenclass} :members:` — the latter is breathe's discrete
    one-signature-per-method layout that the user wanted replaced.

    Falls back to a bare `{doxygenclass}` / `{doxygenstruct}` if the class
    XML can't be read (e.g. XML wasn't regenerated)."""
    page = _class_page_name(cls["refid"])
    out = out_dir / f"{page}.md"
    qualified = cls["qualified"] or cls["name"]
    kind_label = cls["kind"].title()  # "Class" / "Struct"
    title = f"{kind_label} {qualified}"
    # Note: no `{#refid}` anchor in the heading — duplicates the
    # docname-derived target. `_generate_api_stubs` seeds the
    # refid→docname mapping into `_ANCHOR_TO_DOC` for `@ref` resolution.
    lines = [f"# {title}", ""]

    # Class-page header (PR #7): one-line brief + `More...` jump + the
    # `#include <…>` line, right under the title — mirrors docs.opencv.org.
    # Read class data early just for the header; the main read happens below
    # (cheap, and keeps the header above the collaboration diagram).
    _header_data = _read_class_data(cls["refid"], xml_dir)
    if _header_data is not None:
        import html as _html_pkg
        _brief = (_header_data.get("brief") or "").strip()
        if _brief:
            # `More...` only when there's a detailed description to jump to.
            _more = (
                ' <a class="opencv-class-more" href="#detailed-description">More...</a>'
                if _header_data.get("detailed") else ""
            )
            lines.append(
                f'<p class="opencv-class-brief">'
                f'{_html_pkg.escape(_brief)}{_more}</p>'
            )
            lines.append("")
        _inc = (_header_data.get("include") or "").strip()
        if _inc:
            lines.append(
                f'<div class="opencv-class-include">'
                f'<code>#include &lt;{_html_pkg.escape(_inc)}&gt;</code></div>'
            )
            lines.append("")

    # Collaboration diagram — surface the SVG the legacy Doxygen HTML build
    # already rendered (the XML pipeline disables graphs; see
    # `_find_collaboration_svg`). Copy it next to the stub so Sphinx's image
    # collector publishes it to `_images/`, then reference it relative to the
    # api/ doc. The `images/`/`js_assets/` rewrite in `_translate` doesn't
    # touch this path (no such dir segment), and `_img_xtree` leaves
    # non-contrib image refs unchanged. Absent SVG → section silently omitted.
    _svg = _find_collaboration_svg(cls["refid"], xml_dir.parent / "html")
    _light_name = _dark_name = None
    if _svg is not None:
        import hashlib as _hashlib
        try:
            _raw = _svg.read_text(encoding="utf-8")
            # Two theme variants: light = native Doxygen with a transparent
            # backdrop (white page shows through); dark = recoloured to
            # light-on-dark so it matches docs.opencv.org and blends with the
            # dark page. custom.css shows exactly one per active theme.
            #
            # Filenames are content-hashed: Doxygen names every diagram
            # `<refid>__coll__graph.svg`; if a browser cached an older copy
            # under that fixed name it would keep serving the stale image
            # (query-string busts don't always work — some caches key on path
            # only). A hashed filename is a brand-new URL whenever the SVG
            # content changes, so it can never be served stale.
            _light_txt = _svg_make_transparent(_raw)
            _dark_txt = _svg_dark_variant(_raw)
            _lh = _hashlib.md5(_light_txt.encode("utf-8")).hexdigest()[:10]
            _dh = _hashlib.md5(_dark_txt.encode("utf-8")).hexdigest()[:10]
            _light_name = f"{_svg.stem}.{_lh}.svg"
            _dark_name = f"{_svg.stem}.{_dh}.dark.svg"
            (out_dir / _light_name).write_text(_light_txt, encoding="utf-8")
            (out_dir / _dark_name).write_text(_dark_txt, encoding="utf-8")
        except OSError:
            _light_name = _dark_name = None
    if _light_name is not None:
        # `only-light` / `only-dark` are pydata-sphinx-theme's native
        # theme-aware image classes: the theme shows exactly one per active
        # colour mode (via `display:none !important`), and — critically —
        # exempts `.only-dark` images from its
        # `html[data-theme=dark] .bd-content img { background:#fff }` rule, so
        # our dark (transparent-backdrop) variant blends with the dark page
        # instead of getting a white card behind it.
        lines += [
            f"Collaboration diagram for {qualified}:",
            "",
            f"![Collaboration diagram for {qualified}]({_light_name})"
            "{.opencv-coll-graph .only-light}",
            "",
            f"![Collaboration diagram for {qualified}]({_dark_name})"
            "{.opencv-coll-graph .only-dark}",
            "",
        ]

    data = _read_class_data(cls["refid"], xml_dir)
    if data is None:
        # Fallback for missing XML.
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
        # Type-bearing sections (functions, variables, typedefs) get a
        # Return/Type column. Enum-bearing public-type sections get
        # rendered as code-block synopses instead (matches the group-page
        # treatment).
        non_enum_items = [m for m in items if m["kind"] != "enum"]
        enum_items = [m for m in items if m["kind"] == "enum"]
        if non_enum_items:
            lines += ["{.api-reference-table}",
                      "| Return | Name | Description |", "|---|---|---|"]
            for m in non_enum_items:
                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                if m["static"]:
                    ret = "static " + ret
                sig = f"{m['name']}{_md_escape_cell(m['args'])}"
                sig_link = f"[`{sig}`](#{m['id']})"
                lines.append(
                    f"| `{ret}` | {sig_link} | {_md_escape_cell(m['brief'])} |")
            lines.append("")
        for m in enum_items:
            if m["brief"]:
                lines.append(_md_escape_cell(m["brief"]))
                lines.append("")
            # Raw HTML synopsis (NOT a markdown code fence): emits a code-styled
            # `<div class="highlight-cpp">…<pre>` block where each enum and
            # enumerator name is wrapped in `<a class="opencv-enum-link"
            # id="_CPPv4…" href="#_CPPv4…">…</a>`. Same id on the anchor and the
            # href, so clicking the name updates the URL hash to the
            # Sphinx-C++-domain-style id and the page snaps to the synopsis
            # line. `strip_scope=qualified` drops the redundant `cv::_ClassName::`
            # prefix from the displayed names; the ids keep the full path.
            lines.extend(_enum_synopsis_html(m, strip_scope=qualified))
            lines.append("")

    # 2) "Detailed Description" — delegated to Breathe's `{doxygenclass}` (PR #7)
    #    so every Doxygen tag (programlisting, ref, tables, …) renders natively.
    #    `:no-members:` leaves just the class brief + detailed body; the
    #    duplicate signature header + leading brief paragraph Breathe emits are
    #    stripped at build-finished time by `_strip_breathe_class_clutter`. A
    #    per-class "Examples" cross-reference is appended under the same heading.
    _directive = "doxygenstruct" if cls["kind"] == "struct" else "doxygenclass"
    examples = _find_examples_for_class(qualified.rsplit("::", 1)[-1])
    if data["detailed"]:
        lines += [
            "## Detailed Description",
            "",
            f"```{{{_directive}}} {qualified}",
            ":project: opencv",
            ":no-members:",
            "```",
            "",
        ]
        lines += _render_examples_block(examples)
    elif examples:
        # No detailed description, but the class is referenced from sample
        # code — surface the cross-reference under its own heading.
        lines += ["## Examples", ""]
        lines += _render_examples_block(examples)

    # 3) Per-member detail blocks. Functions split into ctor/dtor vs
    #    others (mirrors Doxygen's "Constructor & Destructor Documentation"
    #    + "Member Function Documentation"). Variables → "Member Data
    #    Documentation". Typedefs → "Member Typedef Documentation".
    #    Enums → "Member Enumeration Documentation" (synopsis code block).
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
        # Member Enumeration Documentation: the *destination* the Public Types
        # synopsis links navigate to. We DON'T repeat the code-block synopsis
        # here (visually identical to the source the user came from — felt
        # like clicking went nowhere). Instead each enum becomes a heading +
        # a definition list where every enumerator is its own row with its
        # own `id="_CPPv4…"` anchor and (when Doxygen provided one) a brief
        # description — matching the docs.opencv.org reference layout.
        import html as _html
        lines += ["## Member Enumeration Documentation", ""]
        for m in enum_items_all:
            # Keep the legacy MyST anchor (`(<doxygen-refid>)=`) so any older
            # `@ref` cross-references that point at it still resolve.
            lines.append(f"({m['id']})=")
            enum_qualified = m.get("qualified") or m["name"]
            enum_id = _sphinx_cpp_v4_id(enum_qualified)
            enum_short = enum_qualified.rsplit("::", 1)[-1]
            # Heading carries the enum-level `_CPPv4N…E` anchor — clicking the
            # enum name in Public Types lands here.
            lines.append(
                f'<h3 class="opencv-enum-heading" id="{enum_id}">'
                f'enum <span class="opencv-enum-name">{_html.escape(enum_short)}</span></h3>'
            )
            if m["brief"]:
                lines.append(f"<p>{_html.escape(_md_escape_cell(m['brief']))}</p>")
            # Per-enumerator rows. Each `<dt>` carries the enumerator's own
            # `_CPPv4N…E` id so clicking a specific value name in Public Types
            # lands on its own row, not on the top of the enum block.
            lines.append('<dl class="opencv-enum-detail">')
            for _v in (m.get("enum_values") or []):
                val_id = _sphinx_cpp_v4_id(f"{enum_qualified}::{_v['name']}")
                init = _html.escape(_v["initializer"]) if _v["initializer"] else ""
                init_html = f' <span class="opencv-enum-init">{init}</span>' if init else ""
                # cv2.* Python name (PR #30), if the signatures artifact exists;
                # otherwise _python_enum_name returns None and nothing is added.
                _py = _python_enum_name(enum_qualified, _v["name"],
                                        bool(m.get("strong")))
                py_html = (f' <span class="opencv-enum-pyname">Python: '
                           f'<code>{_html.escape(_py)}</code></span>') if _py else ""
                lines.append(
                    f'  <dt id="{val_id}">'
                    f'<span class="opencv-enum-name">{_html.escape(_v["name"])}</span>'
                    f'{init_html}{py_html}</dt>'
                )
                brief = (_v.get("brief") or "").strip()
                if brief:
                    lines.append(f'  <dd>{_html.escape(brief)}</dd>')
            lines.append('</dl>')
            lines.append("")

    # Dedupe functions: a method can appear in multiple sectiondefs (e.g.
    # the same memberdef appearing in `public-func` and again via a
    # `<member refid>` we inlined). The refid is unique per memberdef.
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
            lines += _render_member_detail(m, f"{qualified}::{m['name']}")

    if var_items:
        lines += ["## Member Data Documentation", ""]
        for m in _dedupe(var_items):
            lines += _render_member_detail(m, f"{qualified}::{m['name']}")

    _stub_write(out, "\n".join(lines))


def _generate_api_stubs(modules, xml_dir, out_dir):
    """Generate the full api/ stub tree. Write-if-changed: only stubs whose
    content actually changed are rewritten, so Sphinx's incremental build
    reprocesses the minimum. Stale files (from removed modules) are swept at
    the end.

    Passes:
      1. Walk each module's group hierarchy → emit one group .md per node
         (parent index pages + leaf pages with summary tables + per-member
         detail blocks), plus one `namespace_<slug>.md` per namespace the
         module's groups expose. `classes_seen` is populated as a side-effect.
      2. Emit one .md per unique inner class — the per-class subpages the
         group pages link to."""
    if not modules:
        return
    if not xml_dir.is_dir():
        return  # No XML yet (sphinx-xml not run); degrade silently.

    # Freshness guard: if the stub tree is at least as new as the source XML
    # index AND already carries namespace stubs (i.e. was produced by this
    # namespace-aware generator), there's nothing to regenerate — just reseed
    # the refid → docname map so `@ref` cross-refs keep resolving, and return.
    # A tree predating namespace stubs falls through to a full rebuild.
    src_index = xml_dir / "index.xml"
    root_md = out_dir / "api_root.markdown"
    if (src_index.is_file() and root_md.is_file()
            and root_md.stat().st_mtime >= src_index.stat().st_mtime
            and any(p.name.startswith("namespace_") and p.suffix == ".md"
                    for p in out_dir.iterdir())):
        for stub in out_dir.iterdir():
            n = stub.name
            if n.endswith(".md") and (n.startswith("class") or n.startswith("struct")):
                _ANCHOR_TO_DOC[n[:-3]] = f"api/{n[:-3]}"
        return

    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global _stub_written
    _stub_written = set()
    root_lines = [
        "API Reference {#api_root}",
        "=============",
        "",
        "Sphinx-rendered API reference for OpenCV main modules. Each entry",
        "below is a module's umbrella `@defgroup`; sub-pages mirror the",
        "Doxygen subgroup hierarchy.",
        "",
    ]
    classes_seen: dict[str, dict] = {}
    written_ns: set[str] = set()   # namespace stubs are shared across groups
    for m in modules:
        tree = _build_api_hierarchy(
            "group__" + m.replace("_", "__"), xml_dir)
        if tree is None:
            continue
        root_lines.append(f"- @subpage api_{tree['name']}")
        # Namespaces exposed by any group in this module's hierarchy. Build the
        # ns→group map once, then attach each group's namespaces to `ns_map`
        # (consumed by `_write_api_stub` to emit the per-group "## Namespaces"
        # section) and write each namespace page once.
        all_group_names = _collect_all_group_names(tree)
        all_refids = ["group__" + n.replace("_", "__") for n in all_group_names]
        ns_group_map = _build_ns_group_map(all_refids, xml_dir)
        ns_map: dict[str, list] = {}
        for group_name in all_group_names:
            for ns in _namespaces_for_group(group_name, xml_dir, ns_group_map):
                anchor = f"api_ns_{ns['name'].replace('::', '__')}"
                if ns["name"] not in written_ns:   # shared across groups
                    _write_namespace_stub(ns, out_dir, xml_dir)
                    written_ns.add(ns["name"])
                ns_map.setdefault(group_name, []).append((ns["name"], anchor))
        _write_api_stub(tree, out_dir, classes_seen, ns_map)
    # Per-class pages (one per unique refid across all groups). We also
    # seed `_ANCHOR_TO_DOC` directly with refid -> docname so `@ref`
    # cross-references in tutorial markdown (and in any group page) keep
    # working — the per-class page no longer carries a `{#refid}` heading
    # anchor (would duplicate the docname-derived target).
    for cls in classes_seen.values():
        _write_class_stub(cls, out_dir, xml_dir)
        _ANCHOR_TO_DOC[cls["refid"]] = f"api/{_class_page_name(cls['refid'])}"
    _stub_write(out_dir / "api_root.markdown", "\n".join(root_lines) + "\n")
    # Sweep stale files left by removed modules / renamed pages.
    for _p in list(out_dir.iterdir()):
        if _p.is_file() and _p not in _stub_written:
            _p.unlink()
    # Flush per-sample example pages (PR #7) into a sibling `examples/` dir.
    # The set was populated by `_find_examples_for_class` while each class stub
    # was written above — generating now avoids orphan pages no class links to.
    _generate_example_pages(out_dir.parent / "examples")
