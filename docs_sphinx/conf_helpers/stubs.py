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


def _write_api_stub(node: dict, out_dir: pathlib.Path,
                    classes_seen: dict) -> None:
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
        lines += ["## Topics", ""]
        for child in node["children"]:
            lines.append(f"- @subpage api_{child['name']}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for child in node["children"]:
            _write_api_stub(child, out_dir, classes_seen)
        return

    # ---- Leaf page ----------------------------------------------------------
    lines = [f"# {title} {{#api_{name}}}", ""]

    # Classes summary table — link to the per-class page that
    # `_generate_api_stubs` emits (one .md per refid, deduped across groups).
    if node["innerclasses"]:
        lines += ["## Classes", "",
                  "{.api-reference-table}",
                  "| Name | Description |", "|---|---|"]
        for c in node["innerclasses"]:
            classes_seen.setdefault(c["refid"], c)
            page = _class_page_name(c["refid"])
            link = f"{c['kind']} [`{c['name']}`]({page}.md)"
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

    def _needs_manual_function_detail(m: dict) -> bool:
        # Breathe resolves these two `cv::operator!=` overloads by scanning
        # same-name candidates first. In this tree that candidate set includes
        # CUDA-style vector operators whose `__device__ __forceinline__`
        # signatures Sphinx's C++ parser rejects, producing noisy warnings.
        # The overloads have no detailed body, so a local signature block with
        # the same anchor is equivalent and avoids the bad candidate parse.
        return (
            m.get("kind") == "function"
            and m.get("qualified") == "cv::operator!="
            and m.get("args") in (
                "(const MatShape &shape1, const MatShape &shape2)",
                "(const Matx< _Tp, m, n > &a, const Matx< _Tp, m, n > &b)",
            )
        )

    def _manual_function_detail(m: dict) -> list[str]:
        signature = " ".join(
            part for part in (
                _md_escape_cell(m["type"]),
                m.get("qualified") or m["name"],
                _md_escape_cell(m["args"]),
            )
            if part
        )
        return [
            f"({m['id']})=",
            "```cpp",
            signature,
            "```",
            "",
        ]

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
                prefixes = []
                if m.get("static") in ("yes", True):
                    prefixes.append("static")
                if m.get("inline") in ("yes", True):
                    prefixes.append("inline")
                if m.get("virtual") in ("yes", "virtual", True):
                    prefixes.append("virtual")
                prefix = (" ".join(prefixes) + " ") if prefixes else ""

                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                full_return_type = f"{prefix}{ret}"
                template = m.get("template", "")
                template_html = (
                    f"<div class='api-template'>{_md_escape_cell(template)}</div>"
                    if template else ""
                )
                label = f"cv::{m['name']}{_md_escape_cell(m['args'])}"
                sig_link = _member_anchor_link(m, label)
                lines.append(
                    f"| {template_html}{full_return_type} | {sig_link} | "
                    f"{_md_escape_cell(m['brief'])} |")
        elif section_title == "Typedefs":
            lines += ["{.api-typedef-table}",
                      "| Type | Name | Description |", "|---|---|---|"]
            for m in items:
                t = _md_escape_cell(m["type"]) or "&nbsp;"
                name_link = _member_anchor_link(m, f"cv::{m['name']}")
                lines.append(
                    f"| typedef {t} | {name_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Variables":
            lines += ["{.api-reference-table}",
                      "| Type | Name | Description |", "|---|---|---|"]
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

    # Per-member detail blocks (Enumeration Type Documentation,
    # Function Documentation, …). One breathe directive per item — except
    # for enums, which are rendered as a single code-style synopsis (one
    # `enum {…}` block listing all values) to match Doxygen's group-page
    # layout. breathe's `{doxygenenum}` instead emits a discrete signature
    # box per enumerator, which looks fragmented and doesn't match the
    # reference rendering. Class members and template specializations are
    # skipped — see `_is_*`. Macros are deduped by name: Doxygen can emit
    # multiple <memberdef>s for an arity-overloaded macro, but breathe
    # renders the same C declaration each time and docutils complains
    # about duplicate IDs.
    seen_define_names: set[str] = set()
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        directive = _MEMBER_DIRECTIVE.get(kind_key)
        if not directive:
            continue
        # `rendered` collects entries with one of two shapes:
        #   ("breathe", spec, directive_name)  — emitted as a breathe block
        #   ("synopsis", brief, code_lines)    — emitted as a fenced block
        rendered = []
        for m in items:
            # Class members render on the class page; skip on the group page.
            if kind_key in ("function", "variable") and _is_class_member(m):
                continue
            # Explicit template specializations: breathe can't parse the
            # `<T>` in the function name, so we leave them out of the
            # detail section (table above still lists them).
            if _is_template_spec(m):
                continue
            if m["kind"] == "enum":
                # Enums are rendered as synopses in the "Enumerations"
                # summary section above; no separate detail block needed.
                continue
            qualified = m["qualified"] or m["name"]
            if m["kind"] == "function":
                # Always pass the param-types disambiguator. breathe sees
                # multiple matches for common names (e.g. cv::cos lives in
                # both core_quaternion and core_utils_softfloat) — without
                # a signature it can't pick. The XML patcher above
                # guarantees the matching <memberdef> is reachable for
                # breathe's lookup.
                spec = qualified + _function_signature(m)
            elif m["kind"] == "define":
                # Preprocessor macros aren't namespaced. Dedupe by name —
                # arity-overloaded macros (e.g. CV_LOG_VERBOSE) appear as
                # multiple memberdefs but render to the same C declaration.
                if m["name"] in seen_define_names:
                    continue
                seen_define_names.add(m["name"])
                spec = m["name"]
            else:
                spec = qualified
            rendered.append(("breathe", spec, directive, m))
        if not rendered:
            continue
        lines.append(f"## {_MEMBER_DETAIL_SECTION[section_title]}")
        lines.append("")
        for entry in rendered:
            if entry[0] == "synopsis":
                _, brief, code_lines = entry
                if brief:
                    lines.append(brief)
                    lines.append("")
                lines.append("```cpp")
                lines.extend(code_lines)
                lines.append("```")
                lines.append("")
            else:
                _, spec, dname = entry[:3]
                source = entry[3] if len(entry) > 3 else None
                if source is not None and _needs_manual_function_detail(source):
                    lines += _manual_function_detail(source)
                else:
                    lines += [
                        f"```{{{dname}}} {spec}",
                        ":project: opencv",
                        "```",
                        "",
                    ]

    # Hidden toctree for per-class pages — needed so Sphinx knows these
    # files exist and so the left sidebar lists them under this group.
    if node["innerclasses"]:
        lines += ["```{toctree}", ":hidden:", ":maxdepth: 1", ""]
        for c in node["innerclasses"]:
            lines.append(_class_page_name(c["refid"]))
        lines += ["```", ""]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        out.write_text("\n".join(lines), encoding="utf-8")
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

    # 2) "Detailed Description" section for the class itself.
    if data["detailed"]:
        lines += ["## Detailed Description", "", data["detailed"], ""]

    # 3) Per-member detail blocks. Functions split into ctor/dtor vs
    #    others (mirrors Doxygen's "Constructor & Destructor Documentation"
    #    + "Member Function Documentation"). Variables → "Member Data
    #    Documentation". Typedefs → "Member Typedef Documentation".
    #    Enums → "Member Enumeration Documentation" (synopsis code block).
    class_simple = qualified.rsplit("::", 1)[-1]

    # Doxygen leaves <qualifiedname> empty for the members of some classes
    # (cv::_InputArray is one), so `m["qualified"]` falls back to the bare
    # member name. A bare name makes breathe search the *whole* project: it
    # resolves only when the name+signature is unique across all documented
    # symbols (e.g. `channels(int) const`), but common methods shared with
    # other classes stay ambiguous — `copyTo`, `empty`, `getFlags`, `size`
    # all collide with Mat/UMat/etc. and render as "Unable to resolve".
    # Scope every member to this class so the lookup is unambiguous.
    def _scoped(m: dict) -> str:
        nm = m["name"]
        # Conversion operators ("operator std::vector< _Tp >", "operator Vec<…>")
        # have a mangled <qualifiedname> in Doxygen XML — the "operator " (and any
        # "std::") prefix is dropped (e.g. "cv::Mat_::vector< _Tp >"), which breaks
        # the breathe lookup. For ANY operator, rebuild the scope from the class's
        # qualified name + the (correct) member name rather than trusting qualified.
        if nm.startswith("operator"):
            return f"{qualified}::{nm}"
        q = m.get("qualified") or nm
        return q if "::" in q else f"{qualified}::{nm}"

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

    def _emit_member_directive(m: dict, directive: str, spec: str) -> list[str]:
        # MyST anchor label so the summary-table `#refid` link resolves.
        return [
            f"({m['id']})=",
            f"```{{{directive}}} {spec}",
            ":project: opencv",
            "```",
            "",
        ]

    if typedef_items:
        lines += ["## Member Typedef Documentation", ""]
        for m in typedef_items:
            lines += _emit_member_directive(m, "doxygentypedef", _scoped(m))

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
                lines.append(
                    f'  <dt id="{val_id}">'
                    f'<span class="opencv-enum-name">{_html.escape(_v["name"])}</span>'
                    f'{init_html}</dt>'
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
            spec = _scoped(m) + _function_signature(m)
            lines += _emit_member_directive(m, "doxygenfunction", spec)

    if func_items:
        lines += ["## Member Function Documentation", ""]
        for m in _dedupe(func_items):
            spec = _scoped(m) + _function_signature(m)
            lines += _emit_member_directive(m, "doxygenfunction", spec)

    if var_items:
        lines += ["## Member Data Documentation", ""]
        for m in _dedupe(var_items):
            lines += _emit_member_directive(m, "doxygenvariable", _scoped(m))

    out.write_text("\n".join(lines), encoding="utf-8")


def _generate_api_stubs(modules, xml_dir, out_dir):
    """Generate the full api/ stub tree. Idempotent — wipes and regenerates
    on every sphinx-build so stale stubs from removed modules disappear.

    Two passes:
      1. Walk each module's group hierarchy → emit one group .md per node
         (parent index pages + leaf pages with summary tables + per-member
         detail blocks). `classes_seen` is populated as a side-effect.
      2. Emit one .md per unique inner class — these are the per-class
         subpages the group pages link to."""
    if not modules:
        return
    if not xml_dir.is_dir():
        return  # No XML yet (sphinx-xml not run); degrade silently.
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    for m in modules:
        tree = _build_api_hierarchy(
            "group__" + m.replace("_", "__"), xml_dir)
        if tree is None:
            continue
        root_lines.append(f"- @subpage api_{tree['name']}")
        _write_api_stub(tree, out_dir, classes_seen)
    # Per-class pages (one per unique refid across all groups). We also
    # seed `_ANCHOR_TO_DOC` directly with refid -> docname so `@ref`
    # cross-references in tutorial markdown (and in any group page) keep
    # working — the per-class page no longer carries a `{#refid}` heading
    # anchor (would duplicate the docname-derived target).
    for cls in classes_seen.values():
        _write_class_stub(cls, out_dir, xml_dir)
        _ANCHOR_TO_DOC[cls["refid"]] = f"api/{_class_page_name(cls['refid'])}"
    (out_dir / "api_root.markdown").write_text(
        "\n".join(root_lines) + "\n", encoding="utf-8")
