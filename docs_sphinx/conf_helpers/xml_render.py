"""Doxygen XML -> Markdown primitives for the API-reference stubs.

Low-level readers and renderers shared by the stub writers in ``stubs``:
flatten XML text, render the group hierarchy, build enum synopses (plain and
clickable-HTML), read per-class data, recolour collaboration SVGs, and patch
the namespace XML so breathe can resolve @addtogroup members. Shared state
(paths, anchor maps) comes from ``state``.
"""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *


def _itertext(el) -> str:
    """Flatten an XML element's inner text. None-safe."""
    return "".join(el.itertext()).strip() if el is not None else ""


# memberdef@kind → display section title. Mirrors Doxygen's group-page order.
_MEMBERDEF_SECTIONS = (
    ("typedef",  "Typedefs"),
    ("enum",     "Enumerations"),
    ("function", "Functions"),
    ("variable", "Variables"),
    ("define",   "Macros"),
)


def _read_class_brief(refid: str, xml_dir: pathlib.Path,
                      _cache: dict = {}) -> str:
    """Read brief description from a class/struct's compound XML. Cached."""
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


def _build_api_hierarchy(refid: str, xml_dir: pathlib.Path,
                         _seen: set | None = None) -> dict | None:
    """Walk a group XML's <innergroup> children recursively.
    Returns {name, title, detailed, innerclasses, sections, children} or None.
    `_seen` guards against the rare case of cycles in the group graph."""
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
    # Detailed description (used on parent index pages; breathe handles it
    # on leaf pages, so we extract it for context-display only).
    detailed_el = cd.find("detaileddescription")
    detailed = ""
    if detailed_el is not None:
        paras = [_itertext(p) for p in detailed_el.findall("para")]
        detailed = "\n\n".join(p for p in paras if p)
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
    # Section members (typedefs, enums, functions, variables, macros).
    # `qualified` and `param_types` exist so we can emit per-member breathe
    # directives (doxygenenum / doxygenfunction / …) instead of one big
    # doxygengroup; the latter inlines every <innerclass> on the group page,
    # which is the opposite of Doxygen's group-page layout.
    sections: dict[str, list[dict]] = {}
    for sd in cd.findall("sectiondef"):
        for md in sd.findall("memberdef"):
            kind = md.get("kind", "")
            section_title = dict(_MEMBERDEF_SECTIONS).get(kind)
            if not section_title:
                continue
            qualified = (md.findtext("qualifiedname") or "").strip()
            if not qualified:
                qualified = (md.findtext("name") or "").strip()
            # Param types: <type> text plus any <array> suffix (Doxygen
            # stores `int foo[3]` as <type>int</type>...<array>[3]</array>;
            # without merging them our breathe disambiguator omits the `[3]`
            # and breathe reports "Unable to resolve function with
            # arguments (…, const double)" even though the function exists).
            def _param_type(p) -> str:
                t = _itertext(p.find("type"))
                arr = (p.findtext("array") or "").strip()
                return (t + arr) if arr else t
            param_types = [_param_type(p) for p in md.findall("param")]
            # Enum values + scoped-vs-unscoped flag (for Doxygen-style
            # synopsis rendering — one code block per enum with all values
            # inside `{ }`, instead of breathe's discrete signature blocks).
            enum_values = []
            is_strong = md.get("strong", "no") == "yes"
            if kind == "enum":
                for ev in md.findall("enumvalue"):
                    enum_values.append({
                        "name":        (ev.findtext("name") or "").strip(),
                        "initializer": (ev.findtext("initializer") or "").strip(),
                    })
            sections.setdefault(section_title, []).append({
                "id":          md.get("id", ""),
                "kind":        kind,
                "name":        (md.findtext("name") or "").strip(),
                "qualified":   qualified,
                "type":        _itertext(md.find("type")),
                "args":        (md.findtext("argsstring") or "").strip(),
                "param_types": param_types,
                "brief":       _itertext(md.find("briefdescription")),
                "enum_values": enum_values,
                "strong":      is_strong,
            })
    # Recurse into subgroups.
    children = []
    for ig in cd.findall("innergroup"):
        child = _build_api_hierarchy(ig.get("refid"), xml_dir, _seen)
        if child is not None:
            children.append(child)
    return {"name": name, "title": title, "detailed": detailed,
            "innerclasses": innerclasses, "sections": sections,
            "children": children}


def _md_escape_cell(text: str) -> str:
    """Make `text` safe for a single Markdown table cell."""
    # Newlines collapse to spaces, pipes escape, angle brackets stay.
    return (text or "").replace("\n", " ").replace("\r", " ") \
                       .replace("|", "\\|").strip()


# Per-member breathe directive selector. The full doxygengroup directive
# recursively inlines every <innerclass> + <innernamespace>, which is the
# *opposite* of how Doxygen's group page lays out (classes are links to
# separate pages there). Emitting one directive per member keeps each
# member's detail block scoped to itself and lets us push classes to their
# own per-class pages — see _write_class_stub.
_MEMBER_DIRECTIVE = {
    "enum":     "doxygenenum",
    "function": "doxygenfunction",
    "typedef":  "doxygentypedef",
    "variable": "doxygenvariable",
    "define":   "doxygendefine",
}
# Section header for each member kind's detail block. Mirrors what Doxygen
# emits on a group page (e.g. group__core__opencl.html has separate
# "Enumeration Type Documentation" and "Function Documentation" sections,
# not one collapsed "Detailed Description").
_MEMBER_DETAIL_SECTION = {
    "Typedefs":     "Typedef Documentation",
    "Enumerations": "Enumeration Type Documentation",
    "Functions":    "Function Documentation",
    "Variables":    "Variable Documentation",
    "Macros":       "Macro Definition Documentation",
}


def _sphinx_cpp_v4_id(qualified_name: str) -> str:
    """Build the Sphinx C++ domain v4 anchor id for a fully-qualified C++ name.

    Format: `_CPPv4` + `N` + each scope segment encoded as `<len><name>` + `E`.
    Examples:
      * `cv::_InputArray::KindFlag` → `_CPPv4N2cv11_InputArray8KindFlagE`
      * `cv::_InputArray::KindFlag::KIND_SHIFT`
        → `_CPPv4N2cv11_InputArray8KindFlag10KIND_SHIFTE`

    Why: when the synopsis is rendered as raw HTML below, each enum /
    enumerator name needs an `id=` so links to `#_CPPv4…` snap there. We
    compute it ourselves instead of asking Breathe to emit a (now hidden)
    `{doxygenenum}` directive — that would bloat the page and create
    duplicate ids. The mangling matches Sphinx's own so URLs follow the
    same convention as every other C++ symbol on the site."""
    parts = qualified_name.split("::")
    mangled = "".join(f"{len(p)}{p}" for p in parts)
    return f"_CPPv4N{mangled}E"


def _enum_synopsis_html(m: dict, strip_scope: str = "") -> list[str]:
    """Render an enum as a code-styled HTML block with clickable enum and
    enumerator names. Used on class pages in place of `_enum_synopsis_lines`
    (the plain Markdown code-fence version) so the names become real `<a>`
    anchors with stable URL hashes.

    Strategy:
      * Span classes (`k`, `n`, `o`, `p`, …) mirror Pygments' cpp-lexer output
        so the existing code-block CSS (`.highlight-cpp`, `.highlight pre`)
        gives this block the same colours and typography as a real
        ` ```cpp ` fence — no special chrome to maintain.
      * Each name is wrapped in `<a class="opencv-enum-link" id="<cpp v4 id>"
        href="#<cpp v4 id>">…</a>`. The id is both the anchor target and the
        click target, so URL hash updates correctly on click and a shared
        link (`…#_CPPv4N2cv11_InputArray8KindFlagE`) snaps the browser to
        the synopsis.
      * `strip_scope` removes the redundant `<class>::` prefix from displayed
        names — but it never changes the *id* we mangle (the id always uses
        the FULL qualified name, so cross-references stay stable).
    """
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
    # Displayed names lose the redundant scope; ids keep it.
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
    # IMPORTANT — the synopsis carries the click *sources* (`href=…`) but NOT
    # the anchor *targets* (no `id=…`). The actual `id="_CPPv4…"` anchors are
    # emitted by the Member Enumeration Documentation loop further down, so a
    # click from here scrolls *to* the detail block rather than self-anchoring.
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
    """Render an enum as a Doxygen-style code synopsis: one `enum {…}` block
    listing every value with its initializer. Used in place of breathe's
    `{doxygenenum}` directive (which emits a discrete signature box per
    enumerator). Doxygen's class/group page renders the enum as a single
    code-style box; this helper reproduces that.

    Value-name qualification follows Doxygen:
      * Scoped (`enum class`) → values prefixed with the enum's own
        qualified name.
      * Unscoped → values prefixed with the enum's *parent* scope
        (namespace or enclosing class), so they look like the C++ name
        you'd actually write in code.

    `strip_scope` lets the caller drop a redundant qualifier when the
    reader is already inside that scope. On the `cv::_InputArray` class
    page, passing `strip_scope="cv::_InputArray"` makes the synopsis read
    `enum KindFlag { KIND_SHIFT = 16, … }` instead of the noisy
    `enum cv::_InputArray::KindFlag { cv::_InputArray::KIND_SHIFT = 16, … }`.
    """
    qualified = m.get("qualified") or m["name"]
    is_strong = bool(m.get("strong"))
    keyword = "enum class" if is_strong else "enum"
    if is_strong:
        prefix = qualified + "::"
    elif "::" in qualified:
        prefix = qualified.rsplit("::", 1)[0] + "::"
    else:
        prefix = ""
    # Drop the leading `<strip_scope>::` from both the enum's own label and
    # the per-value prefix when the caller has identified that the page is
    # already inside that scope.
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
    """Disambiguator used after a qualified function name in `{doxygenfunction}`.
    Breathe expects `name(type1, type2, …)` with parameter names dropped (it
    matches against Doxygen's `<param><type>` text, and on the type-mangled
    signature — parameter names *and* default values are irrelevant to the
    match). Empty-arg functions get `()` — required for breathe to match
    correctly even for non-overloads.

    Trailing `const` is appended for const member functions: breathe matches
    the cv-qualifier as part of the declaration, so a bare `(types)` arg list
    fails to resolve a `const` method. Doxygen stores `int channels(int i=-1)
    const`; `{doxygenfunction} cv::_InputArray::channels(int)` (no const)
    parses to a non-const AST and reports "Unable to resolve function … with
    arguments (int)". Appending ` const` makes the directive arg-list match
    the stored declaration. Group-page members carry no `const` key, so free
    functions are unaffected."""
    types = ", ".join((t or "").strip() for t in member.get("param_types", []))
    sig = f"({types})"
    if member.get("const"):
        sig += " const"
    return sig


def _class_page_name(refid: str) -> str:
    """Filename (without extension) for the per-class page. We use the Doxygen
    refid verbatim so MyST cross-refs and internal links from breathe stay
    stable (breathe's class anchors are the C++-mangled `_CPPv4N…` form, not
    the refid — so there's no collision with the page name)."""
    return refid


def _read_class_data(refid: str, xml_dir: pathlib.Path) -> dict | None:
    """Walk a class/struct compound XML and return everything the per-class
    page needs: brief + detailed for the class itself, and a list of
    members grouped by sectiondef kind. Returns None if the XML file is
    missing or unparseable — callers should fall back to a bare
    `{doxygenclass}` directive in that case.

    Each member dict carries the same fields `_build_api_hierarchy`
    captures, plus the `protection`, `static`, `const`, `virtual`,
    `explicit` flags from memberdef attributes (needed to render
    Doxygen-style annotations in the summary table)."""
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
                        # Per-value brief description (used in the Member
                        # Enumeration Documentation list so each enumerator
                        # row carries its own description, like docs.opencv.org).
                        "brief":       _itertext(ev.find("briefdescription")).strip(),
                    })
            items.append({
                "id":          md.get("id", ""),
                "kind":        mkind,
                "name":        name,
                "qualified":   qualified or name,
                "type":        _itertext(md.find("type")),
                "args":        (md.findtext("argsstring") or "").strip(),
                "param_types": [_param_type(p) for p in md.findall("param")],
                "brief":       _itertext(md.find("briefdescription")),
                "static":      md.get("static") == "yes",
                "virt":        md.get("virt", "non-virtual"),
                "const":       md.get("const") == "yes",
                "explicit":    md.get("explicit") == "yes",
                "enum_values": enum_values,
                "strong":      md.get("strong", "no") == "yes",
            })
        if items:
            sections[skind] = items

    detailed_el = cd.find("detaileddescription")
    detailed_paras = []
    if detailed_el is not None:
        for p in detailed_el.findall("para"):
            txt = _itertext(p)
            if txt:
                detailed_paras.append(txt)

    return {
        "name":     (cd.findtext("compoundname") or "").strip(),
        "brief":    _itertext(cd.find("briefdescription")),
        "detailed": "\n\n".join(detailed_paras),
        "sections": sections,
    }


def _find_collaboration_svg(refid: str, html_root: pathlib.Path) -> pathlib.Path | None:
    """Locate the Doxygen-generated collaboration-diagram SVG for a class.

    Our XML pipeline (CMakeLists.txt's `Doxyfile-xml`) sets
    `COLLABORATION_GRAPH = NO` and friends — graph elements in XML would be
    forwarded by breathe as `graphviz` docutils nodes, which need an
    extension we don't load. So the diagram never reaches the XML.

    The *legacy* Doxygen HTML build (the `doxygen` target, separate from
    `sphinx-xml`) still renders it as `<refid>__coll__graph.svg`, written
    into a content-addressed subdir because the legacy Doxyfile keeps
    `CREATE_SUBDIRS=YES`. The HTML tree sits next to the XML tree
    (`…/doxygen/html` ⟷ `…/doxygen/xml`). We read that asset read-only —
    nothing in the Doxygen output is modified. Returns None when the legacy
    HTML build hasn't run (graphs simply stay absent, no crash)."""
    if not html_root.is_dir():
        return None
    matches = sorted(html_root.rglob(f"{refid}__coll__graph.svg"))
    return matches[0] if matches else None


def _svg_make_transparent(text: str) -> str:
    """Light-mode variant: only the full-canvas backdrop is made transparent
    so the white page shows through (native Doxygen look). Graphviz paints the
    canvas as a single `fill="white" stroke="transparent"` polygon."""
    return text.replace('fill="white" stroke="transparent"',
                        'fill="none" stroke="transparent"', 1)


def _svg_dark_variant(text: str) -> str:
    """Dark-mode variant: recolour the (light) Doxygen SVG into a dark diagram
    matching docs.opencv.org — transparent canvas (page slate shows through),
    dark node fills, light borders/text, lightened connector arrows. We recolour
    the SVG itself (rather than a CSS `filter: invert`, which turns the large
    white node boxes solid black) so the result blends with the dark page.

    Order matters: blank the backdrop first, *then* repaint the remaining white
    node fills, so the two `fill="white"` cases don't collide."""
    import re as _re
    text = _svg_make_transparent(text)              # backdrop → transparent
    # Strokes + arrowheads first (covers Doxygen 1.12 `#666666` and legacy
    # `black` / `#404040`). Done before the per-node pass so per-node text
    # re-colouring isn't confused by stroke values.
    for _old in ('stroke="#666666"', 'stroke="black"', 'stroke="#404040"'):
        text = text.replace(_old, 'stroke="#c9d1d9"')
    text = text.replace('fill="#666666"', 'fill="#c9d1d9"')   # arrowheads match
    # Per-node pass. A Doxygen class node carries a grey HEADER STRIP at the
    # top — `fill="#999999"` in 1.12 (or `#bfbfbf` in 1.9). Per request, the
    # header strip is repainted WHITE so it stands out in dark mode; the
    # class BODY (originally `fill="white"`) becomes the dark slate. The
    # class title (the first <text> in the node) sits on the white strip, so
    # it has to be DARK ink; member texts below it sit on the slate body and
    # stay WHITE. Template-style nodes that don't have a header strip just
    # get their body re-coloured and keep white text.
    def _process_node(m):
        block = m.group(0)
        # Body fills → TRANSPARENT so the page background shows through. This
        # keeps every box exactly the same colour as `cv::_InputArray`'s body
        # (which has no body polygon in the Doxygen SVG and therefore already
        # shows the page directly) — no two-tone blues side-by-side.
        block = block.replace('fill="white"', 'fill="none"')
        block = block.replace('fill="grey"',  'fill="none"')
        # Header strip → a clear DARK GREY card on top of the transparent body.
        # Distinct from the page bg, distinct from the borders, readable with
        # white text. Covers Doxygen 1.12 (`#999999`) and legacy 1.9 (`#bfbfbf`).
        block = block.replace('fill="#999999"', 'fill="#2d333b"')
        block = block.replace('fill="#bfbfbf"', 'fill="#2d333b"')
        # All text on either the (transparent body = dark page) or the dark-grey
        # header strip is plain WHITE — both backgrounds are dark, no special
        # title-on-white-strip handling needed any more.
        block = block.replace('<text ', '<text fill="#ffffff" ')
        return block
    text = _re.sub(r'<g[^>]*class="node"[^>]*>.*?</g>',
                   _process_node, text, flags=_re.DOTALL)
    # Catch any grey fills that live OUTSIDE `<g class="node">` blocks
    # (graphviz cluster/subgraph constructs). Repaint them with the same dark
    # palette: `grey`/keyword → transparent (match the body treatment),
    # named greys (`#999999`/`#bfbfbf`) → dark-grey strip colour.
    text = text.replace('fill="grey"',     'fill="none"')
    text = text.replace('fill="#999999"',  'fill="#2d333b"')
    text = text.replace('fill="#bfbfbf"',  'fill="#2d333b"')
    # Edge labels (#flags / #sz / …) live outside <g class="node"> blocks
    # and haven't been re-coloured yet — paint them white so they read on
    # the dark page. The negative lookahead avoids double-prefixing the
    # per-node texts we already handled above.
    text = _re.sub(r'<text (?!fill)', '<text fill="#ffffff" ', text)
    # Edge lines that Doxygen 1.12 paints blue (`#63b8ff`) stay blue — that
    # reads cleanly on the dark page and matches docs.opencv.org's look.
    # NOTE: the global `<text> → white` block below the original loop is now
    # redundant — the per-node + lookahead passes above handle all texts.
    return text


def _patch_namespace_xml_for_breathe(xml_dir: pathlib.Path,
                                     out_dir: pathlib.Path) -> None:
    """Mirror `xml_dir` into `out_dir` via symlinks, then rewrite every
    *non-group* compound XML to inline `<memberdef>` blocks that exist only
    in the group XML.

    Why: Doxygen lists functions declared inside `@addtogroup` regions as
    `<member refid="group__…">` in the parent namespace XML (for `cv::*`
    free functions) or the parent file XML (for global `hal_ni_*`-style
    functions) — *without* a full `<memberdef>` block. The memberdef lives
    only in the group XML file. breathe's function-by-name lookup walks
    `<memberdef>` blocks in namespace/file XMLs and ignores bare refs, so
    directives like `{doxygenfunction} cv::log` or
    `{doxygenfunction} hal_ni_merge8u` fail with "Cannot find function".

    Patching: for each `<member refid>` in a target compound's sectiondef
    whose id targets `group__…`, we open the group XML, find the matching
    `<memberdef id="…">`, and append it into the compound's sectiondef.
    The original XML on disk is untouched."""
    import xml.etree.ElementTree as _ET
    import os as _osmod, shutil as _shutil
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mirror every file from xml_dir into out_dir as a symlink. Cleaning
    #    out_dir each time keeps this idempotent across rebuilds (Doxygen
    #    XML changes are picked up because the symlinks resolve fresh).
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

    # 2) Cache for parsed group XMLs (each is read once even if referenced
    #    by many compounds).
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

    # 3) For each non-group compound XML (namespace, file, dir, …) patch
    #    `<member refid>` entries that point at group memberdefs.
    #    `index.xml` is the project index (not a compound) → skip it.
    #    `class*.xml`/`struct*.xml`/`union*.xml` already carry full
    #    memberdefs for their methods, but they may *also* have
    #    `<member refid>` from @addtogroup tagged methods — patch them too.
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
                # Member refids inside groups look like
                # "group__core__utils__softfloat_1ga…". The compound id is
                # everything before "_1" (which separates member from
                # compound in Doxygen's id scheme).
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


__all__ = [
    "_itertext", "_MEMBERDEF_SECTIONS", "_read_class_brief",
    "_build_api_hierarchy", "_md_escape_cell", "_MEMBER_DIRECTIVE",
    "_MEMBER_DETAIL_SECTION", "_sphinx_cpp_v4_id", "_enum_synopsis_html",
    "_enum_synopsis_lines", "_function_signature", "_class_page_name",
    "_read_class_data", "_find_collaboration_svg", "_svg_make_transparent",
    "_svg_dark_variant", "_patch_namespace_xml_for_breathe",
]
