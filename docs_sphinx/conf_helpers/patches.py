# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""Runtime patches for Sphinx C++ domain and breathe; applied at import."""
from __future__ import annotations

import re

def _patch_cpp_xref_resolver():
    """Work around Sphinx 8.1.x parentSymbol assert in _resolve_xref_inner."""
    try:
        from sphinx.domains.cpp import CPPDomain
    except ImportError:
        return
    original = CPPDomain._resolve_xref_inner

    def guarded(self, env, fromdocname, builder, typ, target, node, contnode):
        try:
            return original(self, env, fromdocname, builder, typ, target,
                            node, contnode)
        except AssertionError:
            return None, None
    CPPDomain._resolve_xref_inner = guarded

    # Drop breathe unresolvable-xref log noise; text still renders.
    import logging
    _UNRESOLVED_XREF_PATTERNS = (
        "Unable to resolve function",
        "Unable to resolve class",
        "Cannot find function",
        "Cannot find class",
        "Cannot find variable",
        "Cannot find typedef",
        "Cannot find enum",
        "Cannot find enumerator",
        "Cannot find define",
        "Duplicate C++ declaration",
    )

    class _UnresolvedXrefFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not any(p in msg for p in _UNRESOLVED_XREF_PATTERNS)

    _filt = _UnresolvedXrefFilter()
    for _logger_name in ("sphinx", "docutils"):
        logging.getLogger(_logger_name).addFilter(_filt)


_patch_cpp_xref_resolver()


def _silence_breathe_anon_enum_warning():
    """Mute Sphinx parser warning on Doxygen's anonymous nested enums."""
    import logging
    class _AnonEnumFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not (
                "Invalid C++ declaration" in msg
                and "Expected identifier in nested name" in msg
            )
    for _name in ("sphinx", "docutils"):
        logging.getLogger(_name).addFilter(_AnonEnumFilter())


_silence_breathe_anon_enum_warning()


def _patch_breathe_operator_signatures():
    """Fix breathe {doxygenfunction} mis-splitting operator overloads."""
    try:
        import breathe.directives.function as _bf
    except ImportError:
        return

    def _split_operator(s: str):
        rp = s.rfind(")")
        if rp == -1:
            return None
        depth, j = 0, rp
        while j >= 0:
            if s[j] == ")":
                depth += 1
            elif s[j] == "(":
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        if j < 0:
            return None
        func_part, args_part = s[:j].strip(), s[j:]
        k = func_part.find("::operator")
        if k != -1:
            return func_part[:k], func_part[k + 2:], args_part
        if "::" in func_part:
            ns, fn = func_part.rsplit("::", 1)
            return ns, fn, args_part
        return "", func_part, args_part

    class _Shim:
        __slots__ = ("_g",)

        def __init__(self, g1, g2, g3):
            self._g = (None, g1, g2, g3)

        def group(self, i=0):
            return self._g[i]

    class _OperatorAwareRe:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def __getattr__(self, name):
            return getattr(self._real, name)

        def match(self, pattern, string, *args, **kwargs):
            m = self._real.match(pattern, string, *args, **kwargs)
            if (m is not None and getattr(m.re, "groups", 0) >= 3
                    and "::operator" in string):
                res = _split_operator(string)
                if res is not None:
                    ns, fn, ar = res
                    return _Shim(ns or None, fn, ar)
            return m

    if not isinstance(_bf.re, _OperatorAwareRe):
        _bf.re = _OperatorAwareRe(_bf.re)


_patch_breathe_operator_signatures()


def _patch_breathe_docsect():
    """Render title-less docSectN nodes breathe 4.36 drops."""
    try:
        from breathe.renderer import sphinxrenderer as _bsr
    except ImportError:
        return
    _methods = _bsr.SphinxRenderer.methods
    if getattr(_methods.get("docsect1"), "_opencv_docsect_patch", False):
        return
    _orig_visit = _methods["docsect1"]

    def _visit_docsectN(self, node):
        if not getattr(node, "title", None):
            return self.render_iterable(node.content_)
        return _orig_visit(self, node)

    _visit_docsectN._opencv_docsect_patch = True
    for _kind in ("docsect1", "docsect2", "docsect3"):
        _methods[_kind] = _visit_docsectN


_patch_breathe_docsect()


def _silence_orphan_toctree_warning():
    """Mute toctree-orphan warning for intentionally unlinked external pages."""
    import logging

    class _OrphanFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "included in any toctree" not in record.getMessage()

    for _name in ("sphinx", "docutils"):
        logging.getLogger(_name).addFilter(_OrphanFilter())


_silence_orphan_toctree_warning()


def _patch_sidebar_section_root():
    """Root the left sidebar at a page's own top-level section.

    A page in two toctrees (e.g. a `cuda*` extra module also grouped under main
    `cuda`) gets a last-wins parent from `_get_toctree_ancestors`, rooting its
    sidebar at the foreign section. Re-pick the parent sharing the longest path
    prefix (same section) so the full sibling list shows."""
    try:
        import pydata_sphinx_theme.toctree as _pt
        from sphinx.environment.adapters.toctree import TocTree
    except ImportError:
        return

    def _section_aware_ancestor(app, pagename, startdepth):
        ti = app.env.toctree_includes
        cand: dict[str, list[str]] = {}
        for _p, _children in ti.items():
            for _c in _children:
                cand.setdefault(_c, []).append(_p)

        def _shared(parent: str, child: str) -> int:
            a, b, i = parent.split("/"), child.split("/"), 0
            while i < len(a) and i < len(b) and a[i] == b[i]:
                i += 1
            return i

        ancestors: list[str] = []
        d = pagename
        while d not in ancestors:
            ps = cand.get(d)
            if not ps:
                break
            ancestors.append(d)
            d = max(ps, key=lambda p: _shared(p, d))
        try:
            out = ancestors[-startdepth]
        except IndexError:
            out = None
        # Childless root => empty sidebar. Fall back to the dead-end ancestor `d`
        # when it's a same-section page with children, else the section api_root.
        if out is None or not ti.get(out):
            _sec = pagename.split("/", 1)[0]
            _base = pagename.rsplit("/", 1)[-1]
            # Doxygen file/dir-reference pages are orphan utilities, not module
            # content: leave None to suppress the sidebar rather than root at api_root.
            if re.search(r"_8\w+$", _base) or _base.startswith("dir_"):
                out = None
            elif d != pagename and ti.get(d) and d.split("/", 1)[0] == _sec:
                out = d
            elif ti.get(_sec + "/api_root"):
                out = _sec + "/api_root"
        return out, TocTree(app.env)

    _pt._get_ancestor_pagename = _section_aware_ancestor


_patch_sidebar_section_root()


def _patch_sphinx_toctree_ancestors():
    """Fix which branch the collapsed startdepth=0 sidebar auto-expands.

    Sphinx picks it via `_get_toctree_ancestors`, whose last-wins parent map
    mis-picks for a page in two toctrees (e.g. a `cuda*` extra module also under
    main `cuda`), so its section won't expand. Prefer the parent sharing the
    longest path prefix (same section)."""
    try:
        import sphinx.environment.adapters.toctree as _st
    except ImportError:
        return

    def _section_aware(toctree_includes, docname):
        cand: dict[str, list[str]] = {}
        for _p, _children in toctree_includes.items():
            for _c in _children:
                cand.setdefault(_c, []).append(_p)

        def _shared(parent: str, child: str) -> int:
            a, b, i = parent.split("/"), child.split("/"), 0
            while i < len(a) and i < len(b) and a[i] == b[i]:
                i += 1
            return i

        ancestors: list[str] = []
        d = docname
        while d not in ancestors:
            ps = cand.get(d)
            if not ps:
                break
            ancestors.append(d)
            d = max(ps, key=lambda p: _shared(p, d))
        return dict.fromkeys(ancestors).keys()

    _st._get_toctree_ancestors = _section_aware


_patch_sphinx_toctree_ancestors()


def register_global_sidebar(app):
    """Make the sidebar list ALL top-level sections (startdepth=0), current one
    auto-expanded, instead of only the active section's subtree.

    Wrap the theme's `generate_toctree_html` (its sole sidebar nav generator) to
    force startdepth=0, connecting after the theme's html-page-context handler so
    it keeps its own template and we flip only this arg. This makes
    `_patch_sidebar_section_root` a no-op (its lookup only runs when startdepth != 0)."""
    def _globalize(app, pagename, templatename, context, doctree):
        gen = context.get("generate_toctree_html")
        if not callable(gen):
            return
        def wrapped(kind, startdepth=0, show_nav_level=0, **kwargs):
            # collapse=True: expand only the current branch. Without it,
            # startdepth=0 renders the whole tree on every page (slow + bloated).
            kwargs["collapse"] = True
            return gen(kind, startdepth=0, show_nav_level=0, **kwargs)
        context["generate_toctree_html"] = wrapped
    app.connect("html-page-context", _globalize, priority=900)
