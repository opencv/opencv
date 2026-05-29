"""Runtime patches for Sphinx's C++ domain and breathe warnings.

Importing this module applies them (each helper invokes itself at import):
guard the C++ xref resolver against an upstream assertion, and silence the
benign anonymous-enum parse warning breathe triggers. conf.py imports it for
effect.
"""
from __future__ import annotations


def _patch_cpp_xref_resolver():
    """Guard the C++ domain's xref resolver against an upstream assertion on
    template-class cross-references. Sphinx 8.1.x asserts `parentSymbol`
    inside `_resolve_xref_inner`; some breathe-emitted class-page xrefs
    (e.g. `cv::Affine3<T>`) trigger that path with no parent symbol.
    Treat it as an unresolved xref instead of crashing the whole build."""
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

    # Log-filter half (PR #7): when Breathe renders a detailed description it
    # emits a `pending_xref` for every `<ref>` in the prose; many don't resolve
    # to a symbol on our pages (templates, conversion operators, overloads the
    # matcher gives up on) so the C++ domain logs "Unable to resolve …" /
    # "Cannot find …" — dozens per page. The text still renders fine (falls back
    # to plain `<code>`), so these are pure noise. Drop just those shapes;
    # everything else flows through untouched.
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
        # Breathe sometimes re-declares an already-known C++ symbol across an
        # incremental rebuild; the duplicate is harmless (Sphinx ignores it).
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
    """Suppress the docutils "Invalid C++ declaration: Expected identifier
    in nested name." warning that breathe triggers when rendering an
    *anonymous* nested enum inside a struct (e.g. `cv::MatShape` has an
    enum whose `<name>` element is empty — Doxygen XML allows it, but the
    Sphinx C++ domain parser rejects the resulting declaration).

    The render is otherwise fine (the enum values still appear); only the
    parse-time warning is noise. We filter it via a Python logging filter
    rather than monkey-patching the parser so the same fix survives Sphinx
    version bumps."""
    import logging
    class _AnonEnumFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not (
                "Invalid C++ declaration" in msg
                and "Expected identifier in nested name" in msg
            )
    # docutils warning messages route through both 'sphinx' and 'docutils'
    # loggers depending on entry point; attach to both for coverage.
    for _name in ("sphinx", "docutils"):
        logging.getLogger(_name).addFilter(_AnonEnumFilter())


_silence_breathe_anon_enum_warning()


def _patch_breathe_operator_signatures():
    """Fix breathe's ``{doxygenfunction}`` parsing of operator overloads.

    ``breathe.directives.function`` splits the directive argument with the regex
    ``(?:(ns)::)?([^(]+)(.*)`` — the function-name group ``[^(]+`` stops at the
    FIRST ``(``. For ``operator()`` that paren is part of the operator's own
    name, so ``cv::Mat_::operator()(const int *)`` is mis-split into name
    ``operator`` + args ``()(const int *)`` and the lookup fails with
    "Cannot find function cv::Mat_::operator". Conversion operators
    ("operator std::vector< _Tp >") are likewise broken by the ``::`` inside the
    target type.

    ``operator()`` can't be expressed in a form that regex accepts, so we wrap
    the ``re`` reference used *inside function.py* (never the global ``re``) and,
    for member-operator specs only, re-derive (namespace, function_name, args)
    from the rightmost balanced parameter list. Everything else is delegated to
    the real ``re.match`` untouched. Idempotent; degrades to a no-op if breathe
    is absent or its internals change shape."""
    try:
        import breathe.directives.function as _bf
    except ImportError:
        return

    def _split_operator(s: str):
        # args = rightmost top-level (...) group (+ any trailing qualifiers like
        # " const"); function name = everything before it; namespace = the part
        # before "::operator" (so "::" inside a conversion target stays intact).
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
            # Only the 3-group function-directive split, and only for member
            # operators (where the naive first-paren split goes wrong).
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
    """Render Doxygen ``docSect{1,2,3}`` nodes that carry no title.

    Cherry-picked from PR #30. Breathe 4.36 assumes every ``docSectN`` node
    has a title, but Doxygen 1.12 emits ``docSect2TypeSub`` (and deeper)
    elements whose title is optional. When the title is absent breathe raises
    while rendering a "Detailed Description", dropping the section's body. Wrap
    the section visitor so a title-less node just renders its content; nodes
    that do have a title fall through to breathe's original visitor.

    Idempotent and a no-op when breathe is unavailable.
    """
    try:
        from breathe.renderer import sphinxrenderer as _bsr
    except ImportError:
        return
    _methods = _bsr.SphinxRenderer.methods
    # Already patched? our wrapper sets this marker.
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
    """Suppress Sphinx's "document isn't included in any toctree" warning.

    The wrapper renders an external tree (opencv/doc/) we must not edit, which
    contains intentionally-unlinked pages: moved-redirect stubs and sub-TOCs
    whose parent links the children directly. ``_source_read`` already marks
    such pages ``:orphan:`` (the correct fix) — but that only takes effect when
    Sphinx re-reads the source. On an INCREMENTAL build the cached environment
    is reused (Sphinx tracks conf.py, not conf_helpers/), so the consistency
    check still emits the warning from the stale doctree. The condition isn't
    actionable here (we can't add the missing link without editing doc/), so we
    filter the message — same approach as the breathe / cpp-domain noise above.
    A clean build silences it via the ``:orphan:`` metadata regardless."""
    import logging

    class _OrphanFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "included in any toctree" not in record.getMessage()

    for _name in ("sphinx", "docutils"):
        logging.getLogger(_name).addFilter(_OrphanFilter())


_silence_orphan_toctree_warning()
