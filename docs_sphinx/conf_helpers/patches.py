"""Runtime patches for Sphinx C++ domain and breathe; applied at import."""
from __future__ import annotations


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
