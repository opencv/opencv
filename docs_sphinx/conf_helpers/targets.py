# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Keep repeated API documentation's fragment targets local to each page."""
from functools import lru_cache
import html
import re

from .state import SPHINX_INPUT_ROOT

_LABEL = re.compile(r"^\(([^\n]+)\)=\s*$", re.MULTILINE)
_HEADING_LABEL = re.compile(r"\{#([\w-]+)\}")


@lru_cache(maxsize=1)
def _label_owners():
    # All workers use the same sorted input, independent of read order. Prefer
    # group/class pages over namespace summaries for a global reference target.
    owners = {}
    paths = [p for folder in ("main_modules", "extra_modules")
             for p in (SPHINX_INPUT_ROOT / folder).glob("*.md")]
    from .translate import _apply_outside_code
    for path in sorted(paths, key=lambda p: (p.name.startswith("namespace_"), str(p))):
        docname = path.relative_to(SPHINX_INPUT_ROOT).with_suffix("").as_posix()
        text = path.read_text(encoding="utf-8")
        labels = []
        def collect(chunk):
            labels.extend(m.group(1) for m in _LABEL.finditer(chunk))
            labels.extend(_HEADING_LABEL.findall(chunk))
            return chunk
        _apply_outside_code(text, collect)
        for label in labels:
            owners.setdefault(label.lower(), docname)
    return owners


def localize_api_labels(text, docname):
    """Register a label globally once; retain duplicate pages' HTML fragments.

    A member can appear in a group, namespace and class view. Re-registering
    its MyST label makes the global target depend on parallel read order.
    Links to the other views still need their page-local fragment IDs.
    """
    from docutils import nodes
    owners = _label_owners()
    seen = set()
    local_labels = {}

    def replace(match):
        label = match.group(1)
        key = label.lower()
        if key in seen:
            return ""
        seen.add(key)
        if owners.get(key, docname) == docname:
            return match.group(0)
        scoped = f"{docname}:{label}"
        local_labels[key] = scoped
        target = nodes.make_id(label)
        return f'({scoped})=\n<a id="{html.escape(target, quote=True)}"></a>\n'

    # Code samples can contain label-like syntax and must remain untouched.
    from .translate import _apply_outside_code
    text = _apply_outside_code(text, lambda chunk: _LABEL.sub(replace, chunk))

    def local_link(match):
        label = match.group(1)
        return "#" + local_labels.get(label.lower(), label)

    # A bare #label would otherwise resolve to the canonical page's global
    # label. Use the scoped label for this view's own summary/detail links.
    return _apply_outside_code(
        text, lambda chunk: re.sub(r"(?<=\]\()#([^\s)]+)(?=\))", local_link, chunk))
