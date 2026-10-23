#!/usr/bin/env python
"""Docs directive-migration regression tests (issue #29090).

The docs_sphinx rewrite converts the legacy Doxygen-style tutorial markers
(``@code`` / ``@endcode`` / ``@snippet`` / ``@youtube`` / ``@tableofcontents``
/ ``@prev_tutorial`` / ``@next_tutorial``) into MyST/`myst-parser` compatible
constructs (`#` headings, ``:::{div}`` cards, ``$$…$$`` math, fenced ```` ``` ````,
````{doxysnippet}`` and ````{raw} html```` blocks).

These tests pin the migrated pages so a reversion to the old markers fails
CI. They scan the Markdown sources directly, so no built OpenCV is required.
"""
from __future__ import annotations

import pathlib
import re
import unittest

# Repo root = two levels up from this file (…/docs_sphinx/test_*.py).
HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent

# Legacy markers that must no longer appear in migrated tutorial prose.
# Each entry is a compiled regex matched against a single line.
_LEGACY_PATTERNS = [
    re.compile(r"^\s*@code(?:\{[^}]*\})?\s*$"),     # @code{.cpp} / @code
    re.compile(r"^\s*@endcode\s*$"),                # @endcode
    re.compile(r"^\s*@snippet\b"),                  # @snippet <path> [label]
    re.compile(r"^\s*@youtube\{[^}]*\}\s*$"),       # @youtube{…}
    re.compile(r"^\s*@tableofcontents\s*$"),        # @tableofcontents
    re.compile(r"^\s*@prev_tutorial\{[^}]*\}\s*$"), # @prev_tutorial{…}
    re.compile(r"^\s*@next_tutorial\{[^}]*\}\s*$"), # @next_tutorial{…}
]

# The specific page migrated by #29090.
CAMERA_CALIBRATION = (
    REPO_ROOT
    / "doc" / "tutorials" / "calib3d" / "camera_calibration"
    / "camera_calibration.markdown"
)


def _legacy_hits(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (line_no, text) for every line carrying a legacy marker."""
    hits = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("@"):
            continue
        if any(p.match(line) for p in _LEGACY_PATTERNS):
            hits.append((i, stripped))
    return hits


class TestTutorialDirectiveMigration(unittest.TestCase):
    def test_camera_calibration_no_legacy_directives(self) -> None:
        hits = _legacy_hits(CAMERA_CALIBRATION)
        self.assertEqual(
            hits, [],
            "camera_calibration.markdown still has legacy @directives: "
            + ", ".join(f"@L{l}:{t}" for l, t in hits),
        )

    def test_camera_calibration_has_myst_constructs(self) -> None:
        text = CAMERA_CALIBRATION.read_text(encoding="utf-8")
        # MyST card, math fences, code fences, doxysnippet and raw-html iframe.
        self.assertIn(":::{div} opencv-meta-table", text)
        self.assertIn("```cpp", text)
        self.assertIn("```xml", text)
        self.assertIn("```{doxysnippet}", text)
        self.assertIn("```{raw} html", text)
        self.assertIn("responsive-iframe", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
