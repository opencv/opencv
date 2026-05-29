"""Per-class "Examples" cross-reference system (cherry-picked from PR #7).

Reconstructs Doxygen's EXAMPLE_PATH / `@example` machinery natively, since the
upstream Doxyfile is out of scope for the Sphinx wrapper: scan every module
header for `@example` declarations, index the sample-source files, match each
class to the samples that reference it, render an "Examples" footer onto the
class page, and emit one standalone Sphinx page per referenced sample.

`stubs.py` calls `_find_examples_for_class` while writing each class stub and
`_generate_example_pages` once afterwards; `_render_examples_block` turns the
matches into the HTML footer. All heavy scanning (`_EXAMPLE_FILES`,
`_DOXY_EXAMPLE_DECLS`, `_TUTORIAL_SAMPLE_REFS`) runs once at import time.

Shared paths (OPENCV_ROOT, CONTRIB_ROOT, DOC_ROOT) come from `state`.
"""
from __future__ import annotations
import re, pathlib
from .state import *

# ---------------------------------------------------------------------------
# Per-class "Examples" cross-reference (Doxygen EXAMPLE_PATH stand-in).
#
# Upstream `opencv/doc/Doxyfile.in` is out-of-scope per DOC_OPTIMIZATION_V2.md,
# so Doxygen's EXAMPLE_PATH feature (which is what populates `<exampleref>`
# tags in the XML and drives docs.opencv.org's per-class "Examples:" footer)
# can't be enabled there. Without that data, Breathe has nothing to render.
#
# Reconstruct the cross-reference natively: tokenize every sample source file
# once, then per-class look up which files mention the class's name. Result
# is rendered as an HTML `<dl class="opencv-examples">` block appended to the
# Detailed Description in `_write_class_stub`.
#
# MUST be defined before the `_generate_api_stubs(...)` call below — that
# call runs at import time and reaches `_write_class_stub`, which references
# `_find_examples_for_class` by name. (Previously this block lived next to
# `_SNIPPET_INDEX` further down and the build crashed with NameError; that
# index is read later, from `source-read`-time `_translate`, so it could
# stay below the call. Ours can't.)
#
# Token-set matching (not raw substring) keeps the result tight — we won't
# match `_InputArray` inside a comment word like `myInputArrayHelper`, and we
# won't false-positive on hits inside string literals that happen to be
# spelled the same. (The same `[A-Za-z_][A-Za-z0-9_]*` rule Doxygen itself
# uses to identify cross-reference candidates.)
# ---------------------------------------------------------------------------
_EXAMPLE_SOURCE_EXTENSIONS = {
    # Actual sample-program file types only. Headers (.h/.hpp) are excluded
    # on purpose: docs.opencv.org's "Examples:" footer lists program files,
    # not the helper headers some samples ship alongside their main.cpp —
    # an `_InputArray` parameter declaration in `frameProcessor.hpp` isn't
    # an "example of how to use _InputArray", it's API plumbing for the
    # surrounding app. Same reason CMake/shell/config files aren't included.
    ".cpp", ".cc", ".cxx", ".c",
    ".py", ".java", ".js", ".ts",
}
# Per-extension Pygments language tag for the generated example pages.
_EXAMPLE_LANGUAGE = {
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c": "c",
    ".py": "python", ".java": "java",
    ".js": "javascript", ".ts": "typescript",
}
# Canonical-example allow-list (path prefixes under OPENCV_ROOT). Mirrors
# the practical scope of docs.opencv.org's "Examples:" footer:
#   * `samples/cpp/<file>`         — top-level C++ samples (pca.cpp, …)
#   * `samples/cpp/tutorial_code/` — code paired with written tutorials
#   * `samples/python/`            — Python samples (top-level + nested)
#   * `samples/java/`              — Java samples
#   * `samples/dnn/`               — DNN module samples
#   * `samples/gpu/`               — GPU/CUDA samples
#   * `opencv_contrib/modules/<m>/samples/**` — contrib sample trees
#     (HOGDescriptor + peopledetect.cpp moved here in OpenCV 5.x; without
#     including contrib samples we'd lose every contrib-class example
#     even though those samples ARE the canonical examples for their
#     module — contrib modules don't keep a separate tutorial tree.)
#
# Deliberately excluded:
#   * `samples/cpp/snippets/`      — short snippets inlined in tutorials, not
#                                    standalone examples
#   * `samples/cpp/example_cmake/` — build-system demo, no API surface
#   * `samples/tapi/`, `sycl/`, `opencl/`, `opengl/`, `directx/`, `va_intel/`,
#     `winrt*/`, `wp8/`, `android/`, `swift/`, `semihosting/`, `hal/`, `gdb/`
#                                  — platform-/niche-specific; a passing
#                                    reference to a class here isn't a useful
#                                    "example of how to use this class" for
#                                    a general docs reader.
_EXAMPLE_INCLUDE_SUBTREES = (
    "samples/cpp/tutorial_code/",
    "samples/python/",
    "samples/java/",
    "samples/dnn/",
    "samples/gpu/",
)


def _is_canonical_example(rel_path: str) -> bool:
    """True iff this repo-relative path is a canonical example.

    Files directly under `samples/cpp/` qualify (e.g. samples/cpp/pca.cpp)
    but their sibling subdirectories (snippets/, example_cmake/) don't —
    those are inline-tutorial snippets / build-system demos, not example
    programs. The other allow-listed subtrees qualify recursively.

    Anything under `opencv_contrib/modules/<m>/samples/` also qualifies —
    contrib modules don't carry their own tutorial-markdown trees, so by
    convention their `samples/` directory IS the cross-reference target
    for classes documented in that module. peopledetect.cpp lives in
    `opencv_contrib/modules/xobjdetect/samples/` as of OpenCV 5.x's
    HOGDescriptor relocation — it would be invisible otherwise.
    """
    if any(rel_path.startswith(p) for p in _EXAMPLE_INCLUDE_SUBTREES):
        return True
    if rel_path.startswith("samples/cpp/"):
        rest = rel_path[len("samples/cpp/"):]
        return "/" not in rest        # direct child of samples/cpp/, not nested
    # Contrib sample paths show up as `opencv_contrib/modules/<m>/samples/...`
    # because they live outside OPENCV_ROOT — the scanner labels them with
    # that display prefix below.
    if re.match(r"opencv_contrib/modules/[^/]+/samples/", rel_path):
        return True
    return False


def _example_pagename(display_path: str) -> str:
    """`samples/cpp/pca.cpp` → `samples_cpp_pca_cpp` (Sphinx-safe basename).

    Matches the per-page filename we generate under `examples/<name>.md`.
    Lowercased + non-alphanumerics collapsed to underscores so the path is
    valid on every filesystem and produces a clean URL.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", display_path).strip("_").lower()


# `(display path, source filesystem path, identifier tokens)` per scanned
# file. Built once at import time (a few hundred files, sub-second cost) and
# shared by every class-stub render.
_EXAMPLE_FILES: list[tuple[str, pathlib.Path, frozenset[str]]] = []
_EXAMPLE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Walk core `samples/` plus every contrib module's `samples/` tree.
# Critically NOT scoped to `CONTRIB_MODULES`: that list controls toctree
# wiring + breathe scope for the contrib build, but Examples are just
# informational cross-references — they should surface peopledetect.cpp
# regardless of whether `xobjdetect`'s tutorial tree is being built this
# run. Generated example pages live under our own `examples/` directory
# (matched by `include_patterns`), so the renderer always produces them
# even for modules not in `CONTRIB_MODULES`. Contrib paths are displayed
# with the `opencv_contrib/...` prefix so the rendered line on a class
# page reads unambiguously ("this example lives in a contrib module").
_example_scan_roots: list[tuple[pathlib.Path, str]] = [
    (OPENCV_ROOT / "samples", "samples"),
]
if CONTRIB_ROOT.is_dir():
    for _module_dir in sorted(CONTRIB_ROOT.iterdir()):
        if not _module_dir.is_dir():
            continue
        _contrib_samples = _module_dir / "samples"
        if _contrib_samples.is_dir():
            _example_scan_roots.append((
                _contrib_samples,
                f"opencv_contrib/modules/{_module_dir.name}/samples",
            ))

for _root, _display_prefix in _example_scan_roots:
    if not _root.is_dir():
        continue
    for _f in _root.rglob("*"):
        if not _f.is_file() or _f.suffix.lower() not in _EXAMPLE_SOURCE_EXTENSIONS:
            continue
        # Compose the display path as <prefix>/<file's relative path under root>.
        # For core: `samples/cpp/pca.cpp`; for contrib:
        # `opencv_contrib/modules/xobjdetect/samples/peopledetect.cpp`.
        _rel = _f.relative_to(_root).as_posix()
        _display = f"{_display_prefix}/{_rel}"
        if not _is_canonical_example(_display):
            continue
        try:
            _text = _f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        _EXAMPLE_FILES.append((
            _display,
            _f,
            frozenset(_EXAMPLE_TOKEN_RE.findall(_text)),
        ))
# Stable, alphabetised order on every build.
_EXAMPLE_FILES.sort(key=lambda t: t[0])

# Display path → source path of every sample any class actually references.
# Populated by `_find_examples_for_class` during stub generation; flushed
# into one Sphinx page per entry by `_generate_example_pages` afterwards.
# Tracking only referenced files (not every scanned one) means we don't
# generate hundreds of orphan pages that no class links to.
_EXAMPLE_PAGES_NEEDED: dict[str, pathlib.Path] = {}


# ---------------------------------------------------------------------------
# Doxygen `@example` discriminator — the canonical mechanism docs.opencv.org
# actually uses to populate per-class "Examples:" footers.
#
# OpenCV / opencv_contrib headers carry `@example <path>` declarations that
# tell Doxygen "this sample file is an example program". The text *after*
# the `@example` line in the same comment block becomes the description
# Doxygen shows above the rendered code page. For instance,
# core.hpp:2592 carries:
#
#     /** @example samples/cpp/pca.cpp
#     An example using %PCA for dimensionality reduction while maintaining
#     an amount of variance
#     */
#
# That second line is exactly what docs.opencv.org shows above pca.cpp on
# its example page — and the file path is what they list in `_InputArray`'s
# Examples footer.
#
# By scanning every `*.hpp` / `*.h` under `modules/<m>/include/` (both core
# and contrib) for these declarations we reproduce Doxygen's selection
# *exactly* — no heuristics, no false positives from local helper signatures
# in unrelated samples, and we pick up the descriptions verbatim.
#
# Path resolution mirrors what Doxygen does with EXAMPLE_PATH:
#   * If `@example` carries a path starting with `samples/cpp/…`,
#     `samples/python/…` etc., it's relative to the OPENCV repo root.
#   * Otherwise (typically just `samples/<file>` from a contrib header),
#     it's relative to the containing module's `samples/` directory.
#
# We also keep the tutorial-hyperlink table as a *fallback* description
# source for `@example` declarations that don't carry their own description
# line (like peopledetect.cpp, which is declared as just `@example
# samples/peopledetect.cpp` with no following text).
# ---------------------------------------------------------------------------

# Hyperlinks whose URL points at a sample source file. The URL match is
# loose (any `samples/…/<file>.<ext>` substring) so it catches both
# repo-relative paths and absolute github URLs that happen to embed the
# repo path (e.g. `https://github.com/opencv/opencv/tree/5.x/samples/...`).
_TUTORIAL_LINK_RE = re.compile(
    r"\[([^\]]+)\]\(([^)]*?samples/[^)]+?\.(?:cpp|cc|cxx|c|py|java|js|ts))\)",
    re.IGNORECASE,
)
# Trailing prepositional phrases that introduce the hyperlink. Stripped
# from the end of the extracted description so it doesn't peter out at
# "…can be found at" / "…see" / "…in".
_TUTORIAL_LINK_TRAILER_RE = re.compile(
    r"\s*(?:"
    r"can be found at|can be found in|"
    r"are available (?:at|in)|is available (?:at|in)|"
    r"located at|located in|found at|found in|"
    r"see also|see|here|"
    r"is at|is in|at|in"
    r")\s*$",
    re.IGNORECASE,
)
# Leading Doxygen directive (`@note`, `@see`, `@sa`, `@warning`, …) that
# wraps the prose around the link. Stripped from the start of the
# description so it reads as a plain sentence.
_TUTORIAL_LEADING_DIRECTIVE_RE = re.compile(
    r"^@(?:note|see|sa|warning|attention|remark|brief|todo|deprecated)\s+",
    re.IGNORECASE,
)
# `Another example using PCA…` reads as a follow-up to *something earlier
# in the tutorial flow* — when the brief shows up in isolation on a class
# page there's no "first" example to refer back to, so the "Another" is
# stale. docs.opencv.org's older snapshot already renders the same string
# as "An example using PCA…" (their build did this rewrite, or the
# markdown was edited after their snapshot to add the "Another" lead-in).
# Replace `Another <word>` with `An <word>` / `A <word>` based on whether
# the next word starts with a vowel sound. Keeps grammar correct across
# the common follow-up phrasings ("Another approach…", "Another way…").
_ANOTHER_LEAD_RE = re.compile(r"^Another\s+(\w)", re.IGNORECASE)


def _extract_link_lead_in(text: str, link_start: int) -> str:
    """Pull the prose right before a tutorial hyperlink, cleaned up.

    Walks back from the link to the previous sentence/paragraph break,
    drops any leading `@note` / `@see` / etc. directive, collapses
    whitespace from line-wrapping, strips the trailing "can be found
    at" / "see" / "in" connector that introduced the link, and rewrites
    a stale `Another …` follow-up lead-in to a clean `An …` / `A …`
    article so the brief reads naturally in isolation.
    """
    LOOKBACK = 400
    before = text[max(0, link_start - LOOKBACK):link_start]
    # Latest sentence/paragraph boundary before the link.
    cut = max(
        before.rfind("\n\n"),
        before.rfind(". "),
        before.rfind("! "),
        before.rfind("? "),
        before.rfind(":\n"),
    )
    if cut >= 0:
        # Skip past the boundary itself (2 chars for ". ", 2 for "\n\n", …).
        before = before[cut + 2:]
    before = _TUTORIAL_LEADING_DIRECTIVE_RE.sub("", before.lstrip())
    before = re.sub(r"\s+", " ", before).strip()
    before = _TUTORIAL_LINK_TRAILER_RE.sub("", before).strip()
    # `Another example…` → `An example…` (or `A foo…` for consonant-leading
    # nouns). Vowel-letter check is a decent proxy for vowel sound; misfires
    # on edge cases like "honest" / "university" are acceptable for a brief
    # rewrite and beat the alternative of leaving a context-stale "Another".
    m = _ANOTHER_LEAD_RE.match(before)
    if m:
        article = "An " if m.group(1).lower() in "aeiou" else "A "
        before = article + before[m.start(1):]
    return before


def _scan_tutorial_sample_refs() -> dict[str, str]:
    """Walk every tutorial markdown for sample-file hyperlinks.

    Returns `{relative_sample_path: brief description}`. Samples linked
    from multiple tutorials keep the longest description (the most
    informative one). Empty if `DOC_ROOT/tutorials/` doesn't exist.
    """
    refs: dict[str, str] = {}
    tutorial_roots = [
        DOC_ROOT / "tutorials",
        DOC_ROOT / "js_tutorials",
        DOC_ROOT / "py_tutorials",
    ]
    for tut_root in tutorial_roots:
        if not tut_root.is_dir():
            continue
        for md in tut_root.rglob("*.markdown"):
            try:
                text = md.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in _TUTORIAL_LINK_RE.finditer(text):
                sample_m = re.search(
                    r"samples/.+\.(?:cpp|cc|cxx|c|py|java|js|ts)",
                    m.group(2), re.IGNORECASE,
                )
                if not sample_m:
                    continue
                sample = sample_m.group(0)
                brief = _extract_link_lead_in(text, m.start())
                if not brief:
                    continue
                if sample not in refs or len(brief) > len(refs[sample]):
                    refs[sample] = brief
    return refs


_TUTORIAL_SAMPLE_REFS: dict[str, str] = _scan_tutorial_sample_refs()


# ----------------------- @example declaration scanner -----------------------

# Matches a Doxygen `@example` (or `\example`) tag, the sample path on the
# same line, and the description lines that follow inside the same comment
# block. The capture stops as soon as it sees either:
#   * a comment-closing line `*/` (possibly indented) — end of block
#   * a new `@<directive>` / `\<directive>` line — different metadata
# Without the explicit `*/` guard the capture would run on through the
# closing line and into the next `/** ... */` block, picking up unrelated
# `@brief` text from the symbol the example was attached to.
_DOXY_EXAMPLE_RE = re.compile(
    r"[@\\]example\s+(\S+)"              # group 1: declared path
    r"[^\n]*\n"                          # rest of @example line
    r"(?P<desc>"
    r"(?:"
    r"(?!\s*\*/)"                        # not the comment closer
    r"(?!\s*\*?\s*[@\\]\w+)"             # not a new directive
    r"[^\n]*\n"                          # consume one line
    r")*"
    r")"
)

# Doxygen `%X` escape — used to keep `%PCA` from being treated as a
# Doxygen reference. We unescape so descriptions read naturally.
_DOXY_PERCENT_ESCAPE_RE = re.compile(r"%(\w+)")


def _resolve_example_path(decl_path: str, header_path: pathlib.Path) -> str | None:
    """Resolve a `@example <path>` declaration to our repo-relative display
    path, matching Doxygen's EXAMPLE_PATH resolution.

    Two cases:
      1. Path is repo-relative from the OpenCV root (e.g. `samples/cpp/pca.cpp`)
         — used by core headers. Resolve against OPENCV_ROOT.
      2. Path is module-relative (e.g. `samples/peopledetect.cpp` from
         xobjdetect.hpp) — resolve against the containing module's
         directory and render as `opencv_contrib/modules/<m>/<decl_path>`.

    Returns None when neither candidate exists on disk.
    """
    # Case 1: repo-relative.
    abs_main = OPENCV_ROOT / decl_path
    if abs_main.is_file():
        return abs_main.relative_to(OPENCV_ROOT).as_posix()

    # Case 2: module-relative. Walk up from the header to find the module
    # root (the parent of `include/`).
    module_root: pathlib.Path | None = None
    for p in header_path.parents:
        if p.name == "include":
            module_root = p.parent
            break
    if module_root is None:
        return None

    abs_mod = module_root / decl_path
    if not abs_mod.is_file():
        return None

    # Render with the same prefix convention the scanner uses. Use
    # `is_relative_to` (not `startswith`) because OPENCV_ROOT (`.../opencv`)
    # is a string-prefix of `.../opencv_contrib` and string-startswith
    # gives a false-positive on contrib paths.
    if abs_mod.is_relative_to(OPENCV_ROOT):
        return abs_mod.relative_to(OPENCV_ROOT).as_posix()
    # Contrib path. CONTRIB_ROOT is `<...>/opencv_contrib/modules/`, so its
    # parent's parent is the repo's parent dir — sample paths under it
    # render as `opencv_contrib/modules/<m>/samples/...`.
    contrib_parent = CONTRIB_ROOT.parent.parent
    if abs_mod.is_relative_to(contrib_parent):
        return abs_mod.relative_to(contrib_parent).as_posix()
    # Last-ditch fallback — synthesize the conventional display string.
    return f"opencv_contrib/modules/{module_root.name}/{decl_path}"


def _scan_doxygen_example_decls() -> dict[str, tuple[str, str]]:
    """Walk every module header for `@example` declarations.

    Returns `{resolved repo-relative path: (declared path, description)}`:
      * resolved path — the on-disk display path the file scanner uses
        (`samples/cpp/pca.cpp` for core, `opencv_contrib/modules/<m>/
        samples/<f>` for contrib). Used as the dictionary key so it
        joins cleanly with `_EXAMPLE_FILES`, which is keyed the same way.
      * declared path — the literal string from the `@example` tag
        (`samples/peopledetect.cpp` instead of the long contrib resolved
        form). That's what docs.opencv.org displays on the class page,
        so we surface it in the rendered Examples link too.
      * description — the line(s) immediately following `@example` inside
        the same comment block, with Doxygen `%X` escapes unwrapped,
        whitespace collapsed, and any trailing `*/` stripped.

    An empty description means the declaration exists but didn't carry
    one — the renderer falls back to the tutorial-hyperlink table and
    then to the file's own top-comment extraction.
    """
    refs: dict[str, tuple[str, str]] = {}
    roots = [OPENCV_ROOT / "modules"]
    if CONTRIB_ROOT.is_dir():
        roots.append(CONTRIB_ROOT)

    for root in roots:
        if not root.is_dir():
            continue
        for ext in ("*.hpp", "*.h"):
            for header in root.rglob(ext):
                # Only look inside `<module>/include/` — keeps the scan
                # fast and avoids false hits from test/sample sources.
                if "/include/" not in header.as_posix():
                    continue
                try:
                    text = header.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for m in _DOXY_EXAMPLE_RE.finditer(text):
                    declared = m.group(1).strip().rstrip("*/").strip()
                    resolved = _resolve_example_path(declared, header)
                    if not resolved:
                        continue
                    # Description cleanup: strip leading `*` markers from
                    # each continuation line, collapse whitespace, drop
                    # trailing `*/` if any leaked in.
                    raw = m.group("desc") or ""
                    cleaned = re.sub(r"\n\s*\*\s?", " ", raw)
                    cleaned = cleaned.replace("*/", "").strip()
                    cleaned = re.sub(r"\s+", " ", cleaned)
                    cleaned = _DOXY_PERCENT_ESCAPE_RE.sub(r"\1", cleaned)
                    # Keep the longest description if the sample is
                    # declared in multiple headers (rare but possible).
                    existing = refs.get(resolved)
                    if existing is None or len(cleaned) > len(existing[1]):
                        refs[resolved] = (declared, cleaned)
    return refs


_DOXY_EXAMPLE_DECLS: dict[str, tuple[str, str]] = _scan_doxygen_example_decls()


def _find_examples_for_class(class_simple: str) -> list[tuple[str, str]]:
    """Canonical sample files mentioning the class name.

    Returns `(display path, page basename)` per match — the basename is the
    Sphinx page we'll generate under `examples/`, so the renderer can build
    a relative link to it.

    Two-stage filter (matches Doxygen's behaviour exactly):
      1. Sample must be declared with `@example <path>` in some module
         header. That's the canonical mechanism docs.opencv.org's build
         uses to populate per-class Examples footers — only files the
         OpenCV maintainers explicitly mark as example programs qualify,
         which is why aruco_dict_utils.cpp / essential_mat_reconstr.cpp
         (used to be in the match list) drop out: they're sample
         programs but nobody declared them with `@example`.
      2. The sample's source must mention the class's name. Underscore-
         prefixed implementation classes (`_InputArray`, `_OutputArray`)
         also try the unprefixed alias since that's what samples write.
    """
    if not class_simple:
        return []
    candidates = {class_simple}
    if class_simple.startswith("_") and len(class_simple) > 1:
        candidates.add(class_simple[1:])
    out: list[tuple[str, str]] = []
    for display, source_path, tokens in _EXAMPLE_FILES:
        # Stage 1: must be declared with `@example` somewhere.
        decl = _DOXY_EXAMPLE_DECLS.get(display)
        if decl is None:
            continue
        # Stage 2: must mention the class.
        if any(c in tokens for c in candidates):
            _EXAMPLE_PAGES_NEEDED[display] = source_path
            # Surface the *declared* path string (e.g. `samples/peopledetect.cpp`)
            # rather than the resolved one (e.g. `opencv_contrib/modules/
            # xobjdetect/samples/peopledetect.cpp`) — that's what docs.opencv.org
            # shows on the class page, and what the user expects to read.
            declared, _desc = decl
            out.append((declared, _example_pagename(display)))
    return out


def _render_examples_block(examples: list[tuple[str, str]]) -> list[str]:
    """HTML lines for the "Examples" footer; empty list if no matches.

    Mirrors docs.opencv.org's `<dl class="section examples">…</dl>`: a bold
    "Examples" label, then plain blue links (NO inline-code chip) joined
    with Oxford commas and a trailing period. Each link points at a locally
    generated `examples/<page>.html` page that renders the sample's source
    code — same destination semantic the reference site has, just produced
    by us natively instead of by Doxygen's EXAMPLE_PATH machinery.
    """
    if not examples:
        return []
    import html as _html_pkg
    parts = [
        # `../examples/<page>.html` is the link target relative to the api/
        # class page that hosts this block. Plain text inside <a>, no <code>
        # wrapper — chip styling is a docs.opencv.org-divergent look the user
        # rejected; their list is plain blue underlined links.
        f'<a class="opencv-example-link" '
        f'href="../examples/{_html_pkg.escape(page, quote=True)}.html">'
        f'{_html_pkg.escape(display)}</a>'
        for display, page in examples
    ]
    if len(parts) == 1:
        joined = parts[0]
    else:
        # "X, and Y" / "X, Y, and Z" — Oxford-comma form, same as the
        # reference site's rendering.
        joined = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return [
        '<dl class="opencv-examples">',
        '<dt>Examples</dt>',
        f'<dd>{joined}.</dd>',
        '</dl>',
        "",
    ]


# Metadata-paragraph filter used by `_extract_sample_brief` below. A
# paragraph is treated as boilerplate (and skipped) when its first line
# starts with one of these keywords as a whole word — *anything* may
# follow on the same line. The match is word-boundary anchored rather
# than strict-colon: pca.cpp uses `Special Thanks to:` (the colon doesn't
# sit immediately after the keyword), and similar phrasing shows up under
# Author / Copyright across the sample tree.
#
# A separate alternative catches the bare-filename title line some
# authors put at the very top of the comment (e.g. just `pca.cpp` on its
# own line — a non-Doxygen "title").
_SAMPLE_BRIEF_SKIP_RE = re.compile(
    r"^(?:"
    r"author|date|file|copyright|special\s+thanks|see\s+also|maintainer|"
    r"created|modified|version|license|brief"
    r")\b"
    r"|^\w+\.(?:cpp|cc|cxx|c|hpp|h|py|java|js|ts)\s*$",
    re.IGNORECASE,
)


def _extract_sample_brief(text: str) -> str:
    """Best-effort one-line description of a sample file.

    Priority — first hit wins:
      1. An explicit `@brief` / `\\brief` Doxygen tag anywhere in the file,
         text running until the next blank line, the next Doxygen tag, or
         the end of the comment.
      2. The first prose paragraph of the file's leading `/* ... */`
         comment block, after dropping boilerplate paragraphs (Author:,
         Date:, File:, Copyright, Special Thanks, the filename itself).
         Returns just the first sentence of that paragraph.

    Returns an empty string when neither yields anything usable — most
    samples (45 of 47 in `samples/cpp/`) have no description anywhere and
    the page is rendered without one rather than with a fabricated guess.
    """
    # 1) Explicit @brief / \brief — preferred when present.
    m = re.search(
        r"[@\\]brief\s+(.*?)(?:"
        r"\n\s*[*/]?\s*\n"          # paragraph break (blank line inside comment)
        r"|\n\s*[*/]?\s*[@\\]\w+"   # next Doxygen tag
        r"|\*/"                     # end of comment block
        r")",
        text, re.DOTALL,
    )
    if m:
        # Collapse continuation lines into spaces. Some comments use a
        # `*` continuation marker (`* foo\n* bar`), some don't (`  foo\n
        # bar`) — `\*?` makes the asterisk optional so both shapes
        # normalise the same way. Final `\s+ -> " "` removes any extra
        # whitespace left after the collapse.
        brief = re.sub(r"\n\s*\*?\s*", " ", m.group(1))
        brief = re.sub(r"\s+", " ", brief).strip()
        brief = re.split(r"(?<=[.!?])\s", brief, maxsplit=1)[0].strip()
        if brief:
            return brief

    # 2) First /* ... */ block, split into paragraphs on blank `*` lines.
    cm = re.match(r"\s*/\*+([\s\S]*?)\*+/", text)
    if not cm:
        return ""
    # Normalise: drop the leading `* ` (or `*`) from each line, then
    # `.strip()` — using `.rstrip()` would leave a leading space on lines
    # written as `*  Author:` (`*` + two spaces), which then bypasses the
    # boilerplate-keyword skip regex anchored at start of string.
    norm_lines = [
        re.sub(r"^\s*\*+\s?", "", raw).strip()
        for raw in cm.group(1).splitlines()
    ]
    # Group into paragraphs (blank line = paragraph break).
    paragraphs: list[list[str]] = [[]]
    for line in norm_lines:
        if line.strip():
            paragraphs[-1].append(line)
        elif paragraphs[-1]:
            paragraphs.append([])
    paragraphs = [p for p in paragraphs if p]

    for para in paragraphs:
        if _SAMPLE_BRIEF_SKIP_RE.match(para[0]):
            continue
        joined = " ".join(line.strip() for line in para if line.strip())
        # First sentence — split on `.`/`!`/`?` followed by whitespace.
        first = re.split(r"(?<=[.!?])\s", joined, maxsplit=1)[0].strip()
        if first:
            return first
    return ""


def _generate_example_pages(examples_dir: pathlib.Path) -> None:
    """Write one Sphinx page per sample referenced by an Examples block.

    Reproduces what Doxygen's EXAMPLE_PATH produces on docs.opencv.org: a
    standalone page per example file that renders the source with syntax
    highlighting, preceded by a brief description pulled from the file's
    own top comment / `@brief` tag (when the file has one). The page is
    marked `orphan: true` so Sphinx doesn't warn about it sitting outside
    the toctree — its only inbound links come from the raw-HTML Examples
    blocks we inject in `_render_examples_block`.

    Page body uses MyST's colon-fence `:::` form for the code block rather
    than triple-backtick: backticks inside the source (Python f-strings,
    Markdown comments in JS samples, …) can otherwise close the fence
    prematurely. `:::` collisions in a code file are vanishingly rare.
    """
    if not _EXAMPLE_PAGES_NEEDED:
        return
    examples_dir.mkdir(parents=True, exist_ok=True)
    for display, source in _EXAMPLE_PAGES_NEEDED.items():
        try:
            body = source.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        language = _EXAMPLE_LANGUAGE.get(source.suffix.lower(), "text")
        # Priority cascade for the brief:
        #   1. The description line of the `@example` Doxygen declaration
        #      that marked this sample (matches docs.opencv.org verbatim —
        #      both builds read from the same source).
        #   2. Prose around a tutorial markdown hyperlink to this sample
        #      (useful when `@example` carries the path but no description,
        #      e.g. peopledetect.cpp's `/**@example samples/peopledetect.cpp\n*/`).
        #   3. Best-effort from the file's own top comment / `@brief`.
        # Any may be empty; if all are, the page renders without a brief.
        decl = _DOXY_EXAMPLE_DECLS.get(display)
        declared_path = decl[0] if decl else display
        decl_desc = decl[1] if decl else ""
        brief = (
            decl_desc
            or _TUTORIAL_SAMPLE_REFS.get(display)
            or _extract_sample_brief(body)
        )
        lines = [
            "---",
            "orphan: true",
            "---",
            # Title uses the declared path string so it matches the link
            # text shown on the class page that brought the user here
            # (e.g. `samples/peopledetect.cpp` rather than its long
            # resolved contrib form).
            f"# {declared_path}",
            "",
        ]
        if brief:
            # Plain paragraph above the code listing — same spot
            # docs.opencv.org puts its description. Wrapped in a small
            # marker class so CSS can style it distinctly from arbitrary
            # body text if we ever want to.
            lines.append(f'<p class="opencv-example-brief">{brief}</p>')
            lines.append("")
        lines.extend([
            f":::{{code-block}} {language}",
            ":linenos:",
            "",
        ])
        lines.extend(body.splitlines())
        lines.append(":::")
        lines.append("")
        (examples_dir / f"{_example_pagename(display)}.md").write_text(
            "\n".join(lines), encoding="utf-8")



__all__ = [
    "_find_examples_for_class",
    "_render_examples_block",
    "_generate_example_pages",
]
