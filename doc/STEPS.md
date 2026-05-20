# Modernizing OpenCV 5.x Documentation — `core` Module Migration Steps

> Reference PR: [Prasadayus/opencv#27](https://github.com/Prasadayus/opencv/pull/27) — branch `Doc_update_pip` → `5.x`.
> Status: the **core** module has been fully ported; `imgproc` is partly ported (basic tutorials only). This file captures everything that was done so the same recipe can be applied module-by-module to the rest of the tree.

## ⚠ Hard rule — DO NOT change the content, only the format

This applies to every module migration without exception:

- **DO NOT rewrite, paraphrase, summarise, condense, shorten, expand, clarify, "improve", reorder, "modernise", spell-check, or otherwise edit the prose.** Every paragraph, sentence, word, and punctuation mark is reproduced verbatim from the legacy `.markdown`.
- **DO NOT touch the code samples.** Every `@code … @endcode` block and every C++ / Python / Java / JavaScript snippet (whether inline or via `@snippet`) is moved across byte-for-byte. No reformatting, no renaming variables, no inserting comments, no swapping `auto` for explicit types, no fixing pre-existing typos.
- **DO NOT re-encode, crop, resize, recolour, rename, optimise, or substitute the images.** Copy them with `cp` (or `git mv` only when the destination is the new shared `images/` folder). Verify with `cmp` or `md5sum` if in doubt.
- **DO NOT touch the legacy `doc/tutorials/<mod>/` tree** — old and new pipelines coexist, and the old `.markdown` files remain the source of truth until the new docs are accepted upstream.
- **DO NOT translate `@ref` links to different anchors** — the converter resolves them through `opencv.tag`; if a link looks wrong, treat the tag-file or `local_refs.json` as the bug site, never edit the link by hand.
- **DO NOT skip a tutorial** because it "looks redundant" or "the API is deprecated". Port every page that exists in `doc/tutorials/<mod>/`.

What you ARE allowed to change is **only the surrounding syntax** (the Doxygen-flavoured markup that the converter rewrites — see the mapping table below) and the **directory layout** (flatten per-tutorial subdirs into sibling `.md` files, merge per-tutorial `images/` into one shared module-level folder). Anything beyond that is out of scope.

If a converted page renders incorrectly, fix it by changing the **converter** ([_tools/dox2myst.py](_tools/dox2myst.py)) or the **CSS** ([_static/custom.css](_static/custom.css)) — never by editing the migrated `.md` to work around it. A workaround in one file makes the next module diverge.

## Context

The OpenCV 5.x documentation has historically been emitted by **Doxygen + doxygen-awesome.css** from a tree of `.markdown` tutorials and inline source-code comments. The visual style is dated, theme switching is limited, search is coarse, and authors must learn Doxygen-flavoured markdown (`@code`, `@ref`, `@snippet`, `@add_toggle_*`, `@tableofcontents`, …).

This PR introduces a **parallel, opt-in Sphinx-based pipeline** (`docs/`) that:

- Keeps every existing text block, code sample, and image **byte-for-byte identical** — only the surrounding syntax (and the build chain) changes.
- Renders through **PyData Sphinx Theme + MyST Parser**, with a **custom CSS layer** that gives the site a modern look (rounded tables, code-copy buttons, dark/light toggle, sticky blurred header, gradient back-to-top pill, etc.).
- Keeps Doxygen in the loop as the **C++ API source-of-truth** by enabling its XML output and feeding it to **Breathe + Exhale** under Sphinx.
- Does **not** touch the legacy `doc/` tree — both pipelines coexist behind separate CMake switches (`BUILD_DOCS` vs `BUILD_DOCS_MODERN`) so reviewers can A/B compare.

## What was actually added/changed

### A. New top-level tree: `docs/`

```
docs/
├── conf.py                  Sphinx config (PyData theme, MyST, Breathe/Exhale, custom exts)
├── CMakeLists.txt           opencv_docs + opencv_docs_serve targets
├── index.md                 Module landing page (replaces old root.markdown.in)
├── introduction.md, getting_started.md, faq.md, api_reference.md
├── _ext/                    Custom Sphinx extensions
│   ├── doxysnippet.py       Re-implements Doxygen `@snippet` as a Sphinx directive
│   ├── opencv_code_links.py Builds symbol→URL map from opencv.tag → _static/opencv-symbols.json
│   └── tabs.py              Minimal `div` / `tab-set` / `tab-item` directives (no sphinx-design dep)
├── _tools/
│   ├── dox2myst.py          One-shot Doxygen-markdown → MyST converter (1339 LOC)
│   └── local_refs.json      Map of doxygen anchors → local sphinx docnames
├── _templates/              PyData sidebar/header overrides (version-badge.html, external-nav.html)
├── _static/                 CSS + JS + logos + favicon
│   ├── custom.css           ★ The entire visual theme lives here (~970 LOC)
│   ├── copybutton.js        Code-block copy button + breadcrumbs
│   ├── theme-toggle.js      Light/dark switch button (writes localStorage)
│   └── opencv-code-links.js Loads opencv-symbols.json, wraps tokens inside <code> in links
├── api/                     Generated Breathe/Exhale stubs (auto-populated at build)
└── tutorials/{cpp,python,javascript}/<module>/  Ported tutorial pages
```

### B. Edits to existing files

- [doc/CMakeLists.txt](../doc/CMakeLists.txt) — when `BUILD_DOCS_MODERN=ON`, sets `OPENCV_DOXYGEN_GENERATE_XML=YES` so Breathe/Exhale can consume Doxygen's XML. Also: includes `apps/` in `EXAMPLE_PATH`; handles module names that start with a digit (e.g. `3d` → quoted `_3d "3d"` ref).
- [doc/Doxyfile.in](../doc/Doxyfile.in) — `GENERATE_XML = @OPENCV_DOXYGEN_GENERATE_XML@`; legacy doxygen-awesome stylesheet wiring kept.

## Step-by-step recipe for migrating one module

Apply this to each remaining module (`imgproc`, `calib3d`, `3d`, `dnn`, `features`, `objdetect`, `photo`, `gpu`, `others`, `app`, `introduction`, `ios`).

### Step 1 — Mirror the directory structure

For each module under `doc/tutorials/<mod>/`, create `docs/tutorials/cpp/<mod>/` (and python/javascript variants where applicable). The old layout has **one subdirectory per tutorial**, each containing the `.markdown` + an `images/` folder:

```
doc/tutorials/core/mat_the_basic_image_container/
  ├── mat_the_basic_image_container.markdown
  └── images/MatBasicContainerOut1.png, …
```

The new layout **flattens tutorials into sibling `.md` files** and **collapses every tutorial's `images/` into one shared folder** at the module level:

```
docs/tutorials/cpp/core/
  ├── mat_the_basic_image_container.md
  ├── how_to_scan_images.md
  ├── …
  ├── index.md                     ← new module index (replaces table_of_content_*.markdown)
  └── images/                      ← all PNG/JPG/GIF from every tutorial merged here
```

> **Images stay byte-identical** — they are copied, not re-encoded. Verify with `cmp` or by counting files.

### Step 2 — Convert each `.markdown` to MyST `.md`

Use the one-shot converter, which already handles every Doxygen construct in the OpenCV tree:

```bash
python3 docs/_tools/dox2myst.py \
    doc/tutorials/<mod>/<topic>/<topic>.markdown \
    docs/tutorials/cpp/<mod>/<topic>.md \
    --tag /path/to/opencv.tag \
    --local docs/_tools/local_refs.json \
    --out-doc tutorials/cpp/<mod>/<topic>
```

This performs the following rewrites (one-to-one, **text/code/image references untouched**):

| Doxygen (.markdown) | MyST (.md) | Why |
|---|---|---|
| `Goal\n----` (setext heading) | `## Goal` (ATX heading) | MyST prefers ATX; consistent with Sphinx anchors |
| `Title {#tutorial_anchor}` | `# Title` | Sphinx auto-anchors from heading text |
| `@tableofcontents`, `@prev_tutorial{…}`, `@next_tutorial{…}` | *(deleted)* | PyData theme renders prev/next + a right-side TOC from `toctree` |
| `@subpage tutorial_xxx` | `[Title](xxx.md)` | Becomes a normal Markdown link (resolved via `local_refs.json`) |
| `@code{.cpp} … @endcode` | <code>```cpp … ```</code> | Standard fenced code blocks (also work in editor previews) |
| `@snippet path tag` | <code>```{doxysnippet} path<br>:tag: tag<br>:language: cpp<br>```</code> | Custom Sphinx directive in `_ext/doxysnippet.py` re-implements snippet extraction (looks for `//! [tag]` markers) |
| `@ref cv::Mat::clone()` | `[cv::Mat::clone](https://docs.opencv.org/5.x/d3/d63/classcv_1_1Mat.html#a03…)()` | Anchor resolved through `opencv.tag`; falls back to internal Sphinx ref if the symbol is in `local_refs.json` |
| `@note …` / `@warning …` / `@attention …` / `@important …` / `@tip …` | `:::{note} … :::` / `:::{warning} … :::` / `:::{attention} … :::` / `:::{important} … :::` / `:::{tip} … :::` | MyST admonitions (each renders as a coloured callout) |
| `@add_toggle_Cpp …`<br>`@add_toggle_Python …`<br>`@end_toggle` | `::::{tab-set}` / `:::{tab-item}` blocks via `_ext/tabs.py` | Synced multi-language tabs (e.g. C++/Python/Java on one page) |
| `@youtube{ID}` | <code>```{raw} html<br>&lt;iframe …youtube-nocookie.com/embed/ID…&gt;<br>```</code> | Responsive 16:9 wrapper around the iframe |
| `@brief`, `@details`, `@param`, `@return`, `@sa`, `@see`, `@cite`, `@throws`, `@since`, `@todo`, `@cond` / `@endcond` | rewritten or stripped inline | These mostly appear in API doc comments rather than tutorials, but the converter handles them defensively |
| Doxygen numbered-step lists (`-# …` blocks) | regular `1.`, `2.`, `3.` markdown ordered lists | Doxygen's `-#` auto-numbering syntax doesn't exist in MyST |
| Bare bullet list of `- @subpage …` | `{list-table}` with `:class: opencv-module-table` | Used by old `table_of_content_*.markdown` files to list tutorials — converted to the rounded card table |
| Front-matter metadata table | Same table wrapped in <code>:::{div} opencv-meta-table … :::</code> | Hooks the **curved-edge, grey-first-column** styling in `custom.css` |
| Image lines `![](images/foo.png)` | `{figure}` directive (only when alone on a line) | Allows MyST to centre + caption the image; raw `![]()` remains where it's inline |
| Multi-row HTML tables with `rowspan=` | Wrapped in `:::{div} opencv-rowspan-table` | Preserves the layout while giving CSS a hook |

**Auto-linking inside prose and inline code** — the converter also scans for OpenCV identifiers and turns them into links to `docs.opencv.org/5.x/...`:

- `cv::Foo`, `cv::Foo::bar` → linked
- `CV_FOO` macros → linked
- `#ClassName` Doxygen back-reference → linked
- Prefix any of the above with `%` to opt out (e.g. `%cv::Mat`) — the `%` is consumed and no link is generated. This is the same opt-out syntax Doxygen uses.

The text body, every code sample, and every `![](images/foo.png)` reference are passed through verbatim.

### Step 3 — Build the module landing page

Replace `doc/tutorials/<mod>/table_of_content_<mod>.markdown` with `docs/tutorials/cpp/<mod>/index.md`. Two visible sections plus a hidden `toctree` for the sidebar:

````markdown
# The Core Functionality (core module)

## Basic

​```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description
* - [Mat - The Basic Image Container](mat_the_basic_image_container.md)
  - How OpenCV stores and handles images with `cv::Mat` — automatic memory management, …
…
​```

​```{toctree}
:hidden:
:maxdepth: 1

mat_the_basic_image_container
how_to_scan_images
…
​```
````

The `opencv-module-table` class is what gives this index its modern card-like row appearance (see styling notes below).

### Step 4 — Wire the module into the global TOC

Add the module's `index` under the top-level `{toctree}` block in [docs/index.md](index.md) (or `tutorials/cpp/index.md` if that's the layer being touched).

### Step 5 — Build & verify

```bash
cmake -B build -DBUILD_DOCS_MODERN=ON …
cmake --build build --target opencv_docs        # one-shot build
cmake --build build --target opencv_docs_serve  # live-reload on :8000
```

Sanity checks per migrated tutorial:
1. Page renders without Sphinx warnings (other than the suppressed `myst.header`).
2. Every code snippet shows the exact bytes from the C++ source (compare against rendered legacy page).
3. Every image loads from the shared `images/` folder.
4. Every `@ref`-derived link still resolves to `docs.opencv.org/5.x/...`.
5. Light/dark mode both look right (the custom CSS has explicit dark-mode overrides everywhere).
6. Code-copy button appears on hover; copy actually copies the snippet text.

## Key visual takeaways — what the new theme looks like

All visual style lives in [docs/_static/custom.css](_static/custom.css). The notable design decisions:

### 1. Tables have **curved edges**

Three table classes, each with `border-radius: 0.5rem` + `overflow: hidden` to clip the cells back inside the rounded outer border:

- **`opencv-meta-table`** — the tutorial front-matter ("Original author / Compatibility") two-column table.
  - First column gets a darker background (`rgba(0,0,0,0.25)` light / `#2d333b` dark) and bold weight to act as a label column.
  - `thead` is hidden (the empty `|--|--|` row from the markdown disappears).
  - Last row strips its bottom border, last column strips its right border — so the outer rounded border seals cleanly.
- **`opencv-module-table`** — the module landing page table ("Topic / Description").
  - Header row in uppercase, 0.09em letter-spacing, muted color — feels like a section header rather than a data row.
  - Tbody rows have only a thin bottom-divider (no vertical lines) → reads like a list of cards.
  - Row hover paints the row with `--pst-color-surface` for a subtle highlight.
- **`.bd-content table`** (catch-all) — full grid lines with the theme border variable, padded cells (0.65rem × 1rem), uppercase header text.

### 2. Code blocks

- Light-mode background `#f6f8fa`, **3px left accent border `#0550ae`**, `0.4rem` corner radius.
- Dark-mode background `#2d333b`, accent `#539bf5`, foreground `#cdd9e5`, full Pygments token re-mapping (comments `#e3b341`, keywords `#f47067`, strings `#96d0ff`, numbers `#6cb6ff`, …) so colours remain legible after the dark switch.
- **Copy button** (top-right) appears only on `div.highlight:hover`, flips to a green checkmark for 1s on success — defined in `copybutton.js`.

### 3. Light/dark theme

- `theme-toggle.js` injects a sun/moon button into the navbar and writes `data-theme` on `<html>`; honors `prefers-color-scheme` on first load.
- Two CSS root blocks (`html[data-theme="light"]` / `html[data-theme="dark"]`) redefine the accent and link colours; everywhere else uses PyData's CSS variables (`--pst-color-border`, `--pst-color-surface`, …) so the theme is consistent.

### 4. Header & navigation

- Sticky blurred header (`backdrop-filter: blur(8px)`), 3.25rem min-height.
- Nav items rendered uppercase, 0.78rem, 0.06em letter-spacing — feels like a section label, not a link list.
- A small **version-badge** (`5.x`) is rendered next to the logo via [_templates/version-badge.html](_templates/version-badge.html).
- External links (Main Page, Namespaces, Classes, Files, Java doc, …) routed through `_templates/external-nav.html`.

### 5. API reference pages (Breathe / Exhale output)

- Function/class signatures wrap in a `surface`-coloured card with a `3px` left accent border (`--opencv-accent`).
- The redundant `## Function Documentation` header that Exhale emits is hidden via `section#function-documentation > h2 { display: none; }`.
- Parameter `dl.field-list` becomes a bordered "table": uppercase label header bar over a list of rows each with a bottom divider.
- Each Exhale page's `#include` line (the first `ul.simple` after `<h1>`) is recast into a monospace inline-block badge.

### 6. Typography & fonts

Two web fonts are loaded from Google Fonts (declared in [conf.py](conf.py) under `html_css_files`):

```
https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap
```

- **Inter** — body + headings. Weights `400, 500, 600, 700`. CSS variable: `--pst-font-family-base` / `--pst-font-family-heading`.
- **JetBrains Mono** — code/monospace. Weights `400, 500`. CSS variable: `--pst-font-family-monospace`.
- Fallback stack for the base font: `"Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`.
- Fallback stack for monospace: `"SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace`.
- **Font Awesome 6 Brands** — pulled in transitively by the PyData theme; used by `custom.css` to render the GitHub octocat (`\f09b`) prefix on external GitHub links.
- **Base size**: `--pst-font-size-base: 16px` (also forced on `html`).
- **Heading scale**: `h1` = 2.25rem / 700 / line-height 1.2 / letter-spacing `-0.02em`; `h2` = 1.5rem / 600 / `-0.01em`; `h3` = 1.125rem / 600. Body line-height `1.6`.

### 7. Other niceties

- **Heading anchors** invisible until you hover the heading (`a.headerlink { opacity: 0 }` → 1 on hover).
- **Active-section highlight** in the right-side TOC (`#bd-toc-nav a.ocv-toc-active`).
- **Back-to-top pill** — fixed-bottom, centred, blue gradient (`linear-gradient(135deg, #0066cc, #003a6b)`), 2rem radius — uses PyData's built-in `#pst-back-to-top` element.
- **GitHub octocat** prefix is auto-added before any `a.github.reference.external` link via a Font-Awesome `::before`.
- **Wide reading column** — `.bd-page-width { max-width: none }` lifts PyData's narrow default, padding handled by `.bd-main`.

## Critical files to edit / consult

| Purpose | File |
|---|---|
| Sphinx config (theme, extensions, exhale args) | [docs/conf.py](conf.py) |
| CMake target wiring | [docs/CMakeLists.txt](CMakeLists.txt), [doc/CMakeLists.txt](../doc/CMakeLists.txt) |
| Doxygen XML opt-in for Breathe | [doc/Doxyfile.in](../doc/Doxyfile.in) |
| Visual theme | [docs/_static/custom.css](_static/custom.css) |
| Per-file conversion logic (the workhorse) | [docs/_tools/dox2myst.py](_tools/dox2myst.py) |
| `@snippet` replacement | [docs/_ext/doxysnippet.py](_ext/doxysnippet.py) |
| `@add_toggle_*` replacement | [docs/_ext/tabs.py](_ext/tabs.py) |
| Auto-linking C++ identifiers in code blocks | [docs/_ext/opencv_code_links.py](_ext/opencv_code_links.py), [docs/_static/opencv-code-links.js](_static/opencv-code-links.js) |
| Reference example (already migrated) | [docs/tutorials/cpp/core/mat_the_basic_image_container.md](tutorials/cpp/core/mat_the_basic_image_container.md) vs the original [doc/tutorials/core/mat_the_basic_image_container/mat_the_basic_image_container.markdown](../doc/tutorials/core/mat_the_basic_image_container/mat_the_basic_image_container.markdown) |

## Module migration checklist

For each module the task is mechanical and repetitive — what's been finished so far is the template:

- [x] `core` (all 10 tutorials + index)
- [~] `imgproc` (basic tutorials done — `basic_geometric_drawing`, `erosion_dilatation`, `filter_2d`, `gausian_median_blur_bilateral_filter`, `hitOrMiss`, `morph_lines_detection`, `opening_closing_hats`, `pyramids`)
- [ ] `imgproc` (remaining advanced tutorials)
- [ ] `3d`, `calib3d`
- [ ] `dnn`
- [ ] `features`
- [ ] `objdetect`
- [ ] `photo`
- [ ] `gpu`, `ios`
- [ ] `app`, `others`, `introduction`
- [ ] `py_tutorials/`, `js_tutorials/` (Python / JS counterparts)

## Verification

End-to-end check after migrating any module:

1. `pip install pydata-sphinx-theme myst-parser breathe exhale sphinx-autobuild`
2. `cmake -B build -DBUILD_DOCS=ON -DBUILD_DOCS_MODERN=ON …`
3. `cmake --build build --target doxygen opencv_docs`
4. Open `build/doc/modern_html/index.html` — confirm the module appears in the sidebar/TOC, every tutorial page renders, every image loads, every external `docs.opencv.org/5.x/…` link resolves, code-copy + theme toggle work.
5. `cmake --build build --target opencv_docs_serve` for interactive iteration.

## Notes & gotchas

These are the non-obvious things that bit during the `core` port — record them here so the next module doesn't repeat them:

- **Two pipelines must build together.** `BUILD_DOCS_MODERN=ON` does NOT imply `BUILD_DOCS=ON`. Sphinx still needs Doxygen's XML (for Breathe/Exhale to ingest the C++ API). Always pass both: `-DBUILD_DOCS=ON -DBUILD_DOCS_MODERN=ON`.
- **Exhale 0.3.x bug.** Exhale skips `os.makedirs` before writing some RST files on the first run → build fails with "No such file or directory". The monkey-patch in [conf.py](conf.py)'s `setup(app)` (`_eg.ExhaleRoot.generateSingleNodeRST`) fixes this — do not remove it until Exhale releases a fix.
- **Breathe 4.36.x bug.** Doxygen emits some members with `kind="property"`, which Breathe's `cpp_classes` table doesn't know about. The `setup()` in `conf.py` also patches `DomainDirectiveFactory.cpp_classes["property"] = (CPPMemberObject, "var")` to fall back to `var`. Same caveat.
- **opencv.tag is downloaded at configure time** ([CMakeLists.txt](CMakeLists.txt) `file(DOWNLOAD …)`). The file is cached at `${CMAKE_BINARY_DIR}/doc/opencv.tag`. Delete it to force a refresh after a new OpenCV release.
- **`@snippet` markers must be balanced.** The directive (see [_ext/doxysnippet.py](_ext/doxysnippet.py)) needs two `//! [tag]` lines (start + end) — if there's only one, you get a Sphinx warning and an empty block. Re-check after any C++ sample edit.
- **Module names that start with a digit** (e.g. `3d`) break `@ref` because Doxygen IDs can't begin with a digit. [doc/CMakeLists.txt](../doc/CMakeLists.txt) was patched to emit `@ref _3d "3d"` for these. Watch out when adding more digit-prefixed modules.
- **`local_refs.json` is intentionally hand-curated.** Symbols listed there resolve to internal Sphinx docs; symbols not listed fall back to external `docs.opencv.org/5.x/...` links via opencv.tag. Add an entry for any tutorial whose Doxygen anchor (e.g. `tutorial_mat_the_basic_image_container`) should now resolve internally.
- **`thead` of `opencv-meta-table` is hidden.** The front-matter table's empty `|---|---|` header row would render as an empty top stripe otherwise. If you ever add a real header, override `thead { display: revert }` for that table only.
- **`{raw} html` blocks bypass MyST sanitization.** Used for the YouTube iframe — fine, but don't paste arbitrary HTML; prefer admonitions/directives.
- **Image paths are relative to the `.md` file**, but all images for the module sit in the shared `images/` directory next to the `.md` (not inside per-tutorial subdirs as in the legacy layout). Don't introduce `../images/...` paths.
- **Suppressed warning class.** `conf.py` sets `suppress_warnings = ["myst.header"]` because MyST otherwise complains about non-sequential heading levels in some converted tutorials. Don't add more without good cause.
- **`---` lines in tables.** MyST treats `---` after a heading as a horizontal rule. The converter strips the trailing `----` underline from setext headings (Step 2). If you author a new page by hand, use ATX (`## Heading`) directly.

## Self-contained recipe — migrating ANY module

This section is the **generic, module-agnostic** procedure. Substitute `<mod>` everywhere with the module name being ported (`dnn`, `imgproc`, `calib3d`, `3d`, `features`, `objdetect`, `photo`, `gpu`, `ios`, `app`, `others`, `introduction`, …). If a future Claude session is handed this file alone, this is the minimum it needs to know.

> **⚠ Re-read the [Hard rule](#-hard-rule--do-not-change-the-content-only-the-format) section before starting.** Format-only migration. Content (text, code, images) is reproduced verbatim. The legacy `doc/tutorials/<mod>/` tree is never modified or deleted. Both pipelines must continue to build side-by-side.

### 0. Prereqs (run once per environment)

- **Python ≥ 3.10** (the converter uses `X | None` union syntax and `from __future__ import annotations`).
- **Sphinx** + the OpenCV-specific theme stack:

  ```bash
  pip install sphinx pydata-sphinx-theme myst-parser breathe exhale sphinx-autobuild
  ```

  (`sphinx-autobuild` is only needed for the `opencv_docs_serve` live-reload target.)
- **Doxygen ≥ 1.12** is required by the legacy `BUILD_DOCS` pipeline (which produces the XML that Breathe/Exhale ingest).
- Confirm the repo is on branch `Doc_update_pip` (or rebased on top of it) so `docs/` exists.
- Confirm a Doxygen tag-file is available locally — without it, the converter cannot resolve `@ref` links:

  ```bash
  [ -f /tmp/opencv.tag ] || curl -sSL -o /tmp/opencv.tag https://docs.opencv.org/5.x/opencv.tag
  ```

### 1. Enumerate `<mod>` tutorials

```bash
ls doc/tutorials/<mod>/                     # list of per-tutorial subdirs
ls doc/tutorials/<mod>/*/                   # .markdown + images/ inside each
cat doc/tutorials/<mod>/table_of_content_<mod>.markdown   # section grouping
```

If Python / JS counterparts exist they also need ports (each module has at most three tutorial trees):

```bash
ls doc/py_tutorials/py_<mod>/ 2>/dev/null
ls doc/js_tutorials/js_<mod>/ 2>/dev/null
```

### 2. Create target directories

```bash
mkdir -p docs/tutorials/cpp/<mod>/images
# only if applicable:
mkdir -p docs/tutorials/python/<mod>/images
mkdir -p docs/tutorials/javascript/<mod>/images
```

### 3. Run the converter for each tutorial

For each `doc/tutorials/<mod>/<topic>/<topic>.markdown`:

```bash
python3 docs/_tools/dox2myst.py \
    doc/tutorials/<mod>/<topic>/<topic>.markdown \
    docs/tutorials/cpp/<mod>/<topic>.md \
    --tag /tmp/opencv.tag \
    --local docs/_tools/local_refs.json \
    --out-doc tutorials/cpp/<mod>/<topic>
```

A one-liner that does the whole module at once:

```bash
for f in doc/tutorials/<mod>/*/*.markdown; do
  topic=$(basename "$f" .markdown)
  python3 docs/_tools/dox2myst.py "$f" \
      "docs/tutorials/cpp/<mod>/$topic.md" \
      --tag /tmp/opencv.tag \
      --local docs/_tools/local_refs.json \
      --out-doc "tutorials/cpp/<mod>/$topic"
done
```

Then copy images verbatim (the layout flattens — every tutorial's `images/` merges into one shared folder):

```bash
for d in doc/tutorials/<mod>/*/images; do
  [ -d "$d" ] && cp "$d"/* docs/tutorials/cpp/<mod>/images/
done
```

### 4. Author the module index

Create `docs/tutorials/cpp/<mod>/index.md` using [tutorials/cpp/core/index.md](tutorials/cpp/core/index.md) as the canonical template:

- Top-level `# <Module Display Name> (<mod> module)` heading.
- One or more `## <section>` headings — copy the grouping verbatim from `doc/tutorials/<mod>/table_of_content_<mod>.markdown`.
- Each section is a `{list-table}` decorated with `:class: opencv-module-table :widths: 35 65 :header-rows: 1` and `Topic / Description` columns.
- Tail with a `{toctree}` block: `:hidden:` + `:maxdepth: 1`, listing every tutorial filename (without `.md`).

If the description column is empty in the legacy `table_of_content_<mod>.markdown`, write a one-sentence summary by reading the tutorial's "Goal" section.

### 5. Register the module in the parent TOC

[docs/tutorials/cpp/index.md](tutorials/cpp/index.md) already exists. It contains:

- A `{list-table}` (class `opencv-module-table`) — the visible module landing card list.
- A hidden `{toctree}` — the sidebar/prev-next driver.

Each existing entry has one of two shapes:

- **Already ported** — link is local: `[The Core Functionality (core module)](core/index.md)` in the list-table, and `The Core Functionality (core module) <core/index>` in the toctree.
- **Not yet ported** — link points at the legacy site: `[Object Detection (objdetect module)](https://docs.opencv.org/5.x/d2/d64/…)` in both blocks.

When porting `<mod>`, **edit both blocks**: replace the external `https://docs.opencv.org/…` URL with the local `<mod>/index.md` / `<mod>/index` path. Do **not** add new entries — every module already has a placeholder. Likewise for python/javascript trees if they were also ported.

### 6. Build & verify

```bash
cmake -B build -DBUILD_DOCS=ON -DBUILD_DOCS_MODERN=ON .
cmake --build build --target doxygen opencv_docs -j$(nproc)
# OR for interactive editing:
cmake --build build --target opencv_docs_serve     # http://127.0.0.1:8000
```

Per-tutorial checks (see [Verification](#verification) above).

### 7. Spot-check expected output

- `docs/tutorials/cpp/<mod>/<topic>.md` opens with `# <Title>` then the `:::{div} opencv-meta-table` block.
- Every Doxygen directive in the source has been transformed — this grep should return **nothing**:

  ```bash
  grep -nE '@code|@endcode|@ref|@snippet|@note|@warning|@add_toggle|@end_toggle|@tableofcontents|@youtube|@prev_tutorial|@next_tutorial' \
       docs/tutorials/cpp/<mod>/*.md
  ```

- The rendered page in `build/doc/modern_html/tutorials/cpp/<mod>/<topic>.html` matches the legacy page byte-for-byte on text/code/images; only the surrounding chrome differs.
- Image count matches: `find doc/tutorials/<mod> -path '*/images/*' -type f | wc -l` equals the count of files in `docs/tutorials/cpp/<mod>/images/`.
- **Content-equivalence diff** — strip both the legacy and the migrated file of their Doxygen/MyST syntactic chrome and diff the remainder. There should be **zero word-level differences** in prose, **zero character-level differences** in code, and **zero filename differences** in images. If the diff is non-empty, you've edited content — revert and try again.

### 8. Commit pattern (mirrors PR #27's history)

One commit per logical chunk, in order:

1. `Add <mod> module skeleton (index + toctree wiring)`
2. `Port <mod> tutorials to MyST`
3. `Add <mod> images`
4. (if needed) `Fix <mod> build warnings`

### Per-module quirks to look out for

When applying the recipe above, watch for these module-specific shapes:

| Module | Likely quirks |
|---|---|
| `imgproc` | Heavy use of inline math (`\f$ ... \f$`) — confirm `dollarmath` / `amsmath` MyST extensions render them. Many `@snippet` blocks tagged by step. |
| `dnn` | Lots of multi-language code (`@add_toggle_Cpp` / `@add_toggle_Python`) — verify the converter emits `tab-set` correctly. Embedded YouTube videos. External model-zoo links. |
| `3d`, `calib3d` | Module name starts with a digit (`3d`) — `@ref _3d "3d"` workaround already in `doc/CMakeLists.txt`. Verify references resolve. |
| `features` | Renamed from `features2d` in 5.x — old anchors may point to `features2d` group; verify with tag-file. |
| `objdetect` | YOLO / face-detection tutorials embed large Python snippets; check `:language: python` in `doxysnippet` blocks. |
| `gpu` | CUDA-only — sample paths under `samples/gpu/`. Conditionally built; the modern docs target should not depend on CUDA. |
| `ios`, `android` | Tutorials reference platform-specific images and screenshots; ensure they all copy across. |
| `py_tutorials/`, `js_tutorials/` | These are separate trees, not subfolders of a C++ module — use `tutorials/python/<mod>/` and `tutorials/javascript/<mod>/` paths. |

If any module exposes a new Doxygen construct the converter does not yet handle, extend [_tools/dox2myst.py](_tools/dox2myst.py) (the `_OTHER_DIRECTIVE` regex is the index of recognised tags) and re-run on the failing file.
