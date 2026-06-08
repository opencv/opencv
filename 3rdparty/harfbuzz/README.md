# HarfBuzz

<div align="center">

<p><img src="HarfBuzz.png" alt="HarfBuzz Logo" width="256"/></p>

[![Linux CI Status](https://github.com/harfbuzz/harfbuzz/actions/workflows/linux.yml/badge.svg)](https://github.com/harfbuzz/harfbuzz/actions/workflows/linux.yml)
[![macoOS CI Status](https://github.com/harfbuzz/harfbuzz/actions/workflows/macos.yml/badge.svg)](https://github.com/harfbuzz/harfbuzz/actions/workflows/macos.yml)
[![Windows CI Status](https://github.com/harfbuzz/harfbuzz/actions/workflows/msvc.yml/badge.svg)](https://github.com/harfbuzz/harfbuzz/actions/workflows/msvc.yml)
[![OSS-Fuzz Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/harfbuzz.svg)](https://oss-fuzz-build-logs.storage.googleapis.com/index.html#harfbuzz)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/15166/badge.svg)](https://scan.coverity.com/projects/harfbuzz)
[![Packaging status](https://repology.org/badge/tiny-repos/harfbuzz.svg)](https://repology.org/project/harfbuzz/versions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/harfbuzz/harfbuzz/badge)](https://securityscorecards.dev/viewer/?uri=github.com/harfbuzz/harfbuzz)

</div>

HarfBuzz started as a text shaping engine but has grown into a
full font platform — the `ffmpeg` of text shaping.  It primarily
supports [OpenType][1], but also [Apple Advanced Typography][2].

HarfBuzz shapes the majority of text on modern screens.

HarfBuzz is optimized for robustness, correctness, and performance
— in that order. Achieve all.

**[Try it live at harfbuzz-world.cc](https://harfbuzz-world.cc/)** — an interactive playground for shaping, subsetting, rasterization, vector output, and GPU rendering, all running in your browser.

Here is a quick map of its components:

### Core libraries

| Library | Description |
|---------|-------------|
| **libharfbuzz** | Text shaping, draw API, paint API. Highly configurable (see [CONFIG.md](CONFIG.md)). Optional integration backends compiled in: hb-ft (FreeType), hb-coretext (macOS), hb-uniscribe (Windows), hb-directwrite (Windows), hb-gdi (Windows), hb-glib, hb-graphite2. |
| **libharfbuzz-subset** | Font subsetting and variable-font instancing. |

### Auxiliary libraries

| Library | Description |
|---------|-------------|
| **libharfbuzz-icu** | ICU Unicode integration. |
| **libharfbuzz-cairo** | Cairo rendering integration. |
| **libharfbuzz-gobject** | GObject/GI bindings. |

### Experimental libraries

| Library | Description |
|---------|-------------|
| **libharfbuzz-raster** | Glyph rasterization to bitmaps, including color fonts. Uses hb-draw and hb-paint. |
| **libharfbuzz-vector** | Glyph output to vector formats (currently SVG), including color fonts. Uses hb-draw and hb-paint. |
| **libharfbuzz-gpu** | Encodes glyph outlines for GPU rasterization (Slug algorithm). Provides shader sources in GLSL, WGSL, MSL, and HLSL. [Live demo.](https://harfbuzz.github.io/hb-gpu-demo/) |

Notable missing feature: font hinting (including autohinting)
is not implemented.  For hinted rasterization, use FreeType or
Skrifa.

For simplified builds, amalgamated sources are available:
`harfbuzz.cc` (just libharfbuzz), `harfbuzz-subset.cc` (just
libharfbuzz-subset), or `harfbuzz-world.cc` (everything, driven
by a custom `hb-features.h`).  For a live in-browser playground
plus a worked example of the world.cc single-file build, see
[harfbuzz-world.cc][26].

### Command-line tools

| Tool | Description |
|------|-------------|
| **hb-shape** | Shape text and display glyph output. |
| **hb-view** | Render shaped text to an image. |
| **hb-subset** | Subset and optimize fonts. |
| **hb-info** | Display font metadata. |
| **hb-raster** | Render glyphs to bitmap images. |
| **hb-vector** | Render glyphs to vector formats (SVG). |
| **hb-gpu** | Interactive GPU text rendering. |

The canonical source tree and bug trackers are available on [github][4].
Both development and user support discussion around HarfBuzz happen on
[github][4] as well.

For license information, see [COPYING](COPYING).

## API stability

The API that comes with `hb.h` will not change incompatibly. Other, peripheral,
headers are more likely to go through minor modifications, but again, we do our
best to never change API in an incompatible way. We will never break the ABI.

The API and ABI are stable even across major version number jumps. In fact,
current HarfBuzz is API/ABI compatible all the way back to the 0.9.x series.
If one day we need to break the API/ABI, that would be called a new library.

As such, we bump the major version number only when we add major new features,
the minor version when there is new API, and the micro version when there
are bug fixes.

## Documentation

For user manual as well as API documentation, check: https://harfbuzz.github.io

## Download

Tarball releases and Win32/Win64 binary bundles are available on the
[github releases][3] page.

## Development

For build information, see [BUILD.md](BUILD.md).

For custom configurations, see [CONFIG.md](CONFIG.md).

For testing and profiling, see [TESTING.md](TESTING.md).

For using with Python, see [README.python.md](README.python.md). There is also [uharfbuzz](https://github.com/harfbuzz/uharfbuzz).

For cross-compiling to Windows from Linux or macOS, see [README.mingw.md](README.mingw.md).

To report bugs or submit patches please use [github][4] issues and pull-requests.

### Developer documents

To get a better idea of where HarfBuzz stands in the text rendering stack you
may want to read [State of Text Rendering 2024][6].
Here are a few presentation slides about HarfBuzz over the years:

- 2026 – [HarfBuzz at 20!][25]
- 2016 – [Ten Years of HarfBuzz][20]
- 2014 – [Unicode, OpenType, and HarfBuzz: Closing the Circle][7]
- 2012 – [HarfBuzz, The Free and Open Text Shaping Engine][8]
- 2009 – [HarfBuzz: the Free and Open Shaping Engine][9]

More presentations and papers are available on [behdad][11]'s website.
In particular, the following _studies_ are relevant to HarfBuzz development:

- 2025 – [AAT layout caches][24]
- 2025 – [OpenType Layout lookup caches][23]
- 2025 – [Introducing HarfRust][22]
- 2025 – [Subsetting][21]
- 2025 – [Caching][12]
- 2025 – [`hb-decycler`][13]
- 2022 – [`hb-iter`][14]
- 2022 – [A C library written in C++][15]
- 2022 – [The case of the slow `hb-ft` `>h_advance` function][18]
- 2022 – [PackTab: A static integer table packer][16]
- 2020 – [HarfBuzz OT+AAT "Unishaper"][19]
- 2014 – [Building the Indic Shaper][17]
- 2012 – [Memory Consumption][10]


## Name

HarfBuzz /hærfˈbɒːz/

From Persian حرف (*Harf*: letter) and باز (*Buzz*: open).
Transliteration of the Persian calque for *OpenType*.

As a noun: *The* Open Source *text shaping* engine.

As an adjective: Insincerely talkative; glib. A nod to the
GNOME project where HarfBuzz originates from.

The logo shows حرف‌باز in the IranNastaliq font, on a Damascus
steel background.

> Background: Originally there was this font format called TrueType. People and
> companies started calling their type engines all things ending in Type:
> FreeType, CoolType, ClearType, etc. And then came OpenType, which is the
> successor of TrueType. So, for my OpenType implementation, I decided to stick
> with the concept but use the Persian translation. Which is fitting given that
> Persian is written in the Arabic script, and OpenType is an extension of
> TrueType that adds support for complex script rendering, and HarfBuzz is an
> implementation of OpenType text shaping.

## Users

HarfBuzz is used in Android, Chrome, ChromeOS, Firefox, Flutter, GNOME, GTK+, KDE,
Qt, LibreOffice, OpenJDK, XeTeX, Adobe Photoshop, Illustrator, InDesign,
Microsoft Edge, Amazon Kindle, PlayStation, Godot Engine, Unreal Engine,
Figma, Canva, QuarkXPress, Scribus, smart TVs,
car displays, and many other places.

<p align="center">
  <a href="https://xkcd.com/2347/" rel="nofollow">
    <img src="xkcd.png" width="256" alt="xkcd-derived image">
  </a>
</p>

## Distribution

<details>
  <summary>Packaging status of HarfBuzz</summary>

[![Packaging status](https://repology.org/badge/vertical-allrepos/harfbuzz.svg?header=harfbuzz)](https://repology.org/project/harfbuzz/versions)

</details>

[1]: https://docs.microsoft.com/en-us/typography/opentype/spec/
[2]: https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6AATIntro.html
[3]: https://github.com/harfbuzz/harfbuzz/releases
[4]: https://github.com/harfbuzz/harfbuzz
[6]: https://behdad.org/text2024
[7]: https://docs.google.com/presentation/d/1x97pfbB1gbD53Yhz6-_yBUozQMVJ_5yMqqR_D-R7b7I/preview
[8]: https://docs.google.com/presentation/d/1ySTZaXP5XKFg0OpmHZM00v5b17GSr3ojnzJekl4U8qI/preview
[9]: https://behdad.org/doc/harfbuzz2009-slides.pdf
[10]: https://docs.google.com/document/d/12jfNpQJzeVIAxoUSpk7KziyINAa1msbGliyXqguS86M/preview
[11]: https://behdad.org/
[12]: https://docs.google.com/document/d/1_VgObf6Je0J8byMLsi7HCQHnKo2emGnx_ib_sHo-bt4/preview
[13]: https://docs.google.com/document/d/1Y-u08l9YhObRVObETZt1k8f_5lQdOix9TRH3zEXaoAw/preview
[14]: https://docs.google.com/document/d/1o-xvxCbgMe9JYFHLVnPjk01ZY_8Cj0vB9-KTI1d0nyk/preview
[15]: https://docs.google.com/document/d/18hI56KJpvXtwWbc9QSaz9zzhJwIMnrJ-zkAaKS-W-8k/preview
[16]: https://docs.google.com/document/d/1Xq3owVt61HVkJqbLFHl73il6pcTy6PdPJJ7bSouQiQw/preview
[17]: https://docs.google.com/document/d/1wMPwVNBvsIriamcyBO5aNs7Cdr8lmbwLJ8GmZBAswF4/preview
[18]: https://docs.google.com/document/d/1wskYbA-czBt57oH9gEuGf3sWbTx7bfOiEIcDs36-heo/preview
[19]: https://prezi.com/view/THNPJGFVDUCWoM20syev/
[20]: https://behdad.org/doc/harfbuzz10years-slides.pdf
[21]: https://docs.google.com/document/d/1_vZrt97OorJ0jA1YzJ29LRcGr3YGrNJANdOABjVZGEs/preview
[22]: https://docs.google.com/document/d/1aH_waagdEM5UhslQxCeFEb82ECBhPlZjy5_MwLNLBYo/preview
[23]: https://docs.google.com/document/d/1hRd5oYQJLrt0JuwWhEJWi7wh_9rbaIJkX6IR9DW7rZQ/preview
[24]: https://docs.google.com/document/d/1a3K6fHjsiWW36vSzwJwCwEBOgznunKs80PSpBbpfHiA/preview
[25]: https://docs.google.com/presentation/d/1o9Exz1c-Lr-dJjA8dcBn_Vl_Y37cupmFzmclMjBE_Bc/view
[26]: https://harfbuzz-world.cc/
