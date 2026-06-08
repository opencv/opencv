#!/usr/bin/env python3
"""
hb_extract.py -- copy the minimal subset of HarfBuzz sources used by OpenCV.

OpenCV vendors HarfBuzz as a normal static library: every .cc is a separate
translation unit (no unity / amalgamation build). This script copies exactly
the set of sources that the upstream unity file `harfbuzz-world.cc` would
compile for the requested HB_HAS_* sections, plus all headers they need.

What it does NOT do (on purpose, unlike the upstream world build):
  - it does NOT emit `harfbuzz-world.cc` (we compile each .cc directly);
  - it does NOT copy .cc files found under src/OT/ and src/graph/ as if they
    were headers -- only real translation units listed by harfbuzz-world.cc
    are copied, so `file(GLOB_RECURSE src/*.cc)` in CMakeLists.txt picks up
    exactly the right set and never tries to compile a non-TU .cc such as
    src/graph/test-classdef-graph.cc.

The OpenCV subset is: core + HB_HAS_RASTER, with the default HarfBuzz config
(thread-safety on, no HB_TINY, no platform backends).

Usage (to refresh this very directory, just point it at a harfbuzz checkout):
  python hb_extract.py path/to/harfbuzz

  The defaults already produce the OpenCV subset: output '.', features
  HB_HAS_RASTER, and it also copies README.md + COPYING. Override only if needed:
    -o DIR              output root (default '.')
    -f FLAGS            comma-separated HB_HAS_* (default HB_HAS_RASTER)
    -a FILES            extra files (default README.md,COPYING)
    --list-flags        print available HB_HAS_* flags and exit

-a paths are relative to the harfbuzz repo root (parent of src/):
  -a README.md            -> copied to   <out>/README.md
  -a COPYING              -> copied to   <out>/COPYING
  -a src/hb-raster.h      -> copied to   <out>/src/hb-raster.h
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def parse_world_cc(path):
    """Return (core_files, sections) parsed from harfbuzz-world.cc.

    core_files -- list of .cc paths (relative to src/) in the unconditional
                  "Core library" section.
    sections   -- dict HB_HAS_XXX -> list of .cc paths from that #ifdef block.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    include_re = re.compile(r'^\s*#include\s+"([^"]+\.cc)"')

    core_files = []
    sections = {}

    STATE_PREAMBLE, STATE_CORE, STATE_IFDEF = "preamble", "core", "ifdef"
    state = STATE_PREAMBLE
    current_flag = None
    ifdef_depth = 0

    for line in lines:
        if state == STATE_PREAMBLE:
            if "/* Core library." in line:
                state = STATE_CORE

        elif state == STATE_CORE:
            m = include_re.match(line)
            if m:
                core_files.append(m.group(1))
            else:
                m = re.match(r"^\s*#ifdef\s+(HB_HAS_\w+)", line)
                if m:
                    current_flag = m.group(1)
                    sections[current_flag] = []
                    state = STATE_IFDEF
                    ifdef_depth = 1

        elif state == STATE_IFDEF:
            if re.match(r"^\s*#if", line):
                ifdef_depth += 1
            elif re.match(r"^\s*#endif", line):
                ifdef_depth -= 1
                if ifdef_depth == 0:
                    state = STATE_CORE
                    current_flag = None
            else:
                m = include_re.match(line)
                if m and current_flag is not None:
                    sections[current_flag].append(m.group(1))

    return core_files, sections


def copy_files(file_list, src_dir, out_src_dir, label=""):
    for rel in file_list:
        src = src_dir / rel
        dst = out_src_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"  [WARN] not found: {src}", file=sys.stderr)
            continue
        shutil.copy2(src, dst)
        print(f"  [{label or 'core'}] src/{rel}")


def copy_headers(src_dir, out_src_dir, public_h=True):
    """Always copies: src/*.hh, src/*.h (public API), and *.h/*.hh from
    src/OT/** and src/graph/**.

    NOTE: unlike a naive "copy everything", .cc files under OT/ and graph/
    are intentionally NOT copied here -- the only ones we need are real
    translation units and those are copied via the parsed world.cc lists.
    Copying e.g. src/graph/test-classdef-graph.cc would make CMake's
    file(GLOB_RECURSE src/*.cc) try to compile a non-TU file.
    """
    count = 0

    def _copy(p):
        nonlocal count
        dst = out_src_dir / p.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        count += 1

    for hdr in src_dir.glob("*.hh"):
        _copy(hdr)

    for subdir in ("OT", "graph"):
        d = src_dir / subdir
        if not d.exists():
            continue
        for hdr in d.rglob("*"):
            if hdr.is_file() and hdr.suffix in (".h", ".hh"):
                _copy(hdr)

    if public_h:
        for hdr in src_dir.glob("*.h"):
            _copy(hdr)

    return count


def write_features_h(all_flags, enabled_flags, out_src_dir):
    """Generate the public feature-detection header hb-features.h.

    Internal HarfBuzz sources do not include this header; it only lets
    downstream code do `#if HB_HAS_RASTER`. We define the enabled flags and
    leave the rest undefined. HB_TINY is intentionally NOT set.
    """
    lines = [
        "/* Auto-generated by hb_extract.py -- do not edit. */",
        "#ifndef HB_FEATURES_H",
        "#define HB_FEATURES_H",
        "",
        "HB_BEGIN_DECLS",
        "",
    ]
    for flag in all_flags:
        if flag in enabled_flags:
            lines.append(f"#define {flag} 1")
        else:
            lines.append(f"/* #undef {flag} */")
    lines += ["", "HB_END_DECLS", "", "#endif /* HB_FEATURES_H */", ""]
    (out_src_dir / "hb-features.h").write_text("\n".join(lines), encoding="utf-8")
    print("  [gen] src/hb-features.h")


def main():
    parser = argparse.ArgumentParser(
        description="Copy the OpenCV subset of HarfBuzz sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python hb_extract.py ~/work/harfbuzz -o . -f HB_HAS_RASTER -a README.md,COPYING
  python hb_extract.py ~/work/harfbuzz --list-flags
        """,
    )
    parser.add_argument("harfbuzz_dir", metavar="harfbuzz-dir",
                        help="path to the harfbuzz repo root (contains src/)")
    parser.add_argument("-o", "--output", metavar="DIR", default=".",
                        help="output root directory; sources go into DIR/src/ "
                             "(default: current directory)")
    parser.add_argument("-f", "--features", metavar="FLAGS", default="HB_HAS_RASTER",
                        help="comma-separated HB_HAS_* flags (default: HB_HAS_RASTER)")
    parser.add_argument("-a", "--add", metavar="FILES", default="README.md,COPYING",
                        help="comma-separated files relative to the harfbuzz repo "
                             "root (default: README.md,COPYING)")
    parser.add_argument("--no-public-headers", action="store_true",
                        help="skip copying public src/*.h API headers "
                             "(OT/, graph/, *.hh are always copied)")
    parser.add_argument("--list-flags", action="store_true",
                        help="list available HB_HAS_* flags and exit")
    args = parser.parse_args()

    repo_dir = Path(args.harfbuzz_dir).resolve()
    world_cc_path = repo_dir / "src" / "harfbuzz-world.cc"
    if not world_cc_path.exists():
        print(f"error: {world_cc_path} not found", file=sys.stderr)
        sys.exit(1)

    src_dir = world_cc_path.parent
    core_files, sections = parse_world_cc(world_cc_path)

    if args.list_flags:
        for flag, files in sorted(sections.items()):
            print(f"  {flag:<22}  ({len(files)} .cc files)")
        return

    enabled_flags = [f.strip() for f in args.features.split(",") if f.strip()]
    extra_files = [f.strip() for f in args.add.split(",") if f.strip()]

    out_dir = Path(args.output).resolve()
    out_src_dir = out_dir / "src"

    # Start from a clean src/ so removed-upstream files do not linger.
    if out_src_dir.exists():
        shutil.rmtree(out_src_dir)
    out_src_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nsource : {repo_dir}")
    print(f"output : {out_dir}")
    print(f"flags  : {enabled_flags}\n")

    print("core:")
    copy_files(core_files, src_dir, out_src_dir)

    for flag in enabled_flags:
        flag_files = sections.get(flag)
        if not flag_files:
            print(f"\n[{flag}]: no .cc files (flag not found in harfbuzz-world.cc)")
            continue
        print(f"\n[{flag}]:")
        copy_files(flag_files, src_dir, out_src_dir, label=flag)

    if extra_files:
        print("\nextra:")
        for rel in extra_files:
            src = repo_dir / rel
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                print(f"  [WARN] not found: {src}", file=sys.stderr)
                continue
            shutil.copy2(src, dst)
            print(f"  [add] {rel}")

    n = copy_headers(src_dir, out_src_dir,
                     public_h=not args.no_public_headers)
    print(f"\nheaders: {n} files copied")

    print("\ngenerated:")
    write_features_h(sorted(sections.keys()), enabled_flags, out_src_dir)

    all_cc = sorted(out_src_dir.rglob("*.cc"))
    all_h = list(out_src_dir.rglob("*.h")) + list(out_src_dir.rglob("*.hh"))
    print(f"\ndone: {len(all_cc)} .cc  |  {len(all_h)} headers  ->  {out_dir}")


if __name__ == "__main__":
    main()
