#!/usr/bin/env python
r"""Check LaTeX math in OpenCV docs.

Finds Doxygen math (``\f$...\f$`` and ``\f[...\f]``) in files tracked by
git, runs latexcheck's ``doxygen`` mode on each one, and prints hits as::

    path:line:column: CODE: source line

Needs the latexcheck fork that can read math in markup files::

    python -m pip install "git+https://github.com/Artoria2e5/latexcheck-py@markdown"

The PyPI release does not have this API.

Usage::

    python doc/tools/check_latex.py
    python doc/tools/check_latex.py --exclude MULTIPLICATION_SIGN
    python doc/tools/check_latex.py modules/imgproc/doc/colors.markdown
    python doc/tools/check_latex.py --output hits.txt

``--exclude`` takes one or more codes, comma-separated. ``--output`` writes
hits to a file, not stdout. It skips ``3rdparty/``.

Common style errors and fixes
=============================

These checks use rough rules, not a real TeX parser. Some hits are false.
Use ``--exclude`` to skip them.

A full scan of the OpenCV tree found 1,576 hits. ``MULTIPLICATION_SIGN`` is
down to 146 hits, mostly real ``*`` multiplication or ``*`` inside
``\texttt{...}`` code. ``TEXT_IN_MATH_MODE`` hit ``\texttt{...}`` 408 of 744
times, and ``UNNECESSARY_MATH_MODE`` hit many lone Greek letters.

Operator names (``BACKSLASH_NEEDED``)
-------------------------------------

Write operators as commands, not as plain letters. ``max``, ``min``,
``log``, ``sin``, ``lim``, ``deg`` and the like have built-in commands::

    V \leftarrow max(R,G,B)
    V \leftarrow \max(R,G,B)

Use ``\operatorname{...}`` or ``\DeclareMathOperator`` if there is no
built-in command.

Words in formulas (``TEXT_IN_MATH_MODE``)
-----------------------------------------

TeX reads a bare name like ``ChannelRange`` as a product of one-letter vars,
so the spacing is bad. Group the letters with a font command that fits what
they mean:

- ``\mathit``: a math name, even a long one; the default for the rest.
- ``\mathsf``: a multi-word plain-language phrase, often with spaces.
- ``\mathtt``: a code var or function name.

Plain prose and punctuation use ``\text``::

    A \leftarrow \max (\mathit{ChannelRange})
    \mathsf{source\ image} \leftarrow \mathtt{src}
    \mathtt{dst} \leftarrow \mathtt{add}(\mathtt{src1}, \mathtt{src2})

    X \leftarrow X/X_n, \text{where} X_n = 0.950456
    X \leftarrow X/X_n, \quad \text{where} \quad X_n = 0.950456

Math mode eats spaces, so ``\text{where}`` needs ``\quad`` (or a
backslash-space) on both sides when it sits between two formulas.

Greek letters are commands too: ``delta`` is five plain letters, while
``\delta`` is the letter::

    Cr \leftarrow (R-Y) \cdot 0.713 + delta
    Cr \leftarrow (R-Y) \cdot 0.713 + \delta

One-word subscripts and superscripts are the exception. ``V_{max}`` and
``R_{avg}`` are fine; ``\mathit`` there costs more than it fixes.

Most hits in this repo are ``\texttt{...}`` code in a formula; use
``\mathtt``. The check also flags the ``\f[`` that starts a long formula,
so read the line first.

Multiplication sign (``MULTIPLICATION_SIGN``)
---------------------------------------------

The old Doxygen ``*`` prefix noise is fixed in the latexcheck fork. Remaining
hits are real multiplication or ``*`` inside ``\texttt{...}`` code. Use
``\cdot`` for the math; leave the code alone.

In math mode use ``\cdot``. Use ``\times`` only for size, cross products and
the like::

    0.299 * R + 0.587 * G + 0.114 * B
    0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B

``CDOT_FOR_READABILITY`` does the same for products that work without a dot
but read better with one.

Text commands (``REPLACE_MBOX_WITH_TEXT``)
------------------------------------------

Use ``\text{...}`` for words in a formula::

    x \mbox{ if } y > 0
    x \text{ if } y > 0

Ellipsis (``ELLIPSIS_LDOTS``)
-----------------------------

Use ``\ldots`` (or ``\dotsc`` in a comma-separated list)::

    a_1, a_2, ..., a_n
    a_1, a_2, \ldots, a_n

Tall delimiters (``LEFT_RIGHT_RECOMMENDED``)
--------------------------------------------

Use ``\left`` and ``\right`` when the brackets hold a fraction, a sum with
limits, or another tall form::

    (\frac{a}{b})^2
    \left(\frac{a}{b}\right)^2

Do not add them to every pair of parentheses. Use them only when the heights
do not match.

Low-level TeX commands
----------------------

``LOW_LEVEL_FONT_COMMANDS``: use ``\mathrm``, ``\mathbf``, ``\mathit`` in
place of ``\rm``, ``\bf``, ``\it``.

``OVER_VS_FRAC``: write ``\frac{A}{B}``, not ``A \over B``.

``CHOOSE_VS_BINOM``: write ``\binom{A}{B}``, not ``A \choose B``.

``INCORPORATE_NOT``: write ``\ne``, not ``\not=``; write ``\notin``, not
``\not\in``.

``LE_AS_SINGLE_COMMAND``: write ``\le`` and ``\ge``, not ``<=`` and ``>=``.

Spacing
-------

Use ``\quad``, ``\,``, ``\;`` or a backslash-space to add space. Do not indent
with runs of spaces (``INDENTATION_WITH_SPACES``). Do not put spaces around
``~`` when you want a non-breaking space.

Other frequent cases
--------------------

``UNNECESSARY_MATH_MODE``: a formula with one mark is often a split formula,
or a ``$...$`` that should not be there. Arrows between words are common: in
prose, use HTML entities ``\leftrightarrow`` -> ``&harr;``,
``\rightarrow`` -> ``&rarr;``, ``\leftarrow`` -> ``&larr;``. A lone Greek
letter such as ``\f$\theta\f$`` is real math; keep it even when the check
flags it.

``SETS_IN_BBFONT``: number sets use ``\mathbb{N}``, ``\mathbb{R}``, etc.

``MOD_NOT_A_COMMAND``: write ``\bmod`` or ``\pmod``, not ``mod``.

``ITALIC_INSTEAD_OF_EMPH``: in text, use ``\emph{...}`` for stress, not
``\textit{...}``.

Exit status
-----------

``0`` means no hits, ``1`` means hits (or a tool error). A count by code goes
to stderr.
"""
import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

try:
    from latexcheck.embedded import BUILTIN_FORMATS, extract_math, perform_checks_embedded
except ImportError:
    sys.exit(
        "This tool requires the latexcheck fork with embedded-math support "
        "(https://github.com/Artoria2e5/latexcheck-py, branch 'markdown')."
    )

ROOT = Path(__file__).resolve().parents[2]
RULES = BUILTIN_FORMATS["doxygen"].rules

TEXT_EXTENSIONS = {
    ".md", ".markdown", ".rst", ".txt", ".tex",
    ".h", ".hpp", ".hh", ".hxx", ".ipp", ".inl", ".tpp",
    ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh",
    ".py", ".pyx", ".java", ".js", ".ts", ".m", ".mm", ".cs",
    ".go", ".rs", ".swift", ".kt", ".pl", ".pm", ".sh",
    ".cmake", ".yml", ".yaml", ".json", ".xml", ".html", ".css",
    ".in", ".proto", ".cl",
}


def tracked_files():
    try:
        output = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files"], universal_newlines=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit("Could not list tracked files: %s" % exc)
    return output.splitlines()


def matches(rel_path, selectors):
    if not selectors:
        return True
    for selector in selectors:
        selector = selector.replace("\\", "/").rstrip("/")
        if rel_path == selector or rel_path.startswith(selector + "/"):
            return True
    return False


def line_column(text, pos):
    line = text.count("\n", 0, pos) + 1
    col = pos - text.rfind("\n", 0, pos)
    return line, col


def source_line(text, pos):
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end < 0:
        end = len(text)
    line = text[start:end].strip()
    if len(line) > 200:
        line = line[:200] + "..."
    return line


def parse_excludes(values):
    excluded = set()
    for value in values or []:
        for code in value.split(","):
            code = code.strip().upper()
            if code:
                excluded.add(code)
    return excluded


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="*", help="Only check files under these paths (default: all tracked files)")
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="CODE",
        help="Skip a check code (repeatable; comma-separated)",
    )
    parser.add_argument("-o", "--output", help="Write findings to this file instead of stdout")
    args = parser.parse_args()

    excluded = parse_excludes(args.exclude)
    findings = []
    files_with_formulas = 0
    formula_count = 0
    counts = Counter()

    for rel_path in tracked_files():
        if "3rdparty/" in rel_path:
            continue
        if os.path.splitext(rel_path)[1].lower() not in TEXT_EXTENSIONS:
            continue
        if not matches(rel_path, args.paths):
            continue

        full_path = ROOT / rel_path
        try:
            if full_path.stat().st_size > 2_000_000:
                continue
            text = full_path.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            continue

        if "\\f$" not in text and "\\f[" not in text:
            continue

        spans = extract_math(text, RULES)
        if not spans:
            continue

        files_with_formulas += 1
        formula_count += len(spans)
        errors = perform_checks_embedded(text, RULES)
        for code, positions in errors.items():
            if code in excluded:
                continue
            for pos in positions:
                line, col = line_column(text, pos)
                findings.append((rel_path, line, col, code, source_line(text, pos)))
                counts[code] += 1

    findings.sort()
    if args.output:
        stream = open(args.output, "w", encoding="utf-8")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        stream = sys.stdout
    try:
        for rel_path, line, col, code, snippet in findings:
            stream.write("%s:%d:%d: %s: %s\n" % (rel_path, line, col, code, snippet))
    finally:
        if args.output:
            stream.close()

    print(
        "Checked %d files with formulas (%d formulas); %d findings."
        % (files_with_formulas, formula_count, len(findings)),
        file=sys.stderr,
    )
    if counts:
        print("  " + ", ".join("%s=%d" % item for item in counts.most_common()), file=sys.stderr)

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
