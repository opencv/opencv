# G-API Overview

This is the latest overview slide deck on G-API.

## Prerequisites

- [Emacs] v24 or higher;
- [Org]-mode 8.2.10;
- `pdflatex`;
- `texlive-latex-recommended` ([Beamer] package);
- `texlive-font-utils` (`epstopdf`);
- `wget` (for `get_sty.sh`).

## Building

1. Download and build the [Metropolis] theme with the script:

```
$ ./get_sty.sh
```

2. Now open `gapi_overview.org` with Emacs and press `C-c C-e l P`.

[Emacs]:      https://www.gnu.org/software/emacs/
[Org]:        https://orgmode.org/
[Beamer]:     https://ctan.org/pkg/beamer
[Metropolis]: https://github.com/matze/mtheme
