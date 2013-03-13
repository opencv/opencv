# -*- coding: utf-8 -*-
"""
    jinja2._markupsafe._bundle
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script pulls in markupsafe from a source folder and
    bundles it with Jinja2.  It does not pull in the speedups
    module though.

    :copyright: Copyright 2010 by the Jinja team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
import sys
import os
import re


def rewrite_imports(lines):
    for idx, line in enumerate(lines):
        new_line = re.sub(r'(import|from)\s+markupsafe\b',
                          r'\1 jinja2._markupsafe', line)
        if new_line != line:
            lines[idx] = new_line


def main():
    if len(sys.argv) != 2:
        print 'error: only argument is path to markupsafe'
        sys.exit(1)
    basedir = os.path.dirname(__file__)
    markupdir = sys.argv[1]
    for filename in os.listdir(markupdir):
        if filename.endswith('.py'):
            f = open(os.path.join(markupdir, filename))
            try:
                lines = list(f)
            finally:
                f.close()
            rewrite_imports(lines)
            f = open(os.path.join(basedir, filename), 'w')
            try:
                for line in lines:
                    f.write(line)
            finally:
                f.close()


if __name__ == '__main__':
    main()
