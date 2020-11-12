#!/usr/bin/env python
"""
Common utilities.
"""

from __future__ import print_function
import sys
from subprocess import check_call

def execute(cmd, cwd = None):
    print("Executing: %s in %s" % (cmd, cwd), file=sys.stderr)
    print('Executing: ' + ' '.join(cmd))
    retcode = check_call(cmd, cwd = cwd)
    if retcode != 0:
        raise Exception("Child returned:", retcode)

def print_header(text):
    print("="*60)
    print(text)
    print("="*60)

def print_error(text):
    print("="*60, file=sys.stderr)
    print("ERROR: %s" % text, file=sys.stderr)
    print("="*60, file=sys.stderr)