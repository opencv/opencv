#!/usr/bin/env python
"""
Common utilities. These should be compatible with Python 2 and 3.
"""

from __future__ import print_function
import sys, re
from subprocess import check_call, check_output, CalledProcessError

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

def get_xcode_major():
    ret = check_output(["xcodebuild", "-version"]).decode('utf-8')
    m = re.match(r'Xcode\s+(\d+)\..*', ret, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    else:
        raise Exception("Failed to parse Xcode version")

def get_xcode_setting(var, projectdir):
    ret = check_output(["xcodebuild", "-showBuildSettings"], cwd = projectdir)
    m = re.search("\s" + var + " = (.*)", ret)
    if m:
        return m.group(1)
    else:
        raise Exception("Failed to parse Xcode settings")