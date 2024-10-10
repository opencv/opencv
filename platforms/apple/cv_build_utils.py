#!/usr/bin/env python3
"""
Common utilities. These should be compatible with Python3.
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

def get_xcode_version():
    """
    Returns the major and minor version of the current Xcode
    command line tools as a tuple of (major, minor)
    """
    ret = check_output(["xcodebuild", "-version"]).decode('utf-8')
    m = re.match(r'Xcode\s+(\d+)\.(\d+)', ret, flags=re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    else:
        raise Exception("Failed to parse Xcode version")

def get_xcode_setting(var, projectdir):
    ret = check_output(["xcodebuild", "-showBuildSettings"], cwd = projectdir).decode('utf-8')
    m = re.search("\s" + var + " = (.*)", ret)
    if m:
        return m.group(1)
    else:
        raise Exception("Failed to parse Xcode settings")

def get_cmake_version():
    """
    Returns the major and minor version of the current CMake
    command line tools as a tuple of (major, minor, revision)
    """
    ret = check_output(["cmake", "--version"]).decode('utf-8')
    m = re.match(r'cmake\sversion\s+(\d+)\.(\d+).(\d+)', ret, flags=re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        raise Exception("Failed to parse CMake version")
