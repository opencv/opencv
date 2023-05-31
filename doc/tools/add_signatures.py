"""
This code adds Python/Java signatures to the docs.

TODO: Do the same thing for Java
* using javadoc/ get all the methods/classes/constants to a json file

TODO:
* clarify when there are several C++ signatures corresponding to a single Python function.
    i.e: calcHist():
    http://docs.opencv.org/3.2.0/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d
* clarify special case:
    http://docs.opencv.org/3.2.0/db/de0/group__core__utils.html#ga4910d7f86336cd4eff9dd05575667e41
"""
from __future__ import print_function
import sys
sys.dont_write_bytecode = True  # Don't generate .pyc files / __pycache__ directories

import os
from pprint import pprint
import re
import logging
import json

import html_functions
import doxygen_scan

loglevel=os.environ.get("LOGLEVEL", None)
if loglevel:
    logging.basicConfig(level=loglevel)

ROOT_DIR = sys.argv[1]
PYTHON_SIGNATURES_FILE = sys.argv[2]
JAVA_OR_PYTHON = sys.argv[3]

ADD_JAVA = False
ADD_PYTHON = False
if JAVA_OR_PYTHON == "python":
    ADD_PYTHON = True

python_signatures = dict()
with open(PYTHON_SIGNATURES_FILE, "rt") as f:
    python_signatures = json.load(f)
    print("Loaded Python signatures: %d" % len(python_signatures))

import xml.etree.ElementTree as ET
root = ET.parse(ROOT_DIR + 'opencv.tag')
files_dict = {}

# constants and function from opencv.tag
namespaces = root.findall("./compound[@kind='namespace']")
#print("Found {} namespaces".format(len(namespaces)))
for ns in namespaces:
    ns_name = ns.find("./name").text
    #print('NS: {}'.format(ns_name))
    doxygen_scan.scan_namespace_constants(ns, ns_name, files_dict)
    doxygen_scan.scan_namespace_functions(ns, ns_name, files_dict)

# class methods from opencv.tag
classes = root.findall("./compound[@kind='class']")
#print("Found {} classes".format(len(classes)))
for c in classes:
    c_name = c.find("./name").text
    file = c.find("./filename").text
    #print('Class: {} => {}'.format(c_name, file))
    doxygen_scan.scan_class_methods(c, c_name, files_dict)

print('Doxygen files to scan: %s' % len(files_dict))

files_processed = 0
files_skipped = 0
symbols_processed = 0

for file in files_dict:
    #if file != "dd/d9e/classcv_1_1VideoWriter.html":
    #if file != "d4/d86/group__imgproc__filter.html":
    #if file != "df/dfb/group__imgproc__object.html":
    #    continue
    #print('File: ' + file)

    anchor_list = files_dict[file]
    active_anchors = [a for a in anchor_list if a.cppname in python_signatures]
    if len(active_anchors) == 0: # no linked Python symbols
        #print('Skip: ' + file)
        files_skipped = files_skipped + 1
        continue

    active_anchors_dict = {a.anchor: a for a in active_anchors}
    if len(active_anchors_dict) != len(active_anchors):
        logging.info('Duplicate entries detected: %s -> %s (%s)' % (len(active_anchors), len(active_anchors_dict), file))

    files_processed = files_processed + 1

    #pprint(active_anchors)
    symbols_processed = symbols_processed + len(active_anchors_dict)

    logging.info('File: %r' % file)
    html_functions.insert_python_signatures(python_signatures, active_anchors_dict, ROOT_DIR + file)

print('Done (processed files %d, symbols %d, skipped %d files)' % (files_processed, symbols_processed, files_skipped))
