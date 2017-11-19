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
import os
import re
import sys
import logging
import html_functions
import doxygen_scan

loglevel=os.environ.get("LOGLEVEL", None)
if loglevel:
    logging.basicConfig(level=loglevel)


ROOT_DIR = sys.argv[1]
PYTHON_SIGNATURES_FILE = sys.argv[2]
JAVA_PYTHON = sys.argv[3]

ADD_JAVA = False
ADD_PYTHON = False
if JAVA_PYTHON == "python":
    ADD_PYTHON = True

import json
python_signatures = dict()
with open(PYTHON_SIGNATURES_FILE, "rt") as f:
    python_signatures = json.load(f)
    print("Loaded Python signatures: %d" % len(python_signatures))

# only name -> class
# name and ret -> constant
# name, ret, arg-> function / class method

class Configuration():
    def __init__(self):
        self.ADD_PYTHON = ADD_PYTHON
        self.python_signatures = python_signatures
        self.ADD_JAVA = ADD_JAVA

config = Configuration()

import xml.etree.ElementTree as ET
root = ET.parse(ROOT_DIR + 'opencv.tag')
files_dict = dict()

# constants and function from opencv.tag
namespaces = root.findall("./compound[@kind='namespace']")
#print("Found {} namespaces".format(len(namespaces)))
for ns in namespaces:
    ns_name = ns.find("./name").text
    #print('NS: {}'.format(ns_name))

    files_dict = doxygen_scan.scan_namespace_constants(ns, ns_name, files_dict)
    files_dict = doxygen_scan.scan_namespace_functions(ns, ns_name, files_dict)

# class methods from opencv.tag
classes = root.findall("./compound[@kind='class']")
#print("Found {} classes".format(len(classes)))
for c in classes:
    c_name = c.find("./name").text
    name = ns_name + '::' + c_name
    file = c.find("./filename").text
    #print('Class: {} => {}'.format(name, file))
    files_dict = doxygen_scan.scan_class_methods(c, c_name, files_dict)

# test
for file in files_dict:
    soup = html_functions.load_html_file(ROOT_DIR + file)
    if file == "dd/d9e/classcv_1_1VideoWriter.html":#"d4/d86/group__imgproc__filter.html":#"d4/d86/group__imgproc__filter.html":
        anchor_list = files_dict[file]
        counter = 0
        anchor_tmp_list = []
        for anchor in anchor_list:
            counter += 1
            # if the next anchor shares the same C++ name (= same method/function), join them together
            if counter < len(anchor_list) and anchor_list[counter].cppname == anchor.cppname:
                anchor_tmp_list.append(anchor)
                continue
            else:
                anchor_tmp_list.append(anchor)
            # check if extists a python equivalent signature
            for signature in python_signatures: # signature is a key with the C++ name
                if signature == anchor.cppname: # if available name in python
                    # they should also have the same type
                    soup = html_functions.append_python_signature(python_signatures[signature], anchor_tmp_list, soup)
                    #print(signature)
            # reset anchor temporary list
            anchor_tmp_list[:] = []
        html_functions.update_html(ROOT_DIR + file, soup)
