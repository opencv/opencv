#/usr/bin/env python

import sys, re, os.path
from string import Template
from hdr_parser import CppHeaderParser
from parse_tree import ParseTree, todict
from filters import *
from jinja2 import Environment, PackageLoader

class MatlabWrapperGenerator(object):

    def gen(self, input_files, output_files):
        # parse each of the files and store in a dictionary
        # as a separate "namespace"
        parser = CppHeaderParser()
        ns = {}
        for file in input_files:
            # get the file name
            name = os.path.splitext(os.path.basename(file))[0]
            ns[name] = parser.parse(file)

        # cleanify the parser output
        parse_tree = ParseTree()
        parse_tree.build(ns)
       
        # setup the template engine
        jtemplate = Environment(loader=PackageLoader('templates', ''))

        # add the custom filters
        jtemplate.filters['toUpperCamelCase'] = toUpperCamelCase
        jtemplate.filters['toLowerCamelCase'] = toLowerCamelCase
        jtemplate.filters['toUnderCase'] = toUnderCase
        jtemplate.filters['comment'] = comment

        # load the templates
        function  = jtemplate.get_template('template_function_base.cpp')
        classm    = jtemplate.get_template('template_class_base.m')
        classc    = jtemplate.get_template('template_class_base.cpp')
        doc       = jtemplate.get_template('template_doc_base.m')

        # populate!
