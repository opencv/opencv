#/usr/bin/env python

import sys, re, os, time
from string import Template
from hdr_parser import CppHeaderParser
from parse_tree import ParseTree, todict
from filters import *
from jinja2 import Environment, PackageLoader

class MatlabWrapperGenerator(object):

    def gen(self, input_files, output_dir):
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
        jtemplate = Environment(loader=PackageLoader('templates', ''), trim_blocks=True, lstrip_blocks=True)

        # add the custom filters
        jtemplate.filters['toUpperCamelCase'] = toUpperCamelCase
        jtemplate.filters['toLowerCamelCase'] = toLowerCamelCase
        jtemplate.filters['toUnderCase'] = toUnderCase
        jtemplate.filters['comment'] = comment

        # load the templates
        tfunction  = jtemplate.get_template('template_function_base.cpp')
        tclassm    = jtemplate.get_template('template_class_base.m')
        tclassc    = jtemplate.get_template('template_class_base.cpp')
        tdoc       = jtemplate.get_template('template_doc_base.m')

        # create the build directory
        output_source_dir  = output_dir+'/src'
        output_private_dir = output_source_dir+'/private' 
        output_class_dir   = output_dir+'/+cv'
        if not os.path.isdir(output_source_dir):
          os.mkdir(output_source_dir)
        if not os.path.isdir(output_private_dir):
          os.mkdir(output_private_dir)
        if not os.path.isdir(output_class_dir):
          os.mkdir(output_class_dir)

        # populate templates
        for namespace in parse_tree.namespaces:
            # functions
            for function in namespace.functions:
                populated = tfunction.render(fun=function, time=time)
                with open(output_source_dir+'/'+function.name+'.cpp', 'wb') as f:
                    f.write(populated)
            # classes
            for clss in namespace.classes:
                # cpp converter
                populated = tclassc.render(clss=clss, time=time)
                with open(output_private_dir+'/'+clss.name+'Bridge.cpp', 'wb') as f:
                    f.write(populated)
                # matlab classdef
                populated = tclassm.render(clss=clss, time=time)
                with open(output_class_dir+'/'+clss.name+'.m', 'wb') as f:
                    f.write(populated)
