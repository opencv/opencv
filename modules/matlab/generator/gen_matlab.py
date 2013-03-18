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
        jtemplate = Environment(loader=PackageLoader('templates', ''), trim_blocks=True)

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
        if not os.path.isdir(output_dir):
          os.mkdir(output_dir)

        # populate!
        function  = parse_tree.namespaces[0].functions[0]
        print function
        populated = tfunction.render(fun=function, time=time)
        with open(output_dir+'/'+function.name+'.cpp', 'wb') as f:
          f.write(populated)
        #for name, namespace in ns:
        #  for function in namespace.functions:
        #    print 'populating function tempaltes from '+name
        #    populated = tfunction.render(function)

