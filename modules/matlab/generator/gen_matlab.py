#/usr/bin/env python

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
        jtemplate.filters['convertibleToInt'] = convertibleToInt
        jtemplate.filters['toUpperCamelCase'] = toUpperCamelCase
        jtemplate.filters['toLowerCamelCase'] = toLowerCamelCase
        jtemplate.filters['toUnderCase'] = toUnderCase
        jtemplate.filters['comment']  = comment
        jtemplate.filters['inputs']   = inputs
        jtemplate.filters['ninputs'] = ninputs
        jtemplate.filters['outputs']  = outputs
        jtemplate.filters['noutputs'] = noutputs
        jtemplate.filters['only'] = only
        jtemplate.filters['void'] = void 
        jtemplate.filters['not'] = flip


        # load the templates
        tfunction  = jtemplate.get_template('template_function_base.cpp')
        tclassm    = jtemplate.get_template('template_class_base.m')
        tclassc    = jtemplate.get_template('template_class_base.cpp')
        tdoc       = jtemplate.get_template('template_doc_base.m')
        tconstc    = jtemplate.get_template('template_map_base.cpp')
        tconstm    = jtemplate.get_template('template_map_base.m')

        # create the build directory
        output_source_dir  = output_dir+'/src'
        output_private_dir = output_source_dir+'/private' 
        output_class_dir   = output_dir+'/+cv'
        output_map_dir     = output_dir+'/map'
        if not os.path.isdir(output_source_dir):
          os.mkdir(output_source_dir)
        if not os.path.isdir(output_private_dir):
          os.mkdir(output_private_dir)
        if not os.path.isdir(output_class_dir):
          os.mkdir(output_class_dir)
        if not os.path.isdir(output_map_dir):
          os.mkdir(output_map_dir)

        # populate templates
        for namespace in parse_tree.namespaces:
            # functions
            for function in namespace.functions:
                populated = tfunction.render(fun=function, time=time, includes=namespace.name)
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

        # create a global constants lookup table
        const = dict(constants(todict(parse_tree.namespaces)))
        populatedc = tconstc.render(constants=const)
        populatedm = tconstm.render(constants=const)
        with open(output_map_dir+'/map.cpp', 'wb') as f:
            f.write(populatedc)
        with open(output_dir+'/cv.m', 'wb') as f:
            f.write(populatedm)



if __name__ == "__main__":
    
    # add the hdr_parser to the path
    import sys, re, os, time
    sys.path.append(sys.argv[1])
    from string import Template
    from hdr_parser import CppHeaderParser
    from parse_tree import ParseTree, todict, constants
    from filters import *
    from jinja2 import Environment, PackageLoader

    # get the IO from the command line arguments
    input_files = sys.argv[2:-1]
    output_dir  = sys.argv[-1]

    # create the generator
    mwg = MatlabWrapperGenerator()
    mwg.gen(input_files, output_dir)
