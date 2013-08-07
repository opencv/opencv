#/usr/bin/env python

class MatlabWrapperGenerator(object):

    def gen(self, module_root, modules, extras, output_dir):
        # parse each of the files and store in a dictionary
        # as a separate "namespace"
        parser = CppHeaderParser()
        rst    = rst_parser.RstParser(parser)
        rst_parser.verbose = False
        rst_parser.show_warnings = False
        rst_parser.show_errors = False
        rst_parser.show_critical_errors = False

        ns  = dict((key, []) for key in modules)
        doc = dict((key, []) for key in modules)
        path_template = Template('${module}/include/opencv2/${module}.hpp')

        for module in modules:
            # construct a header path from the module root and a path template
            header = os.path.join(module_root, path_template.substitute(module=module))
            # parse the definitions
            ns[module] = parser.parse(header)
            # parse the documentation
            rst.parse(module, os.path.join(module_root, module))
            doc[module] = rst.definitions
            rst.definitions = {}

        for extra in extras:
            module = extra.split(":")[0]
            header = extra.split(":")[1]
            ns[module] = ns[module] + parser.parse(header) if module in ns else parser.parse(header)

        # cleanify the parser output
        parse_tree = ParseTree()
        parse_tree.build(ns)

        # setup the template engine
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        jtemplate    = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)

        # add the custom filters
        jtemplate.filters['formatMatlabConstant'] = formatMatlabConstant
        jtemplate.filters['convertibleToInt'] = convertibleToInt
        jtemplate.filters['toUpperCamelCase'] = toUpperCamelCase
        jtemplate.filters['toLowerCamelCase'] = toLowerCamelCase
        jtemplate.filters['toUnderCase'] = toUnderCase
        jtemplate.filters['stripTags'] = stripTags
        jtemplate.filters['filename'] = filename
        jtemplate.filters['comment']  = comment
        jtemplate.filters['inputs']   = inputs
        jtemplate.filters['ninputs'] = ninputs
        jtemplate.filters['outputs']  = outputs
        jtemplate.filters['noutputs'] = noutputs
        jtemplate.filters['qualify'] = qualify
        jtemplate.filters['slugify'] = slugify
        jtemplate.filters['only'] = only
        jtemplate.filters['void'] = void 
        jtemplate.filters['not'] = flip

        # load the templates
        tfunction  = jtemplate.get_template('template_function_base.cpp')
        tclassm    = jtemplate.get_template('template_class_base.m')
        tclassc    = jtemplate.get_template('template_class_base.cpp')
        tdoc       = jtemplate.get_template('template_doc_base.m')
        tconst     = jtemplate.get_template('template_map_base.m')

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
            for method in namespace.methods:
                populated = tfunction.render(fun=method, time=time, includes=namespace.name)
                with open(output_source_dir+'/'+method.name+'.cpp', 'wb') as f:
                    f.write(populated)
                if namespace.name in doc and method.name in doc[namespace.name]:
                    populated = tdoc.render(fun=method, doc=doc[namespace.name][method.name], time=time)
                    with open(output_class_dir+'/'+method.name+'.m', 'wb') as f:
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
        populated = tconst.render(constants=const, time=time)
        with open(output_dir+'/cv.m', 'wb') as f:
            f.write(populated)



if __name__ == "__main__":
   
    # parse the input options
    import sys, re, os, time
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--hdrparser')
    parser.add_argument('--rstparser')
    parser.add_argument('--moduleroot', default='', required=False)
    parser.add_argument('--modules', nargs='*', default=[], required=False)
    parser.add_argument('--extra', nargs='*', default=[], required=False) 
    parser.add_argument('--outdir')
    args = parser.parse_args()

    # add the hdr_parser and rst_parser modules to the path
    sys.path.append(args.hdrparser)
    sys.path.append(args.rstparser)

    from string import Template
    from hdr_parser import CppHeaderParser
    import rst_parser
    from parse_tree import ParseTree, todict, constants
    from filters import *
    from jinja2 import Environment, FileSystemLoader

    # create the generator
    mwg = MatlabWrapperGenerator()
    mwg.gen(args.moduleroot, args.modules, args.extra, args.outdir)
