#/usr/bin/env python

def substitute(build, output_dir):

    # setup the template engine
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    jtemplate    = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)

    # add the filters
    jtemplate.filters['csv'] = csv
    jtemplate.filters['stripExtraSpaces'] = stripExtraSpaces

    # load the template
    template = jtemplate.get_template('template_build_info.m')

    # create the build directory
    output_dir  = output_dir+'/+cv'
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)

    # populate templates
    populated = template.render(build=build)
    with open(os.path.join(output_dir, 'buildInformation.m'), 'wb') as f:
        f.write(populated)

if __name__ == "__main__":
   
    # parse the input options
    import sys, re, os
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--os')
    parser.add_argument('--arch', nargs=2)
    parser.add_argument('--compiler', nargs='+')
    parser.add_argument('--mex_arch')
    parser.add_argument('--mex_script')
    parser.add_argument('--mex_opts', default=['-largeArrayDims'], nargs='*')
    parser.add_argument('--cxx_flags', default=[], nargs='*')
    parser.add_argument('--opencv_version', default='', nargs='?')
    parser.add_argument('--commit', default='Not in working git tree', nargs='?')
    parser.add_argument('--modules', nargs='+')
    parser.add_argument('--configuration')
    parser.add_argument('--outdir')
    build = parser.parse_args()

    from filters import *
    from jinja2 import Environment, FileSystemLoader

    # populate the build info template
    substitute(build, build.outdir)
