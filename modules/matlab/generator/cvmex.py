#!/usr/bin/env python

def substitute(cv, output_dir):

    # setup the template engine
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    jtemplate    = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)

    # add the filters
    jtemplate.filters['cellarray'] = cellarray
    jtemplate.filters['split'] = split
    jtemplate.filters['csv'] = csv

    # load the template
    template = jtemplate.get_template('template_cvmex_base.m')

    # create the build directory
    output_dir  = output_dir+'/+cv'
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)

    # populate template
    populated = template.render(cv=cv, time=time)
    with open(os.path.join(output_dir, 'mex.m'), 'wb') as f:
        f.write(populated.encode('utf-8'))

if __name__ == "__main__":
    """
    Usage: python cvmex.py  --jinja2 /path/to/jinja2/engine
                            --opts [-list -of -opts]
                            --include_dirs [-list -of -opencv_include_directories]
                            --lib_dir opencv_lib_directory
                            --libs [-lopencv_core -lopencv_imgproc ...]
                            --flags [-Wall -opencv_build_flags ...]
                            --outdir /path/to/generated/output

    cvmex.py generates a custom mex compiler that automatically links OpenCV
    libraries to built sources where appropriate. The calling syntax is the
    same as the builtin mex compiler, with added cv qualification:
      >> cv.mex(..., ...);
    """

    # parse the input options
    import sys, re, os, time
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--jinja2')
    parser.add_argument('--opts')
    parser.add_argument('--include_dirs')
    parser.add_argument('--lib_dir')
    parser.add_argument('--libs')
    parser.add_argument('--flags')
    parser.add_argument('--outdir')
    cv = parser.parse_args()

    # add jinja to the path
    sys.path.append(cv.jinja2)

    from filters import *
    from jinja2 import Environment, FileSystemLoader

    # populate the mex base template
    substitute(cv, cv.outdir)
