#!/usr/bin/env python

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

    # populate template
    populated = template.render(build=build, time=time)
    with open(os.path.join(output_dir, 'buildInformation.m'), 'wb') as f:
        f.write(populated.encode('utf-8'))

if __name__ == "__main__":
    """
    Usage: python build_info.py --jinja2 /path/to/jinja2/engine
                                --os os_version_string
                                --arch [bitness processor]
                                --compiler [id version]
                                --mex_arch arch_string
                                --mex_script /path/to/mex/script
                                --cxx_flags [-list -of -flags -to -passthrough]
                                --opencv_version version_string
                                --commit commit_hash_if_using_git
                                --modules [core imgproc highgui etc]
                                --configuration Debug/Release
                                --outdir /path/to/write/build/info

    build_info.py generates a Matlab function that can be invoked with a call to
      >> cv.buildInformation();

    This function prints a summary of the user's OS, OpenCV and Matlab build
    given the information passed to this module. build_info.py invokes Jinja2
    on the template_build_info.m template.
    """

    # parse the input options
    import sys, re, os, time
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--jinja2')
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

    # add jinja to the path
    sys.path.append(build.jinja2)

    from filters import *
    from jinja2 import Environment, FileSystemLoader

    # populate the build info template
    substitute(build, build.outdir)
