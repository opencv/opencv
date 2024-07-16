'''
This script creates the templates for .nuspec and .targets files. These files are required to be generated with the correct attributes / key-value pairs, in order to auto-configure the nuget package to include the DLL/lib files on importing in your Visual Studio project.
'''

import os, argparse
from pathlib import Path
from shutil import copyfile
from jinja2 import Environment, FileSystemLoader
PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)

VERSION_PATH = str(Path(PATH).parents[1]) + "/modules/core/include/opencv2/core/version.hpp"
version_file = Path(VERSION_PATH).read_text()

version_major = version_file.split("CV_VERSION_MAJOR")[1].split("\n")[0].replace(" ", "")
version_minor = version_file.split("CV_VERSION_MINOR")[1].split("\n")[0].replace(" ", "")
version_revision = version_file.split("CV_VERSION_REVISION")[1].split("\n")[0].replace(" ", "")

version_string = f'{version_major}.{version_minor}.{version_revision}'
version_string_raw = f'{version_major}{version_minor}{version_revision}'

params_nuspec = {
'id': 'OpenCVNuget',
'version': version_string,
'description': 'OpenCV Nuget Package for C++',
'tags': 'OpenCV, opencv',
'authors': 'OpenCV',
'compilers': ['vc14', 'vc15', 'vc16'],
'architectures': ['x64'],
'directory': '',
}

params_targets = {
'version': version_string_raw,
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenCV CPP create nuget spec script ",
                                     usage='')
    # Main arguments
    parser.add_argument("--package_name", required=False, help="Package name. e.g.: OpenCV.CPP")
    parser.add_argument("--package_version", required=False, help="Package version. e.g: 1.0.0")
    parser.add_argument("--build_directory", required=False, help="Path to the build directory where DLL & lib files exist. e.g: C:\some-folder")

    return parser.parse_args()

def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)

def create_nuspec_targets():
    # Write the nuspec file
    FILE_PATH = str(Path(PATH).parents[2])
    
    context = {
        'params': params_nuspec
    }
    UP_PATH = os.path.dirname(PATH)
    UP_PATH = os.path.dirname(UP_PATH)
    UP_PATH = os.path.dirname(UP_PATH)
    fname_nuspec = UP_PATH + r'\build\install\opencv.nuspec'

    with open(fname_nuspec, 'w') as f:
        html = render_template('OpenCVNuget.nuspec', context)
        f.write(html)

    # Write the targets file
    context = {
        'params': params_targets
    }
    fname_targets = UP_PATH + r'\build\install\OpenCVNuget.targets'
    
    with open(fname_targets, 'w') as f:
        html = render_template('OpenCVNuget.targets', context)
        f.write(html)

def main():
    # Parse arguments
    args = parse_arguments()
    if args.package_version is not None:
        params_nuspec['compilers'] = args.package_version.split(',')
    if args.build_directory is not None:
        params_nuspec['directory'] = args.build_directory
    # Create template files
    create_nuspec_targets()
if __name__ == "__main__":
    main()