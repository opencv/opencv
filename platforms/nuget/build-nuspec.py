# run script
# .\build-nuspec.py --package_name "OpenCV CPP" --package_version "0.0.1" --native_build_path "C:\opencv-cpp-build" --packages_path "C:\opencv-cpp-build" --sources_path "D:\Github\opencv"
import os, sys, argparse

authors = 'OpenCV'
owners = 'OpenCV'
description = 'This is description.'
copyright = '\xc2\xa9 ' + 'OpenCV. All rights reserved.'
tags = 'OpenCV Computer Vision'
iconUrl = 'https://avatars.githubusercontent.com/u/5009934?s=200&v=4'
project_url = 'https://github.com/opencv/opencv'

def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenCV CPP create nuget spec script ",
                                     usage='')
    # Main arguments
    parser.add_argument("--package_name", required=True, help="Package name. e.g.: OpenCV.CPP")
    parser.add_argument("--package_version", required=True, help="Package version. e.g: 1.0.0")
    # parser.add_argument("--target_architecture", required=True, help="e.g.: x64")
    parser.add_argument("--native_build_path", required=True, help="Native build output directory.")
    parser.add_argument("--packages_path", required=True, help="Nuget packages output directory.")
    parser.add_argument("--sources_path", required=True, help="OpenCV source code root.")
    parser.add_argument("--targets_path", required=True, help="Path name for the generated targets file.")
    parser.add_argument("--targets_file", required=True, help="File name for the generated targets file.")

    return parser.parse_args()



def generate_files(list, args):
    files_list = ['<files>']

    # Process headers
    # files_list.append('<file src=' + '"' + os.path.join(args.sources_path,
    #                                                     'include\\onnxruntime\\core\\session\\onnxruntime_*.h') +
    #                   '" target="build\\native\\include" />')

    # Process Readme, License, ThirdPartyNotices, Privacy
    # files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'LICENSE.txt') + '" target="LICENSE.txt" />')
    
    # add source includes folder
    derived_path = os.path.join(args.sources_path, 'include\**\*.*')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\include" />')

    # add vc14 compiled lib & dlls
    derived_path = os.path.join(args.sources_path, 'x64\\vc14\\lib\\**\\opencv_world*.lib')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\x64\\vc14.0\\lib" />')
    derived_path = os.path.join(args.sources_path, 'x64\\vc14\\lib\\**\\opencv_world*.dll')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\x64\\vc14.0\\bin" />')
    derived_path = os.path.join(args.sources_path, 'x64\\vc14\\lib\\**\\opencv_world*.pdb')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\x64\\vc14.0\\bin" />')

    # add vc15 compiled lib & dlls
    derived_path = os.path.join(args.sources_path, 'x64\\vc15\\lib\\**\\opencv_world*.lib')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\x64\\vc15.0\\lib" />')
    derived_path = os.path.join(args.sources_path, 'x64\\vc15\\lib\\**\\opencv_world*.dll')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\x64\\vc15.0\\bin" />')
    derived_path = os.path.join(args.sources_path, 'x64\\vc15\\lib\\**\\opencv_world*.pdb')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\x64\\vc15.0\\bin" />')

    # add vc16 compiled lib & dlls
    derived_path = os.path.join(args.sources_path, 'x64\\vc16\\lib\\**\\opencv_world*.lib')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\x64\\vc16.0\\lib" />')
    derived_path = os.path.join(args.sources_path, 'x64\\vc16\\lib\\**\\opencv_world*.dll')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\x64\\vc16.0\\bin" />')
    derived_path = os.path.join(args.sources_path, 'x64\\vc16\\lib\\**\\opencv_world*.pdb')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="build\\native\\x64\\vc16.0\\bin" />')

    # add readme file
    derived_path = os.path.join(args.sources_path, 'README.md')
    files_list.append(f'<file src="{os.path.join(derived_path)}" target="README.md" />')

    # add generated targets file
    files_list.append(f'<file src="{args.targets_path}" target="{args.targets_file}" />')
    files_list.append('</files>')

    list += files_list



def generate_nuspec(args):
    lines = ['<?xml version="1.0"?>']
    lines.append(f'<package>')
    generate_metadata(lines, args)
    generate_files(lines, args)
    lines.append(f'</package>')
    return lines


def generate_metadata(list, args):
    metadata_list = ['<metadata>']
    # package id
    metadata_list.append(f'<id>{args.package_name}</id>')
    # package version
    metadata_list.append(f'<version>{args.package_version}</version>')
    # package authors
    metadata_list.append(f'<authors>{authors}</authors>')
    # package owners
    metadata_list.append(f'<owners>{owners}</owners>')
    # package description
    metadata_list.append(f'<description>{description}</description>')
    # package copyright
    metadata_list.append(f'<copyright>{copyright}</copyright>')
    # package tags
    metadata_list.append(f'<tags>{tags}</tags>')
    # package icon
    metadata_list.append(f'<iconUrl>{iconUrl}</iconUrl>')
    # package license
    derived_path = os.path.join(args.sources_path, 'LICENSE.txt')
    metadata_list.append(f'<license type="file">{os.path.join(derived_path)}</license>')
    # package project url
    metadata_list.append(f'<projectUrl>{project_url}</projectUrl>')

    # generate_repo_url(metadata_list, 'https://github.com/opencv/opencv.git', args.commit_id)
    metadata_list.append('</metadata>')

    list += metadata_list


def main():
    # Parse arguments
    args = parse_arguments()

    print("started parser")

    # Generate nuspec
    lines = generate_nuspec(args)

    # Create the nuspec needed to generate the Nuget
    with open(os.path.join(args.native_build_path, 'opencv-cpp.nuspec'), 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    print("stopped parser")

if __name__ == "__main__":
    sys.exit(main())