import os, sys, argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenCV CPP create nuget spec script ",
                                     usage='')
    # Main arguments
    parser.add_argument("--package_name", required=True, help="Package name. e.g.: OpenCV.CPP")
    parser.add_argument("--package_version", required=True, help="Package version. e.g: 1.0.0")
    parser.add_argument("--target_architecture", required=True, help="e.g.: x64")
    parser.add_argument("--native_build_path", required=True, help="Native build output directory.")
    parser.add_argument("--packages_path", required=True, help="Nuget packages output directory.")
    # parser.add_argument("--sources_path", required=True, help="OpenCV source code root.")

    return parser.parse_args()



def generate_id(list, package_name):
    list.append('<id>' + package_name + '</id>')


def generate_version(list, package_version):
    list.append('<version>' + package_version + '</version>')


def generate_authors(list, authors):
    list.append('<authors>' + authors + '</authors>')


def generate_owners(list, owners):
    list.append('<owners>' + owners + '</owners>')


def generate_description(list, description):
    list.append('<description>' + description + '</description>')


def generate_copyright(list, copyright):
    list.append('<copyright>' + copyright + '</copyright>')


def generate_tags(list, tags):
    list.append('<tags>' + tags + '</tags>')


def generate_icon_url(list, icon_url):
    list.append('<iconUrl>' + icon_url + '</iconUrl>')


def generate_license(list):
    list.append('<license type="file">LICENSE.txt</license>')


def generate_project_url(list, project_url):
    list.append('<projectUrl>' + project_url + '</projectUrl>')


def generate_repo_url(list, repo_url, commit_id):
    list.append('<repository type="git" url="' + repo_url + '"' + ' commit="' + commit_id + '" />')



def generate_files(list, args):
    files_list = ['<files>']

    # Process headers
    # files_list.append('<file src=' + '"' + os.path.join(args.sources_path,
    #                                                     'include\\onnxruntime\\core\\session\\onnxruntime_*.h') +
    #                   '" target="build\\native\\include" />')

    # Process Readme, License, ThirdPartyNotices, Privacy
    # files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'LICENSE.txt') + '" target="LICENSE.txt" />')
    files_list.append('<file src=' + '"' + os.path.join('D:\Github\opencv', 'README.md') + '" target="README.md" />')
    files_list.append('</files>')

    list += files_list



def generate_nuspec(args):
    lines = ['<?xml version="1.0"?>']
    lines.append('<package>')
    generate_metadata(lines, args)
    generate_files(lines, args)
    lines.append('</package>')
    return lines


def generate_metadata(list, args):
    metadata_list = ['<metadata>']
    generate_id(metadata_list, args.package_name)
    generate_version(metadata_list, args.package_version)
    generate_authors(metadata_list, 'OpenCV')
    generate_owners(metadata_list, 'OpenCV')
    generate_description(metadata_list, 'This is description.')
    generate_copyright(metadata_list, '\xc2\xa9 ' + 'OpenCV. All rights reserved.')
    generate_tags(metadata_list, 'OpenCV Computer Vision')
    generate_icon_url(metadata_list, 'https://avatars.githubusercontent.com/u/5009934?s=200&v=4')
    generate_license(metadata_list)
    generate_project_url(metadata_list, 'https://github.com/opencv/opencv')
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


if __name__ == "__main__":
    sys.exit(main())