# run script
# .\build-targets.py --targets_path "C:\opencv-cpp-build" --targets_file "opencv-cpp.targets"
import os, sys, argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenCV CPP create targets file script", usage='')
    # Main arguments
    parser.add_argument("--targets_path", required=True, help="Path name for the generated targets file.")
    parser.add_argument("--targets_file", required=True, help="File name for the generated targets file.")

    return parser.parse_args()

def generate_property_group(list, args):
    files_list = ['<PropertyGroup>']
    files_list.append(f'<VisualStudioVersion Condition="\'$(VisualStudioVersion)\' == \'\'">10.0</VisualStudioVersion>')
    files_list.append(f'<LibraryType Condition="\'$(Configuration)\'==\'Debug\'">d</LibraryType>')
    files_list.append(f'<LibraryType Condition="\'$(Configuration)\'==\'Release\'"></LibraryType>')
    files_list.append('</PropertyGroup>')

    list += files_list

def generate_item_definition_group(list, args):
    files_list = ['<ItemDefinitionGroup>']
    files_list.append(f'<ClCompile>')
    files_list.append(f'<AdditionalIncludeDirectories>$(MSBuildThisFileDirectory)\\include\\;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>')
    files_list.append(f'</ClCompile>')
    files_list.append('</ItemDefinitionGroup>')

    list += files_list

def generate_item_definition_group_2(list, args):
    files_list = ['<ItemDefinitionGroup>']
    files_list.append(f'<Link>')
    files_list.append(f'<AdditionalLibraryDirectories>$(MSBuildThisFileDirectory)\\x$(Platform)\\vc$(VisualStudioVersion)\\lib\\;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>')
    files_list.append(f'<AdditionalDependencies>opencv_world*$(LibraryType).lib;%(AdditionalDependencies)</AdditionalDependencies>')
    files_list.append(f'</Link>')
    files_list.append('</ItemDefinitionGroup>')

    list += files_list

def generate_item_group(list, args):
    files_list = ['<ItemGroup>']
    files_list.append(f'<None Include="$(MSBuildThisFileDirectory)\\x64\\vc$(VisualStudioVersion)\\bin\\opencv_world*$(LibraryType).dll">')
    files_list.append(f'<Link>opencv_world*$(LibraryType).dll</Link>')
    files_list.append(f'<CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>')
    files_list.append(f'<Visible>false</Visible>')
    files_list.append(f'</None>')
    files_list.append('</ItemGroup>')

    list += files_list


def generate_targets(args):
    lines = ['<?xml version="1.0"?>']
    lines.append(f'<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003" ToolsVersion="15.0">')
    generate_property_group(lines, args)
    generate_item_definition_group(lines, args)
    generate_item_definition_group_2(lines, args)
    generate_item_group(lines, args)
    lines.append(f'</Project>')
    return lines


def main():
    # Parse arguments
    args = parse_arguments()
    print("started parser")

    # Generate targets
    lines = generate_targets(args)

    # Create the nuspec needed to generate the Nuget
    with open(os.path.join(args.targets_path, args.targets_file), 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    print("stopped parser")

if __name__ == "__main__":
    sys.exit(main())