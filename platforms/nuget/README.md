# Building the NuGet package for OpenCV

This repository helps generate the OpenCV Nuget package. The generated .targets, .nuspec and .nupkg files are excluded here, but a sample [nupkg file can be viewed online](https://drive.google.com/file/d/1-rMDqsAbpoUVXg601yGZWCW0TIuUr2Dg/view?usp=sharing)

## Pre-requisites
- CMake
- [NuGet](https://www.nuget.org/downloads)
- [Visual Studio](https://visualstudio.microsoft.com/downloads/)

## Setup 
- Add nuget.exe to your `PATH` environment variable
- Install Visual Studio for the current user, preferably:
    - Visual Studio 2019 (has compiler vc16)
    - Visual Studio 2018 (has compiler vc15)
    - Visual Studio 2017 (has compiler vc14)
- Build OpenCV from source - [Reference](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html)
    - Generate separate build folders for vc14, vc15 & vc16
    - Make sure to select the option/ variable to build the "opencv_build" binary (dll file)
- TODO: Merge the build files/ folders in the source opencv pull directory

## (Manual) Package creation and pushing

### TODO: Create a code signing certificate
https://docs.microsoft.com/en-us/nuget/create-packages/sign-a-package#register-the-certificate-on-nugetorg

### Create the .targets file
- In command prompt, run `build-targets.py` with arguments
    - target_path - directory where the targets file should go
    - target_file - file name (e.g. opencv-cpp.targets)
- `.\build-targets.py --targets_path "<path-to-output-targets-file>" --targets_file "opencv-cpp.targets"`

### Create the nuspec file
- In command prompt, run `build-nuspec.py` with arguments
    - package_name - name of the package
    - package_version - version number of the package
    - output_path - directory where nuspec file is written to
    - sources_path - directory where opencv source code was pulled and built (NOT the /source directory; instead the outer /opencv directory should be passed here)
    - targets_path - directory where targets file was generated in previous step
    - targets_file - targets file name from previous step
- `.\build-nuspec.py --package_name "OpenCV CPP" --package_version "0.0.1" --output_path "<path-to-output-folder>" --sources_path "<path to opencv git pull>" --targets_path "<path-to-output-targets-file" --targets_file "opencv-cpp.targets"`


### Create the nuget package (nupkg)
- In command prompt, cd to the output_path from previous steps, where you have already generated the .targets and .nuspec files.
- run `nuget pack opencv-cpp.nuspec -OutputDirectory <path-to-output-nupkg>`
- TODO: sign the package

### Publish the package (nupkg) to nuget.org
- [Reference](https://docs.microsoft.com/en-us/nuget/nuget-org/publish-a-package)
- In command prompt, `nuget setApiKey <your_API_key>` to set the API key
- `nuget push opencv-cpp.nupkg -Source https://api.nuget.org/v3/index.json` to push package to nuget.org
## (Automated) Package creation and pushing

TODO: Create a .bat file to automate the manual process documented above (i.e. CI will only have to run this one .bat file.)