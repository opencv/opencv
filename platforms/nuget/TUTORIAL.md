# Using the NuGet Package (OpenCVNuget) in your native C++ project
This tutorial takes you through downloading the .nupkg file and setting it up in your C++ project, so you can begin writing C++ applications with OpenCV quickly.

## Downloading the latest .nupkg (NuGet Package) file

- You can download the package from
    - The releases section on Github
    (https://github.com/opencv/opencv/releases)
    - The releases section on OpenCV website
    (https://opencv.org/releases/)
- Ensure that you have the correct package based on your Visual Studio or Microsoft Visual C++ Version. Currently supported versions are:
    - Visual Studio 2019 - msvc 16.0
    - Visual Studio 2018 - msvc 15.0
    - Visual Studio 2017 - msvc 14.0

## Setting up a local Package Source in the Visual Studio IDE

- Make sure you have a *Source* directory, such as your Downloads folder setup, where you can keep all your .nupkg files
- Start Visual Studio IDE
- Create a new *Console App* Project
- In the *Solution Manager*, right-click on References and select *Manage NuGet Packages*
- A new NuGet window opens; click on the *settings icon* in the top right corner of the window
- A new pop-up window appears with Package Sources selected and shown to the user
- Click on the Green button on the top right corner to add a new source
- You can name the new source anything, and select the *Source* directory on your computer
- Once you have selected a directory, you can click on *Update*, then click on OK to close the pop-up window


## Browsing and selecting the newly added offline package(s) for installation

- You can now choose the newly added Offline directory from the dropdown on the top right corner (before the settings button)
- In the already opened NuGet window, you can select *Browse* option on the top left corner
- You can now see all your offline packages in the *Browse* section, click on OpenCVNuget and select *Install*
- On the prompt window, click OK to agree and install the *OpenCVNuget* package

## Getting started with your new project

- You can now import OpenCV directly in your C++ project.
`#include <opencv/core/core.hpp>`