# Using the NuGet Package (OpenCVNuget) in your native C++ project
This tutorial takes you through downloading the .nupkg file and setting it up in your C++ project, so you can begin writing C++ applications with OpenCV quickly.

## Downloading the latest .nupkg (NuGet Package) file

- You can download the package from
    - The releases section on Github
    (https://github.com/AdityaMulgundkar/opencv-nuget/releases)
- Ensure that you have the correct package based on your Visual Studio or Microsoft Visual C++ Version. Currently supported versions are:
    - Visual Studio 2019 - msvc 16.0
    - Visual Studio 2018 - msvc 15.0
    - Visual Studio 2017 - msvc 14.0

## Setting up a local Package Source in the Visual Studio IDE

- Make sure you have a **Source** directory, such as your Downloads folder (or create a new folder such as C:\nuget-packages) setup, where you can keep all your .nupkg files
- Start Visual Studio IDE
- Click on Tool > Options
- Browse to Nuget Package Manager > Package Sources 
- Click on the Green button on the top right corner to add a new source
![Adding Offline package sources](https://imgur.com/0p8dv1D.png)
- You can name the new source anything, and select the **Source** directory on your computer
- Once you have selected a directory, you can click on *Update*, then click on OK to close the pop-up window


## Browsing and selecting the newly added offline package(s) for installation

- In the NuGet window, change package source to *All* in order to see online as well as the recently added offline packages
- Browse and install the *OpenCVNuget* package
![Adding Offline package sources](https://imgur.com/dnz0nF3.png)

## Getting started with your new project

- You can now import OpenCV directly in your C++ project.
```
#include <opencv/core/core.hpp>
#include <opencv/core/highgui.hpp>
#include <iostream>

int main()
{
    cv::Mat img = cv::imread("<path to image>");
    cv::imshow("a", img);
    cv::waitKey();
}
```