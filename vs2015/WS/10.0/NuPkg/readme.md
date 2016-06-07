# OpenCV.UWP.native NuGet packages

This VS project will build NuGet packages for all the OpenCV modules.

## Instructions

1. Build `..\x86\OpenCV.sln` in both Debug and Release mode
2. Build `..\x64\OpenCV.sln` in both Debug and Release mode
3. Build `..\ARM\OpenCV.sln` in both Debug and Release mode
4. In Notepad, edit `NuPkg.vcxproj` to point to the correct GitHub repository (e.g. Microsoft/opencv) and the correct GitHub Commit ID that correspond exactly to the sources that you built in steps 1..3. This is needed if you want to get "debug source server" working properly.
5. Now open this `NuPkg.sln` project and build it (it only has Release mode). This will produce all the NuGet packages.
6. You can upload the resulting NuGet packages using `nuget push` command so long as you have permissions. (The NuGet binary is automatically downloaded by step 5 and can be found in the packages\NuGet.CommandLine.3.3.0\tools directory).

## Debug symbols

Theoretically, step 5 above is supposed to also push PDBs to symbolsource.org. But that site has been down for a while, and it will probably time out.

As an alternative, step 5 has also produced an `OpenCV.UWP.native.symbols.zip` file. I suggest to copy this into the NuPkg project directory and check it into GitHub. That way folks will at least be able to download symbols from somewhere.
