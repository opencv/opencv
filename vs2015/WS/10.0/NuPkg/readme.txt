TODO: delete each individual targets file.
Instead generate them from just a single targets file.
The .vcxproj AfterBuild task needs to hard-code the "D" suffix based on NuPkg build-time.
And since it's parameterizing that, it might as well also parameterize DLL/LIB name
(which is the only other thing that needs parameterizing)

Although when we come to opencv_winrt with its cvRT.winmd file,
that will probably require its own unique targets file

Here's a powershell to replace

powershell -Command "(gc myFile.txt) -replace 'foo', 'bar' | Out-File myFile.txt"


powershell -Command "(Get-Content generic.targets) -replace 'XXX', 'opencv_core300d' | Out-File OpenCV.UWP.native.core.targets

TODO: verify that for each opencv module, when it #includes stuff from other modules,
it always does it via <opencv2/...>. That's because if it uses quotes, then it won't
survive being fragmented up.

NUGET:
1. In a web-browser, log into NuGet.org and click on your profile. That will show you your API key
2. NuGet setApiKey xxx-xxxxx-xxx-xxx
3. NuGet push OpenCV.UWP.native.core.nupkg
