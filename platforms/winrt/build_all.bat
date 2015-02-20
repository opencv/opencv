@echo off


set FOUND_VC=0

if defined VS120COMNTOOLS (
    set VSTOOLS="%VS120COMNTOOLS%"
    set VC_VER=120
    set FOUND_VC=1
)

set VSTOOLS=%VSTOOLS:"=%
set "VSTOOLS=%VSTOOLS:\=/%"

set VSVARS="%VSTOOLS%vsvars32.bat"

if not defined VSVARS (
    echo Can't find VC2013 installed!
    goto ERROR
)


echo./*
echo. * Building OpenCV
echo. */
echo.


call %VSVARS%

if %FOUND_VC%==1 (
    call:DoMSBuild ..\..\bin\WP\8.1\x86\OpenCV.sln Debug
    call:DoMSBuild ..\..\bin\WP\8.1\x86\INSTALL.vcxproj Debug
    call:DoMSBuild ..\..\bin\WP\8.1\x86\OpenCV.sln Release
    call:DoMSBuild ..\..\bin\WP\8.1\x86\INSTALL.vcxproj Release

    call:DoMSBuild ..\..\bin\WP\8.1\ARM\OpenCV.sln Debug
    call:DoMSBuild ..\..\bin\WP\8.1\ARM\INSTALL.vcxproj Debug
    call:DoMSBuild ..\..\bin\WP\8.1\ARM\OpenCV.sln Release
    call:DoMSBuild ..\..\bin\WP\8.1\ARM\INSTALL.vcxproj Release

    call:DoMSBuild ..\..\bin\WS\8.1\x86\OpenCV.sln Debug
    call:DoMSBuild ..\..\bin\WS\8.1\x86\INSTALL.vcxproj Debug
    call:DoMSBuild ..\..\bin\WS\8.1\x86\OpenCV.sln Release
    call:DoMSBuild ..\..\bin\WS\8.1\x86\INSTALL.vcxproj Release

    call:DoMSBuild ..\..\bin\WS\8.1\ARM\OpenCV.sln Debug
    call:DoMSBuild ..\..\bin\WS\8.1\ARM\INSTALL.vcxproj Debug
    call:DoMSBuild ..\..\bin\WS\8.1\ARM\OpenCV.sln Release
    call:DoMSBuild ..\..\bin\WS\8.1\ARM\INSTALL.vcxproj Release

    call:DoMSBuild ..\..\bin\WP\8.0\x86\OpenCV.sln Debug
    call:DoMSBuild ..\..\bin\WP\8.0\x86\INSTALL.vcxproj Debug
    call:DoMSBuild ..\..\bin\WP\8.0\x86\OpenCV.sln Release
    call:DoMSBuild ..\..\bin\WP\8.0\x86\INSTALL.vcxproj Release

    call:DoMSBuild ..\..\bin\WP\8.0\ARM\OpenCV.sln Debug
    call:DoMSBuild ..\..\bin\WP\8.0\ARM\INSTALL.vcxproj Debug
    call:DoMSBuild ..\..\bin\WP\8.0\ARM\OpenCV.sln Release
    call:DoMSBuild ..\..\bin\WP\8.0\ARM\INSTALL.vcxproj Release

    call:DoMSBuild ..\..\bin\WS\8.0\x86\OpenCV.sln Debug
    call:DoMSBuild ..\..\bin\WS\8.0\x86\INSTALL.vcxproj Debug
    call:DoMSBuild ..\..\bin\WS\8.0\x86\OpenCV.sln Release
    call:DoMSBuild ..\..\bin\WS\8.0\x86\INSTALL.vcxproj Release

    call:DoMSBuild ..\..\bin\WS\8.0\ARM\OpenCV.sln Debug
    call:DoMSBuild ..\..\bin\WS\8.0\ARM\INSTALL.vcxproj Debug
    call:DoMSBuild ..\..\bin\WS\8.0\ARM\OpenCV.sln Release
    call:DoMSBuild ..\..\bin\WS\8.0\ARM\INSTALL.vcxproj Release
)

echo.&goto:EOF


::--------------------------------------------------------
::-- DoMSBuild
::--------------------------------------------------------
:DoMSBuild
msbuild %~1 /p:Configuration="%~2" /m
@if errorlevel 1 goto :ERROR
goto:EOF

:ERROR
pause
:EOF