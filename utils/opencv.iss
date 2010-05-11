; -- OpenCV script for Inno Setup 2.0 (or later) Installer --

[Setup]
AppName=Open Source Computer Vision Library
AppVerName=Open Source Computer Vision Library 1.1pre1
AppCopyright=Copyright (C) 2000-2006 Intel Corporation
DefaultDirName={pf}\OpenCV
DefaultGroupName=OpenCV
;UninstallDisplayIcon={app}\apps\CamShiftDemo\res\CamShiftDemo.ico
SourceDir=..
Compression=bzip/9
LicenseFile="docs\license.txt"
OutputBaseFilename=OpenCV_1.1pre1a
WizardImageFile=utils/splash.bmp
SetupIconFile=utils/opencv.ico
; uncomment the following line if you want your installation to run on NT 3.51 too.
; MinVersion=4,3.51

[Dirs]

; workspaces
Name: "{app}\_make"
;Name: "{app}\_make\cbuilderx"

; cxcore
Name: "{app}\cxcore"
Name: "{app}\cxcore\include"
Name: "{app}\cxcore\src"

; cv
Name: "{app}\cv"
Name: "{app}\cv\include"
Name: "{app}\cv\src"

; cvaux
Name: "{app}\cvaux"
Name: "{app}\cvaux\include"
Name: "{app}\cvaux\src"
Name: "{app}\cvaux\src\vs"

; ml
Name: "{app}\ml"
Name: "{app}\ml\include"
Name: "{app}\ml\src"


; training data
Name: "{app}\data"
Name: "{app}\data\haarcascades"

; otherlibs
Name: "{app}\otherlibs"
Name: "{app}\otherlibs\_graphics"
Name: "{app}\otherlibs\_graphics\include"
Name: "{app}\otherlibs\_graphics\include\jasper"
Name: "{app}\otherlibs\_graphics\lib"
Name: "{app}\otherlibs\_graphics\src"
Name: "{app}\otherlibs\_graphics\src\libjasper"
Name: "{app}\otherlibs\_graphics\src\libjpeg"
Name: "{app}\otherlibs\_graphics\src\libpng"
Name: "{app}\otherlibs\_graphics\src\libtiff"
Name: "{app}\otherlibs\_graphics\src\zlib"
Name: "{app}\otherlibs\highgui"
Name: "{app}\otherlibs\ffopencv"

; interfaces
Name: "{app}\interfaces"
Name: "{app}\interfaces\ipp"

Name: "{app}\interfaces\swig"
Name: "{app}\interfaces\swig\filtered"
Name: "{app}\interfaces\swig\general"
Name: "{app}\interfaces\swig\python"
Name: "{app}\interfaces\swig\python\build"
Name: "{app}\interfaces\swig\python\build\lib.win32-2.6"
Name: "{app}\interfaces\swig\python\build\lib.win32-2.6\opencv"

; documentation
Name: "{app}\docs"
Name: "{app}\docs\ref"
Name: "{app}\docs\ref\pics"
Name: "{app}\docs\papers"
Name: "{app}\docs\vidsurv"

; sample code
Name: "{app}\samples"
Name: "{app}\samples\c"
Name: "{app}\samples\python"

; batch tests
Name: "{app}\tests"
Name: "{app}\tests\cxts"
Name: "{app}\tests\cxcore"
Name: "{app}\tests\cxcore\src"
Name: "{app}\tests\cv"
Name: "{app}\tests\cv\src"
Name: "{app}\tests\cv\testdata"
Name: "{app}\tests\cv\testdata\cameracalibration"
;Name: "{app}\tests\cv\testdata\gesturerecognition"
Name: "{app}\tests\cv\testdata\optflow"
Name: "{app}\tests\cv\testdata\snakes"
Name: "{app}\tests\python"
Name: "{app}\tests\python\highgui"

; utilities
Name: "{app}\utils"

; applications
Name: "{app}\apps\"
Name: "{app}\apps\HaarTraining"
Name: "{app}\apps\HaarTraining\include"
Name: "{app}\apps\HaarTraining\make"
Name: "{app}\apps\HaarTraining\src"
Name: "{app}\apps\HaarTraining\doc"

; precompiled binaries
Name: "{app}\bin"
Name: "{app}\lib"


[Files]

; root
Source: "README"; DestDir: "{app}"
Source: "ChangeLog"; DestDir: "{app}"
Source: "TODO"; DestDir: "{app}"
Source: "AUTHORS"; DestDir: "{app}"
Source: "THANKS"; DestDir: "{app}"
Source: "INSTALL"; DestDir: "{app}"

; _make
Source: "_make\opencv*.dsw"; DestDir: "{app}\_make"
Source: "_make\opencv*.sln"; DestDir: "{app}\_make"
Source: "_make\makefile.*"; DestDir: "{app}\_make"
Source: "_make\*.mak"; DestDir: "{app}\_make"

; cxcore
Source: "cxcore\include\*.h*"; DestDir: "{app}\cxcore\include"
Source: "cxcore\src\*.c*"; DestDir: "{app}\cxcore\src"
Source: "cxcore\src\*.h*"; DestDir: "{app}\cxcore\src"
Source: "cxcore\src\Makefile.ms"; DestDir: "{app}\cxcore\src"
Source: "cxcore\src\Makefile.gnu"; DestDir: "{app}\cxcore\src"
Source: "cxcore\src\*.rc"; DestDir: "{app}\cxcore\src"
Source: "cxcore\src\*.dsp"; DestDir: "{app}\cxcore\src"
Source: "cxcore\src\*.vcproj"; DestDir: "{app}\cxcore\src"


; cv
Source: "cv\include\*.h*"; DestDir: "{app}\cv\include"
Source: "cv\src\*.c*"; DestDir: "{app}\cv\src"
Source: "cv\src\*.h*"; DestDir: "{app}\cv\src"
Source: "cv\src\Makefile.ms"; DestDir: "{app}\cv\src"
Source: "cv\src\Makefile.gnu"; DestDir: "{app}\cv\src"
Source: "cv\src\*.rc"; DestDir: "{app}\cv\src"
Source: "cv\src\*.dsp"; DestDir: "{app}\cv\src"
Source: "cv\src\*.vcproj"; DestDir: "{app}\cv\src"

; cvaux
Source: "cvaux\include\*.h*"; DestDir: "{app}\cvaux\include"
Source: "cvaux\src\*.c*"; DestDir: "{app}\cvaux\src"
Source: "cvaux\src\vs\*.c*"; DestDir: "{app}\cvaux\src\vs"
Source: "cvaux\src\*.h*"; DestDir: "{app}\cvaux\src"
Source: "cvaux\src\Makefile.ms"; DestDir: "{app}\cvaux\src"
Source: "cvaux\src\Makefile.gnu"; DestDir: "{app}\cvaux\src"
Source: "cvaux\src\*.rc"; DestDir: "{app}\cvaux\src"
Source: "cvaux\src\*.dsp"; DestDir: "{app}\cvaux\src"
Source: "cvaux\src\*.vcproj"; DestDir: "{app}\cvaux\src"

; ml
Source: "ml\include\*.h*"; DestDir: "{app}\ml\include"
Source: "ml\src\*.c*"; DestDir: "{app}\ml\src"
Source: "ml\src\*.h*"; DestDir: "{app}\ml\src"
Source: "ml\src\Makefile.ms"; DestDir: "{app}\ml\src"
Source: "ml\src\Makefile.gnu"; DestDir: "{app}\ml\src"
Source: "ml\src\*.rc"; DestDir: "{app}\ml\src"
Source: "ml\src\*.dsp"; DestDir: "{app}\ml\src"
Source: "ml\src\*.vcproj"; DestDir: "{app}\ml\src"

Source: "data\*.txt"; DestDir: "{app}\data"
Source: "data\haarcascades\*.xml"; DestDir: "{app}\data\haarcascades"

; graphic libraries
Source: "otherlibs\_graphics\include\*.h"; DestDir: "{app}\otherlibs\_graphics\include"
Source: "otherlibs\_graphics\include\jasper\*.h"; DestDir: "{app}\otherlibs\_graphics\include\jasper"
Source: "otherlibs\_graphics\lib\*.a"; DestDir: "{app}\otherlibs\_graphics\lib"
Source: "otherlibs\_graphics\lib\*.lib"; DestDir: "{app}\otherlibs\_graphics\lib"
Source: "otherlibs\_graphics\readme.txt"; DestDir: "{app}\otherlibs\_graphics"

Source: "otherlibs\_graphics\src\libjasper\license"; DestDir: "{app}\otherlibs\_graphics\src\libjasper"
Source: "otherlibs\_graphics\src\libjasper\*.c"; DestDir: "{app}\otherlibs\_graphics\src\libjasper"
Source: "otherlibs\_graphics\src\libjasper\*.h"; DestDir: "{app}\otherlibs\_graphics\src\libjasper"
Source: "otherlibs\_graphics\src\libjasper\*.dsp"; DestDir: "{app}\otherlibs\_graphics\src\libjasper"
Source: "otherlibs\_graphics\src\libjasper\*.vcproj"; DestDir: "{app}\otherlibs\_graphics\src\libjasper"
Source: "otherlibs\_graphics\src\libjasper\readme"; DestDir: "{app}\otherlibs\_graphics\src\libjasper"

Source: "otherlibs\_graphics\src\libjpeg\*.c"; DestDir: "{app}\otherlibs\_graphics\src\libjpeg"
Source: "otherlibs\_graphics\src\libjpeg\*.h"; DestDir: "{app}\otherlibs\_graphics\src\libjpeg"
Source: "otherlibs\_graphics\src\libjpeg\*.dsp"; DestDir: "{app}\otherlibs\_graphics\src\libjpeg"
Source: "otherlibs\_graphics\src\libjpeg\*.vcproj"; DestDir: "{app}\otherlibs\_graphics\src\libjpeg"
Source: "otherlibs\_graphics\src\libjpeg\makefile.*"; DestDir: "{app}\otherlibs\_graphics\src\libjpeg"
Source: "otherlibs\_graphics\src\libjpeg\readme"; DestDir: "{app}\otherlibs\_graphics\src\libjpeg"

Source: "otherlibs\_graphics\src\libpng\*.c"; DestDir: "{app}\otherlibs\_graphics\src\libpng"
Source: "otherlibs\_graphics\src\libpng\*.dsp"; DestDir: "{app}\otherlibs\_graphics\src\libpng"
Source: "otherlibs\_graphics\src\libpng\*.vcproj"; DestDir: "{app}\otherlibs\_graphics\src\libpng"
Source: "otherlibs\_graphics\src\libpng\readme"; DestDir: "{app}\otherlibs\_graphics\src\libpng"

Source: "otherlibs\_graphics\src\libtiff\*.c*"; DestDir: "{app}\otherlibs\_graphics\src\libtiff"
Source: "otherlibs\_graphics\src\libtiff\*.h*"; DestDir: "{app}\otherlibs\_graphics\src\libtiff"
Source: "otherlibs\_graphics\src\libtiff\*.dsp"; DestDir: "{app}\otherlibs\_graphics\src\libtiff"
Source: "otherlibs\_graphics\src\libtiff\*.vcproj"; DestDir: "{app}\otherlibs\_graphics\src\libtiff"
Source: "otherlibs\_graphics\src\libtiff\*.def"; DestDir: "{app}\otherlibs\_graphics\src\libtiff"
Source: "otherlibs\_graphics\src\libtiff\Makefile.*"; DestDir: "{app}\otherlibs\_graphics\src\libtiff"

Source: "otherlibs\_graphics\src\zlib\*.c"; DestDir: "{app}\otherlibs\_graphics\src\zlib"
Source: "otherlibs\_graphics\src\zlib\*.h"; DestDir: "{app}\otherlibs\_graphics\src\zlib"
Source: "otherlibs\_graphics\src\zlib\*.dsp"; DestDir: "{app}\otherlibs\_graphics\src\zlib"
Source: "otherlibs\_graphics\src\zlib\*.vcproj"; DestDir: "{app}\otherlibs\_graphics\src\zlib"
Source: "otherlibs\_graphics\src\zlib\readme"; DestDir: "{app}\otherlibs\_graphics\src\zlib"

; otherlibs: highgui
Source: "otherlibs\highgui\*.c*"; DestDir: "{app}\otherlibs\highgui"
Source: "otherlibs\highgui\*.h*"; DestDir: "{app}\otherlibs\highgui"
Source: "otherlibs\highgui\*.rc"; DestDir: "{app}\otherlibs\highgui"
Source: "otherlibs\highgui\*.dsp"; DestDir: "{app}\otherlibs\highgui"
Source: "otherlibs\highgui\*.vcproj"; DestDir: "{app}\otherlibs\highgui"
Source: "otherlibs\highgui\Makefile.ms"; DestDir: "{app}\otherlibs\highgui"
Source: "otherlibs\highgui\Makefile.gnu"; DestDir: "{app}\otherlibs\highgui"
Source: "otherlibs\highgui\*.sh"; DestDir: "{app}\otherlibs\highgui"

; otherlibs: ffopencv
Source: "otherlibs\ffopencv\*.c*"; DestDir: "{app}\otherlibs\ffopencv"
Source: "otherlibs\ffopencv\*.h*"; DestDir: "{app}\otherlibs\ffopencv"
Source: "otherlibs\ffopencv\*.ds*"; DestDir: "{app}\otherlibs\ffopencv"
Source: "otherlibs\ffopencv\*.vcproj"; DestDir: "{app}\otherlibs\ffopencv"
Source: "otherlibs\ffopencv\*.sln"; DestDir: "{app}\otherlibs\ffopencv"

; interfaces
Source: "interfaces\ipp\*.c"; DestDir: "{app}\interfaces\ipp"
Source: "interfaces\ipp\*.h"; DestDir: "{app}\interfaces\ipp"
Source: "interfaces\ipp\*.def"; DestDir: "{app}\interfaces\ipp"
Source: "interfaces\ipp\*.ds*"; DestDir: "{app}\interfaces\ipp"
Source: "interfaces\ipp\*.py"; DestDir: "{app}\interfaces\ipp"
Source: "interfaces\ipp\*.txt"; DestDir: "{app}\interfaces\ipp"

Source: "interfaces\swig\filtered\*.h"; DestDir: "{app}\interfaces\swig\filtered"
Source: "interfaces\swig\general\*.i"; DestDir: "{app}\interfaces\swig\general"
Source: "interfaces\swig\python\*.py"; DestDir: "{app}\interfaces\swig\python"
Source: "interfaces\swig\python\*.i"; DestDir: "{app}\interfaces\swig\python"
Source: "interfaces\swig\python\*.c*"; DestDir: "{app}\interfaces\swig\python"
Source: "interfaces\swig\python\*.h*"; DestDir: "{app}\interfaces\swig\python"
Source: "interfaces\swig\python\build\lib.win32-2.6\opencv\*.py*"; DestDir: "{app}\interfaces\swig\python\build\lib.win32-2.6\opencv"

; documentation
Source: "docs\*.htm*"; DestDir: "{app}\docs"
Source: "docs\*.jp*"; DestDir: "{app}\docs"
Source: "docs\*.png"; DestDir: "{app}\docs"
Source: "docs\*.txt"; DestDir: "{app}\docs"
Source: "docs\*.pdf"; DestDir: "{app}\docs"
;Source: "docs\*.rtf"; DestDir: "{app}\docs"
Source: "docs\ref\*.htm*"; DestDir: "{app}\docs\ref"
Source: "docs\ref\*.css"; DestDir: "{app}\docs\ref"
Source: "docs\ref\pics\*.jp*"; DestDir: "{app}\docs\ref\pics"
Source: "docs\ref\pics\*.png"; DestDir: "{app}\docs\ref\pics"
Source: "docs\papers\*.pdf"; DestDir: "{app}\docs\papers"
Source: "docs\papers\*.ps"; DestDir: "{app}\docs\papers"
Source: "docs\vidsurv\*.doc"; DestDir: "{app}\docs\vidsurv"

; sample code
Source: "samples\c\*.c*"; DestDir: "{app}\samples\c"
Source: "samples\c\*.sh"; DestDir: "{app}\samples\c"
Source: "samples\c\*.jp*"; DestDir: "{app}\samples\c"
Source: "samples\c\*.txt"; DestDir: "{app}\samples\c"
Source: "samples\c\*.png"; DestDir: "{app}\samples\c"
Source: "samples\c\*.dsp"; DestDir: "{app}\samples\c"
Source: "samples\c\*.vcproj"; DestDir: "{app}\samples\c"
Source: "samples\c\Makefile.ms"; DestDir: "{app}\samples\c"
Source: "samples\c\Makefile.gnu"; DestDir: "{app}\samples\c"
Source: "samples\c\*.exe"; DestDir: "{app}\samples\c"
Source: "samples\c\*.data"; DestDir: "{app}\samples\c"

Source: "samples\python\*.py"; DestDir: "{app}\samples\python"

; batch tests
Source: "tests\cv\src\*.c*"; DestDir: "{app}\tests\cv\src"
Source: "tests\cv\src\*.h*"; DestDir: "{app}\tests\cv\src"
Source: "tests\cv\src\Makefile.ms"; DestDir: "{app}\tests\cv\src"
Source: "tests\cv\src\Makefile.gnu"; DestDir: "{app}\tests\cv\src"
Source: "tests\cv\src\*.dsp"; DestDir: "{app}\tests\cv\src"
Source: "tests\cv\src\*.vcproj"; DestDir: "{app}\tests\cv\src"
Source: "tests\cv\src\*.inc"; DestDir: "{app}\tests\cv\src"
Source: "tests\cv\testdata\cameracalibration\*.*"; DestDir: "{app}\tests\cv\testdata\cameracalibration"
;Source: "tests\cv\testdata\gesturerecognition\*.*"; DestDir: "{app}\tests\cv\testdata\gesturerecognition"
Source: "tests\cv\testdata\optflow\*.*"; DestDir: "{app}\tests\cv\testdata\optflow"
Source: "tests\cv\testdata\snakes\*.*"; DestDir: "{app}\tests\cv\testdata\snakes"
Source: "tests\cxts\*.c*"; DestDir: "{app}\tests\cxts"
Source: "tests\cxts\*.h*"; DestDir: "{app}\tests\cxts"
Source: "tests\cxts\*.dsp"; DestDir: "{app}\tests\cxts"
Source: "tests\cxts\*.vcproj"; DestDir: "{app}\tests\cxts"
Source: "tests\cxts\Makefile.ms"; DestDir: "{app}\tests\cxts"
Source: "tests\cxts\Makefile.gnu"; DestDir: "{app}\tests\cxts"
Source: "tests\cxcore\src\*.c*"; DestDir: "{app}\tests\cxcore\src"
Source: "tests\cxcore\src\*.h*"; DestDir: "{app}\tests\cxcore\src"
Source: "tests\cxcore\src\*.dsp"; DestDir: "{app}\tests\cxcore\src"
Source: "tests\cxcore\src\*.vcproj"; DestDir: "{app}\tests\cxcore\src"
Source: "tests\cxcore\src\Makefile.ms"; DestDir: "{app}\tests\cxcore\src"
Source: "tests\cxcore\src\Makefile.gnu"; DestDir: "{app}\tests\cxcore\src"
Source: "tests\python\highgui\*.py"; DestDir: "{app}\tests\python\highgui"

; utilities
Source: "utils\*.cmd"; DestDir: "{app}\utils"
Source: "utils\*.py"; DestDir: "{app}\utils"
Source: "utils\*.iss"; DestDir: "{app}\utils"
Source: "utils\*.bmp"; DestDir: "{app}\utils"
Source: "utils\*.ico"; DestDir: "{app}\utils"

; applications
Source: "apps\HaarTraining\*.*"; DestDir: "{app}\apps\HaarTraining"
Source: "apps\HaarTraining\include\*.h*"; DestDir: "{app}\apps\HaarTraining\include"
Source: "apps\HaarTraining\make\*.ds*"; DestDir: "{app}\apps\HaarTraining\make"
Source: "apps\HaarTraining\make\*.vcproj"; DestDir: "{app}\apps\HaarTraining\make"
Source: "apps\HaarTraining\make\*.sln"; DestDir: "{app}\apps\HaarTraining\make"
Source: "apps\HaarTraining\src\*.c*"; DestDir: "{app}\apps\HaarTraining\src"
Source: "apps\HaarTraining\src\*.h*"; DestDir: "{app}\apps\HaarTraining\src"
Source: "apps\HaarTraining\src\*.h*"; DestDir: "{app}\apps\HaarTraining\src"
Source: "apps\HaarTraining\doc\*.htm*"; DestDir: "{app}\apps\HaarTraining\doc"

; precompiled binaries
Source: "C:\Program Files\Microsoft Visual Studio 8\VC\redist\x86\Microsoft.VC80.OPENMP\vcomp.dll"; DestDir: "{app}\bin"
Source: "C:\Program Files\Microsoft Visual Studio 8\VC\redist\x86\Microsoft.VC80.CRT\msvcp80.dll"; DestDir: "{app}\bin"
Source: "C:\Program Files\Microsoft Visual Studio 8\VC\redist\x86\Microsoft.VC80.CRT\msvcr80.dll"; DestDir: "{app}\bin"
Source: "bin\cvtest.exe"; DestDir: "{app}\bin"
Source: "bin\cxcoretest.exe"; DestDir: "{app}\bin"
Source: "bin\cxcore110.dll"; DestDir: "{app}\bin"
Source: "bin\cxcore110.pdb"; DestDir: "{app}\bin"
Source: "bin\cv110.dll"; DestDir: "{app}\bin"
Source: "bin\cv110.pdb"; DestDir: "{app}\bin"
Source: "bin\highgui110.dll"; DestDir: "{app}\bin"
Source: "bin\highgui110.pdb"; DestDir: "{app}\bin"
Source: "bin\cvaux110.dll"; DestDir: "{app}\bin"
Source: "bin\cvaux110.pdb"; DestDir: "{app}\bin"
Source: "bin\ml110.dll"; DestDir: "{app}\bin"
Source: "bin\ml110.pdb"; DestDir: "{app}\bin"
Source: "bin\ffopencv110.dll"; DestDir: "{app}\bin"
Source: "bin\ffopencv110.pdb"; DestDir: "{app}\bin"
Source: "bin\cxts001.dll"; DestDir: "{app}\bin"
Source: "bin\haartraining.exe"; DestDir: "{app}\bin"
Source: "bin\createsamples.exe"; DestDir: "{app}\bin"
Source: "bin\performance.exe"; DestDir: "{app}\bin"

; import libraries
Source: "lib\cxcore.lib"; DestDir: "{app}\lib"
Source: "lib\cv.lib"; DestDir: "{app}\lib"
Source: "lib\highgui.lib"; DestDir: "{app}\lib"
Source: "lib\cvaux.lib"; DestDir: "{app}\lib"
Source: "lib\ml.lib"; DestDir: "{app}\lib"
Source: "lib\cvhaartraining.lib"; DestDir: "{app}\lib"
Source: "lib\cxts.lib"; DestDir: "{app}\lib"

[Icons]
Name: "{group}\OpenCV Workspace MSVC6"; Filename: "{app}\_make\opencv.dsw"
Name: "{group}\OpenCV Workspace .NET 2005,2008"; Filename: "{app}\_make\opencv.vs2005.sln"
Name: "{group}\OpenCV Workspace .NET 2005,2008 (Express or Standard Edition)"; Filename: "{app}\_make\opencv.vs2005.no_openmp.sln"
Name: "{group}\Documentation"; Filename: "{app}\docs\index.htm"
Name: "{group}\Samples"; Filename: "{app}\samples\c\"

[Tasks]
Name: add_opencv_path; Description: "Add <...>\OpenCV\bin to the system PATH"; Flags: checkedonce

[Registry]
; Start "Software\My Company\My Program" keys under HKEY_CURRENT_USER
; and HKEY_LOCAL_MACHINE. The flags tell it to always delete the
; "My Program" keys upon uninstall, and delete the "My Company" keys
; if there is nothing left in them.
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "Path"; ValueData: "{app}\bin;{olddata}"; Flags: createvalueifdoesntexist; Tasks: add_opencv_path
Root: HKCU; Subkey: "Software\OpenCV"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\OpenCV\Settings"; ValueType: string; ValueName: "Path"; ValueData: "{app}"

[Run]
Filename: "{reg:HKLM\Software\Python\PythonCore\2.6\InstallPath,|C:\Python26\}python.exe"; Parameters: "setup-for-win.py install"; WorkingDir: "{app}\interfaces\swig\python"; Flags: skipifdoesntexist; StatusMsg: "Installing OpenCV Module for Python..."
Filename: "{app}\docs\index.htm"; Description: "View Documentation"; Flags: postinstall shellexec

[UninstallRun]


