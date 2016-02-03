@rem
@rem  Program: GDCM (Grassroots DICOM). A DICOM library
@rem
@rem  Copyright (c) 2006-2011 Mathieu Malaterre
@rem  All rights reserved.
@rem  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.
@rem
@rem     This software is distributed WITHOUT ANY WARRANTY; without even
@rem     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
@rem     PURPOSE.  See the above copyright notice for more information.
@rem

@rem generate GDCM release on Windows

@rem get tmpdir:
set TMPDIR=%TMP%\gdcm_release

set major=2
set minor=2
set patch=5
set version="%major%.%minor%.%patch%"

@rem use VCExpress 2008 for compatibilities with OpenSSL binaries
call "%VS90COMNTOOLS%vsvars32.bat"

@rem User32.lib and al.
SET LIB=C:\Program Files\Microsoft SDKs\Windows\v6.0A\Lib;%LIB%
SET INCLUDE=C:\Program Files\Microsoft SDKs\Windows\v6.0A\Include;%INCLUDE%

@rem GDCM deps:

@rem IF "%ProgramFiles(x86)%"=="" (
SET PATH=%PATH%;%ProgramFiles%\Git\bin
SET PATH=%PATH%;%ProgramFiles%\Swig\swigwin-2.0.7
SET PATH=%PATH%;%ProgramFiles%\Java\jdk1.6.0_25\bin
@rem ) ELSE (
@rem SET PATH=%PATH%;%ProgramFiles(x86)%\Git\bin
@rem SET PATH=%PATH%;%ProgramFiles(x86)%\Swig\swigwin-2.0.8
@rem SET PATH=%PATH%;%ProgramFiles(x86)%\Java\jdk1.6.0_34\bin
@rem )
ECHO %PATH%
PAUSE
@rem needed to get RC.EXE:
SET PATH=C:\Program Files\Microsoft SDKs\Windows\v6.0A\bin;%PATH%

@rem prepare target dir
mkdir %TMPDIR%
mkdir %TMPDIR%\gdcm-build

copy config.win32 %TMPDIR%\gdcm-build\CMakeCache.txt

c:
cd %TMPDIR%
@rem git is itselft a batch:
call git clone --branch release git://git.code.sf.net/p/gdcm/gdcm > git.log 2>&1
cd gdcm
call git checkout "v%version%"
cd ..

cd %TMPDIR%\gdcm-build
cmake -G "NMake Makefiles" ..\gdcm > config.log 2>&1

@rem build gdcm
nmake > nmake.log 2>&1

@rem create NSIS installer
cpack -G NSIS > nsis.log 2>&1

@rem create binary zip
cpack -G ZIP > zip.log 2>&1

@rem create source zip
cpack -G ZIP --config CPackSourceConfig.cmake szip.log 2>&1
