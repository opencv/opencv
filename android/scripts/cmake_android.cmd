cd %~dp0\..

::rmdir /S /Q build
mkdir build 2>null

SET ANDROID_NDK=C:\apps\android-ndk-r5b
SET CMAKE_EXE=C:\apps\cmake\bin\cmake.exe
SET MAKE_EXE=C:\apps\gnuport\make.exe

cd build
%CMAKE_EXE% -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=..\android.toolchain.cmake -DCMAKE_MAKE_PROGRAM=%MAKE_EXE% ..\..
