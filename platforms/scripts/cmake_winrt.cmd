mkdir build
cd build

rem call "C:\Program Files\Microsoft Visual Studio 11.0\VC\bin\x86_arm\vcvarsx86_arm.bat"

cmake.exe -GNinja -DCMAKE_BUILD_TYPE=Release -DWITH_FFMPEG=OFF -DBUILD_opencv_gpu=OFF -DBUILD_opencv_python=OFF -DCMAKE_TOOLCHAIN_FILE=..\..\winrt\arm.winrt.toolchain.cmake ..\..\..
