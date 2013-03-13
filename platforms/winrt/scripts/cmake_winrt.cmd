mkdir build
cd build

rem call "C:\Program Files\Microsoft Visual Studio 11.0\VC\bin\x86_arm\vcvarsx86_arm.bat"

SET PATH=C:\Program Files\Ninja;%PATH%

"C:\Program Files\CMake 2.8\bin\cmake.exe" -GNinja -DCMAKE_BUILD_TYPE=Release -DWITH_TIFF=OFF -DWITH_FFMPEG=OFF -DBUILD_opencv_gpu=OFF -DENABLE_SSE=OFF -DENABLE_SSE2=OFF ..\..\..