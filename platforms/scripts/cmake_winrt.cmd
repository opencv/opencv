mkdir build_winrt_arm
cd build_winrt_arm

rem call "C:\Program Files\Microsoft Visual Studio 11.0\VC\bin\x86_arm\vcvarsx86_arm.bat"

cmake.exe -GNinja -DWITH_TBB=ON -DBUILD_TBB=ON -DCMAKE_BUILD_TYPE=Release -DWITH_FFMPEG=OFF -DBUILD_opencv_gpu=OFF -DBUILD_opencv_python=OFF -DCMAKE_TOOLCHAIN_FILE=..\winrt\arm.winrt.toolchain.cmake ..\..
