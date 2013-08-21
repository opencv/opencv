mkdir build_winrt_arm
cd build_winrt_arm

set msvc_path=C:\Program Files\Microsoft Visual Studio 11.0

call "%msvc_path%\Common7\Tools\VsDevCmd.bat"
call "%msvc_path%\VC\bin\x86_arm\vcvarsx86_arm.bat"

cmake.exe -GNinja -DCMAKE_BUILD_TYPE=Release -DENABLE_WINRT_MODE=ON -DWITH_FFMPEG=OFF -DWITH_MSMF=OFF -DWITH_DSHOW=OFF -DWITH_VFW=OFF -DWITH_TIFF=OFF -DWITH_OPENEXR=OFF -DWITH_CUDA=OFF -DBUILD_opencv_gpu=OFF -DBUILD_opencv_python=OFF -DBUILD_opencv_java=OFF -DCMAKE_TOOLCHAIN_FILE=..\winrt\arm.winrt.toolchain.cmake  %* ..\..
