The build script is to be fixed.
Right now it assumes that 32-bit MinGW is in the system path and
64-bit mingw is installed to c:\Apps\MinGW64.

It is important that gcc is used, not g++!
Otherwise the produced DLL will likely be dependent on libgcc_s_dw2-1.dll or similar DLL.
While we want to make the DLLs with minimum dependencies: Win32 libraries + msvcrt.dll.

ffopencv.c is really a C++ source, hence -x c++ is used.

How to update opencv_ffmpeg.dll and opencv_ffmpeg_64.dll when a new version of FFMPEG is release?

1. Install 32-bit MinGW + MSYS from
   http://sourceforge.net/projects/mingw/files/Automated%20MinGW%20Installer/mingw-get-inst/
   Let's assume, it's installed in C:\MSYS32.
2. Install 64-bit MinGW. http://mingw-w64.sourceforge.net/
   Let's assume, it's installed in C:\MSYS64
3. Copy C:\MSYS32\msys to C:\MSYS64\msys. Edit C:\MSYS64\msys\etc\fstab, change C:\MSYS32 to C:\MSYS64.

4. Now you have working MSYS32 and MSYS64 environments.
   Launch, one by one, C:\MSYS32\msys\msys.bat and C:\MSYS64\msys\msys.bat to create your home directories.

4. Download ffmpeg-x.y.z.tar.gz (where x.y.z denotes the actual ffmpeg version).
   Copy it to C:\MSYS{32|64}\msys\home\<loginname> directory.

5. To build 32-bit ffmpeg libraries, run C:\MSYS32\msys\msys.bat and type the following commands:

   5.1. tar -xzf ffmpeg-x.y.z.tar.gz
   5.2. mkdir build
   5.3. cd build
   5.4. ../ffmpeg-x.y.z/configure --enable-w32threads
   5.5. make
   5.6. make install
   5.7. cd /local/lib
   5.8. strip -g *.a

6. Then repeat the same for 64-bit case. The output libs: libavcodec.a etc. need to be renamed to libavcodec64.a etc.

7. Then, copy all those libs to <opencv>\3rdparty\lib\, copy the headers to <opencv>\3rdparty\include\ffmpeg_.

8. Then, go to <opencv>\3rdparty\ffmpeg, edit make.bat
   (change paths to the actual paths to your msys32 and msys64 distributions) and then run make.bat
