The build script is to be fixed.
Right now it assumes that 32-bit MinGW is in the system path and
64-bit mingw is installed to c:\Apps\MinGW64.

It is important that gcc is used, not g++!
Otherwise the produced DLL will likely be dependent on libgcc_s_dw2-1.dll or similar DLL.
While we want to make the DLLs with minimum dependencies: Win32 libraries + msvcrt.dll.

ffopencv.c is really a C++ source, hence -x c++ is used.
