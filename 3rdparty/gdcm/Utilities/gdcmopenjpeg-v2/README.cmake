Basic instructions on how to build using CMake (CMake 2.4.5 or newer is required)

  svn co http://www.openjpeg.org/svn/trunk
  cd trunk
  mkdir bin
  cd bin
  cmake .. -DBUILD_EXAMPLES:BOOL=ON
  make
  ./bin/j2k_to_image
