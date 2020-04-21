if(WINCE)
  # CommCtrl.lib does not exist in headless WINCE Adding this will make CMake
  # Try_Compile succeed and therefore also C/C++ ABI Detetection work
  # https://gitlab.kitware.com/cmake/cmake/blob/master/Modules/Platform/Windows-
  # MSVC.cmake
  set(CMAKE_C_STANDARD_LIBRARIES_INIT "coredll.lib oldnames.lib")
  set(CMAKE_CXX_STANDARD_LIBRARIES_INIT ${CMAKE_C_STANDARD_LIBRARIES_INIT})
  foreach(ID EXE SHARED MODULE)
    string(APPEND CMAKE_${ID}_LINKER_FLAGS_INIT
           " /NODEFAULTLIB:libc.lib")
  endforeach()
endif()
