# ----------------------------------------------------------------------------
#  CMake file for IPP IW. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${IPP_IW_LIBRARY})

ocv_include_directories(${IPP_INCLUDE_DIRS} ${IPP_IW_PATH}/include)
add_definitions(-DIW_BUILD)
if(HAVE_IPP_ICV)
  add_definitions(-DICV_BASE)
endif()

file(GLOB lib_srcs ${IPP_IW_PATH}/src/*.c ${IPP_IW_PATH}/src/*.cpp)
file(GLOB lib_hdrs ${IPP_IW_PATH}/include/*.h ${IPP_IW_PATH}/include/iw/*.h ${IPP_IW_PATH}/include/iw++/*.hpp)

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_library(${IPP_IW_LIBRARY} STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${lib_srcs} ${lib_hdrs})

if(UNIX)
  if(CV_GCC OR CV_CLANG OR CV_ICC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function -Wno-missing-braces -Wno-missing-field-initializers")
  endif()
  if(CV_CLANG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-self-assign -Wno-strict-prototypes")
  endif()
endif()

set_target_properties(${IPP_IW_LIBRARY}
  PROPERTIES OUTPUT_NAME ${IPP_IW_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${IPP_IW_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${IPP_IW_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${IPP_IW_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${IPP_IW_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()
