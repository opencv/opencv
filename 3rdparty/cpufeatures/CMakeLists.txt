
ocv_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB cpuf_s *.c)
file(GLOB cpuf_h *.h)

set(lib_srcs  ${cpuf_s})
set(lib_hdrs  ${cpuf_h})

set(CPUFEATURES_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
set(CPUFEATURES_LIBRARIES cpufeatures  CACHE INTERNAL "")

add_library(cpufeatures STATIC ${lib_srcs} ${lib_hdrs})

set_target_properties(cpufeatures
  PROPERTIES OUTPUT_NAME cpufeatures
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME cpufeatures
  COMPILE_PDB_NAME_DEBUG "cpufeatures${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )
  
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(cpufeatures PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(cpufeatures EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

