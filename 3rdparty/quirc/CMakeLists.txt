project(quirc)

set(CURR_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")

set_property(GLOBAL PROPERTY QUIRC_INCLUDE_DIR ${CURR_INCLUDE_DIR})
ocv_include_directories(${CURR_INCLUDE_DIR})

file(GLOB_RECURSE quirc_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "include/*.h")
file(GLOB_RECURSE quirc_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "src/*.c")

add_library(${PROJECT_NAME} STATIC ${quirc_headers} ${quirc_sources})
ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-variable -Wshadow)

set_target_properties(${PROJECT_NAME}
  PROPERTIES OUTPUT_NAME ${PROJECT_NAME}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${PROJECT_NAME}
  COMPILE_PDB_NAME_DEBUG "${PROJECT_NAME}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${PROJECT_NAME} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ocv_install_3rdparty_licenses(${PROJECT_NAME} LICENSE)
