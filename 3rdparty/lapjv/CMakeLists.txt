project(lapjv)

add_library(lapjv STATIC lapjv.cpp)
target_include_directories(lapjv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lapjv)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wmissing-declarations -Wunused-parameter)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(lapjv PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(lapjv EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()

ocv_install_3rdparty_licenses(lapjv LICENSE)
