# This file is included from a subdirectory
set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")

find_package(HDF5)
include_directories(${HDF5_INCLUDE_DIRS})

ocv_add_module(${MODULE_NAME} BINDINGS PRIVATE_REQUIRED opencv_python_bindings_generator)

ocv_module_include_directories(
    "${${PYTHON}_INCLUDE_PATH}"
)
include_directories(
    ${${PYTHON}_NUMPY_INCLUDE_DIRS}
    "${PYTHON_SOURCE_DIR}/src2"
    "${OPENCV_PYTHON_BINDINGS_DIR}"
)

# try to use dynamic symbols linking with libpython.so
set(OPENCV_FORCE_PYTHON_LIBS OFF CACHE BOOL "")
string(REPLACE "-Wl,--no-undefined" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
if(NOT WIN32 AND NOT APPLE AND NOT OPENCV_PYTHON_SKIP_LINKER_EXCLUDE_LIBS)
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--exclude-libs=ALL")
endif()

ocv_add_library(${the_module} MODULE ${PYTHON_SOURCE_DIR}/src2/cv2.cpp ${cv2_generated_hdrs} ${opencv_userdef_hdrs} ${cv2_custom_hdr})
add_dependencies(${the_module} gen_opencv_python_source)

if(APPLE)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
elseif(WIN32 OR OPENCV_FORCE_PYTHON_LIBS)
  if(${PYTHON}_DEBUG_LIBRARIES AND NOT ${PYTHON}_LIBRARIES MATCHES "optimized.*debug")
    ocv_target_link_libraries(${the_module} LINK_PRIVATE debug ${${PYTHON}_DEBUG_LIBRARIES} optimized ${${PYTHON}_LIBRARIES})
  else()
    ocv_target_link_libraries(${the_module} LINK_PRIVATE ${${PYTHON}_LIBRARIES})
  endif()
endif()

set(deps ${OPENCV_MODULE_${the_module}_DEPS})
list(REMOVE_ITEM deps opencv_python_bindings_generator) # don't add dummy module to target_link_libraries list
ocv_target_link_libraries(${the_module} LINK_PRIVATE ${deps})

if(DEFINED ${PYTHON}_CVPY_SUFFIX)
  set(CVPY_SUFFIX "${${PYTHON}_CVPY_SUFFIX}")
else()
  execute_process(COMMAND ${${PYTHON}_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SO'))"
                  RESULT_VARIABLE PYTHON_CVPY_PROCESS
                  OUTPUT_VARIABLE CVPY_SUFFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT PYTHON_CVPY_PROCESS EQUAL 0)
    set(CVPY_SUFFIX ".so")
  endif()
endif()

set_target_properties(${the_module} PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY  "${LIBRARY_OUTPUT_PATH}/${MODULE_INSTALL_SUBDIR}"
                      ARCHIVE_OUTPUT_NAME ${the_module}  # prevent name conflict for python2/3 outputs
                      DEFINE_SYMBOL CVAPI_EXPORTS
                      PREFIX ""
                      OUTPUT_NAME cv2
                      SUFFIX ${CVPY_SUFFIX})

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_module} PROPERTIES FOLDER "bindings")
endif()

if(MSVC)
  add_definitions(-DCVAPI_EXPORTS)
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
endif()

if(MSVC AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100") #unreferenced formal parameter
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4127") #conditional expression is constant
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4505") #unreferenced local function has been removed
  string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Woverloaded-virtual -Wunused-private-field)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef) # accurate guard via #pragma doesn't work (C++ preprocessor doesn't handle #pragma)

if(MSVC AND NOT BUILD_SHARED_LIBS)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
endif()

if(MSVC AND NOT ${PYTHON}_DEBUG_LIBRARIES)
  set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
else()
  set(PYTHON_INSTALL_CONFIGURATIONS "")
endif()

if(WIN32)
  set(PYTHON_INSTALL_ARCHIVE "")
else()
  set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${${PYTHON}_PACKAGES_PATH} COMPONENT python)
endif()

if(NOT INSTALL_CREATE_DISTRIB AND DEFINED ${PYTHON}_PACKAGES_PATH)
  set(__dst "${${PYTHON}_PACKAGES_PATH}")
endif()
if(NOT __dst)
  if(DEFINED ${PYTHON}_VERSION_MAJOR)
    set(__ver "${${PYTHON}_VERSION_MAJOR}.${${PYTHON}_VERSION_MINOR}")
  elseif(DEFINED ${PYTHON}_VERSION_STRING)
    set(__ver "${${PYTHON}_VERSION_STRING}")
  else()
    set(__ver "unknown")
  endif()
  if(INSTALL_CREATE_DISTRIB)
    set(__dst "python/${__ver}/${OpenCV_ARCH}")
  else()
    set(__dst "python/${__ver}")
  endif()
endif()

install(TARGETS ${the_module}
        ${PYTHON_INSTALL_CONFIGURATIONS}
        RUNTIME DESTINATION "${__dst}" COMPONENT python
        LIBRARY DESTINATION "${__dst}" COMPONENT python
        ${PYTHON_INSTALL_ARCHIVE}
        )

unset(PYTHON_SRC_DIR)
unset(PYTHON_CVPY_PROCESS)
unset(CVPY_SUFFIX)
unset(PYTHON_INSTALL_CONFIGURATIONS)
unset(PYTHON_INSTALL_ARCHIVE)
