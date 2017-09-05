# This file is included from a subdirectory
set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")

# try to use dynamic symbols linking with libpython.so
set(OPENCV_FORCE_PYTHON_LIBS OFF CACHE BOOL "")
string(REPLACE "-Wl,--no-undefined" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")

ocv_add_module(${MODULE_NAME} BINDINGS)

ocv_module_include_directories(
    "${${PYTHON}_INCLUDE_PATH}"
    ${${PYTHON}_NUMPY_INCLUDE_DIRS}
    "${PYTHON_SOURCE_DIR}/src2"
    )

# get list of modules to wrap
# message(STATUS "Wrapped in ${MODULE_NAME}:")
set(OPENCV_PYTHON_MODULES)
foreach(m ${OPENCV_MODULES_BUILD})
  if (";${OPENCV_MODULE_${m}_WRAPPERS};" MATCHES ";${MODULE_NAME};" AND HAVE_${m})
    list(APPEND OPENCV_PYTHON_MODULES ${m})
    # message(STATUS "\t${m}")
  endif()
endforeach()

set(opencv_hdrs "")
set(opencv_userdef_hdrs "")
foreach(m ${OPENCV_PYTHON_MODULES})
  list(APPEND opencv_hdrs ${OPENCV_MODULE_${m}_HEADERS})
  file(GLOB userdef_hdrs ${OPENCV_MODULE_${m}_LOCATION}/misc/python/pyopencv*.hpp)
  list(APPEND opencv_userdef_hdrs ${userdef_hdrs})
endforeach(m)

# header blacklist
ocv_list_filterout(opencv_hdrs "modules/.*\\\\.h$")
ocv_list_filterout(opencv_hdrs "modules/core/.*/cuda")
ocv_list_filterout(opencv_hdrs "modules/cuda.*")
ocv_list_filterout(opencv_hdrs "modules/cudev")
ocv_list_filterout(opencv_hdrs "modules/core/.*/hal/")
ocv_list_filterout(opencv_hdrs "modules/.+/utils/.*")
ocv_list_filterout(opencv_hdrs "modules/.*\\\\.inl\\\\.h*")
ocv_list_filterout(opencv_hdrs "modules/.*_inl\\\\.h*")
ocv_list_filterout(opencv_hdrs "modules/.*\\\\.details\\\\.h*")
ocv_list_filterout(opencv_hdrs "modules/.*/detection_based_tracker\\\\.hpp") # Conditional compilation

set(cv2_generated_hdrs
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_include.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_funcs.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_types.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_type_reg.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_ns_reg.h"
)

set(OPENCV_${PYTHON}_SIGNATURES_FILE "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_signatures.json" CACHE INTERNAL "")

set(cv2_generated_files ${cv2_generated_hdrs}
    "${OPENCV_${PYTHON}_SIGNATURES_FILE}"
)

string(REPLACE ";" "\n" opencv_hdrs_ "${opencv_hdrs}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/headers.txt" "${opencv_hdrs_}")
add_custom_command(
    OUTPUT ${cv2_generated_files}
    COMMAND ${PYTHON_DEFAULT_EXECUTABLE} "${PYTHON_SOURCE_DIR}/src2/gen2.py" ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_BINARY_DIR}/headers.txt" "${PYTHON}"
    DEPENDS ${PYTHON_SOURCE_DIR}/src2/gen2.py
    DEPENDS ${PYTHON_SOURCE_DIR}/src2/hdr_parser.py
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/headers.txt
    DEPENDS ${opencv_hdrs}
    COMMENT "Generate files for ${the_module}"
)

add_custom_target(gen_${the_module} DEPENDS ${cv2_generated_files})

set(cv2_custom_hdr "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_custom_headers.h")
file(WRITE ${cv2_custom_hdr} "//user-defined headers\n")
foreach(uh ${opencv_userdef_hdrs})
    file(APPEND ${cv2_custom_hdr} "#include \"${uh}\"\n")
endforeach(uh)

ocv_add_library(${the_module} MODULE ${PYTHON_SOURCE_DIR}/src2/cv2.cpp ${cv2_generated_hdrs} ${opencv_userdef_hdrs} ${cv2_custom_hdr})
add_dependencies(${the_module} gen_${the_module})

if(APPLE)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
elseif(WIN32 OR OPENCV_FORCE_PYTHON_LIBS)
  if(${PYTHON}_DEBUG_LIBRARIES AND NOT ${PYTHON}_LIBRARIES MATCHES "optimized.*debug")
    ocv_target_link_libraries(${the_module} LINK_PRIVATE debug ${${PYTHON}_DEBUG_LIBRARIES} optimized ${${PYTHON}_LIBRARIES})
  else()
    ocv_target_link_libraries(${the_module} LINK_PRIVATE ${${PYTHON}_LIBRARIES})
  endif()
endif()
ocv_target_link_libraries(${the_module} LINK_PRIVATE ${OPENCV_MODULE_${the_module}_DEPS})

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
