# This file is included from a subdirectory
set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")

set(candidate_deps "")
foreach(mp ${OPENCV_MODULES_PATH} ${OPENCV_EXTRA_MODULES_PATH})
    file(GLOB names "${mp}/*")
    foreach(m IN LISTS names)
        if(IS_DIRECTORY ${m})
            get_filename_component(m ${m} NAME)
            list(APPEND candidate_deps "opencv_${m}")
        endif()
    endforeach(m)
endforeach(mp)

# module blacklist
ocv_list_filterout(candidate_deps "^opencv_cud(a|ev)")
ocv_list_filterout(candidate_deps "^opencv_adas$")
ocv_list_filterout(candidate_deps "^opencv_face$")
ocv_list_filterout(candidate_deps "^opencv_matlab$")
ocv_list_filterout(candidate_deps "^opencv_tracking$")
ocv_list_filterout(candidate_deps "^opencv_optflow$")
ocv_list_filterout(candidate_deps "^opencv_bgsegm$")
ocv_list_filterout(candidate_deps "^opencv_xfeatures2d$")
ocv_list_filterout(candidate_deps "^opencv_ts$")

ocv_add_module(${MODULE_NAME} BINDINGS OPTIONAL ${candidate_deps})

ocv_module_include_directories(
    "${PYTHON_INCLUDE_PATH}"
    ${PYTHON_NUMPY_INCLUDE_DIRS}
    "${PYTHON_SOURCE_DIR}/src2"
    )

set(opencv_hdrs "")
foreach(m IN LISTS OPENCV_MODULE_opencv_${MODULE_NAME}_DEPS)
    list(APPEND opencv_hdrs ${OPENCV_MODULE_${m}_HEADERS})
endforeach(m)

# header blacklist
ocv_list_filterout(opencv_hdrs ".h$")
ocv_list_filterout(opencv_hdrs "cuda")
ocv_list_filterout(opencv_hdrs "cudev")
ocv_list_filterout(opencv_hdrs "opencv2/objdetect/detection_based_tracker.hpp")
ocv_list_filterout(opencv_hdrs "opencv2/optim.hpp")

set(cv2_generated_hdrs
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_include.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_funcs.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_func_tab.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_types.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_type_reg.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_const_reg.h")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/headers.txt" "${opencv_hdrs}")
add_custom_command(
   OUTPUT ${cv2_generated_hdrs}
   COMMAND ${PYTHON_EXECUTABLE} "${PYTHON_SOURCE_DIR}/src2/gen2.py" ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_BINARY_DIR}/headers.txt"
   DEPENDS ${PYTHON_SOURCE_DIR}/src2/gen2.py
   DEPENDS ${PYTHON_SOURCE_DIR}/src2/hdr_parser.py
   DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/headers.txt
   DEPENDS ${opencv_hdrs})

ocv_add_library(${the_module} SHARED ${PYTHON_SOURCE_DIR}/src2/cv2.cpp ${cv2_generated_hdrs})
set_target_properties(${the_module} PROPERTIES COMPILE_DEFINITIONS OPENCV_NOSTL)

if(PYTHON_DEBUG_LIBRARIES AND NOT PYTHON_LIBRARIES MATCHES "optimized.*debug")
  ocv_target_link_libraries(${the_module} debug ${PYTHON_DEBUG_LIBRARIES} optimized ${PYTHON_LIBRARIES})
else()
  ocv_target_link_libraries(${the_module} ${PYTHON_LIBRARIES})
endif()
ocv_target_link_libraries(${the_module} ${OPENCV_MODULE_${the_module}_DEPS})

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SO'))"
                RESULT_VARIABLE PYTHON_CVPY_PROCESS
                OUTPUT_VARIABLE CVPY_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set_target_properties(${the_module} PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY  "${LIBRARY_OUTPUT_PATH}/${MODULE_INSTALL_SUBDIR}"
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

if(MSVC AND NOT BUILD_SHARED_LIBS)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
endif()

if(MSVC AND NOT PYTHON_DEBUG_LIBRARIES)
  set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
else()
  set(PYTHON_INSTALL_CONFIGURATIONS "")
endif()

if(WIN32)
  set(PYTHON_INSTALL_ARCHIVE "")
else()
  set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python)
endif()

if(NOT INSTALL_CREATE_DISTRIB)
  install(TARGETS ${the_module}
          ${PYTHON_INSTALL_CONFIGURATIONS}
          RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
          LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
          ${PYTHON_INSTALL_ARCHIVE}
          )
else()
  if(DEFINED PYTHON_VERSION_MAJOR)
    set(__ver "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  else()
    set(__ver "unknown")
  endif()
  install(TARGETS ${the_module}
          CONFIGURATIONS Release
          RUNTIME DESTINATION python/${__ver}/${OpenCV_ARCH} COMPONENT python
          LIBRARY DESTINATION python/${__ver}/${OpenCV_ARCH} COMPONENT python
          )
endif()

unset(PYTHON_SRC_DIR)
unset(PYTHON_CVPY_PROCESS)
unset(CVPY_SUFFIX)
unset(PYTHON_INSTALL_CONFIGURATIONS)
unset(PYTHON_INSTALL_ARCHIVE)
