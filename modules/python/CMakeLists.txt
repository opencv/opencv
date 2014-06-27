# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------

if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  ocv_module_disable(python)
endif()

if(ANDROID OR IOS OR NOT PYTHONLIBS_FOUND OR NOT PYTHON_NUMPY_INCLUDE_DIRS)
  ocv_module_disable(python)
endif()

set(the_description "The python bindings")
ocv_add_module(python BINDINGS opencv_core opencv_highgui opencv_flann OPTIONAL opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_calib3d opencv_photo opencv_objdetect opencv_contrib opencv_legacy opencv_softcascade opencv_nonfree opencv_optim opencv_shape)

#ocv_add_module(python BINDINGS opencv_core opencv_flann opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_highgui opencv_calib3d opencv_photo opencv_objdetect OPTIONAL opencv_nonfree)

ocv_module_include_directories(
    "${PYTHON_INCLUDE_PATH}"
    ${PYTHON_NUMPY_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src2"
    )

# -- Check if EXTRA modules path is provided. If provided, build it.

if(NOT OPENCV_EXTRA_MODULES_PATH STREQUAL "")
    set(BUILD_PYTHON_CONTRIB ON)
    set(TEMP_EXTRA_MODULES_PATH ${OPENCV_EXTRA_MODULES_PATH})
else()
    set(BUILD_PYTHON_CONTRIB OFF)
    set(TEMP_EXTRA_MODULES_PATH "null") # Assign something to avoid bug in string(FIND)
endif()

# -- Find the modules to build. Split them to Python modules and Python-Extra modules

string(REPLACE "opencv_" "" OPENCV_PYTHON_MODULES "${OPENCV_MODULES_BUILD}")
ocv_list_filterout(OPENCV_PYTHON_MODULES "^cuda.*$")
foreach(module ${OPENCV_PYTHON_MODULES})
    # Check if module location matches with Extra-modules location
    string(FIND ${OPENCV_MODULE_opencv_${module}_LOCATION} ${TEMP_EXTRA_MODULES_PATH} IS_PYTHON_EXTRA)
    if(NOT HAVE_opencv_${module})
         list(REMOVE_ITEM OPENCV_PYTHON_MODULES ${module})
    elseif(NOT (${OPENCV_MODULE_opencv_${module}_CLASS} STREQUAL "PUBLIC"))
         list(REMOVE_ITEM OPENCV_PYTHON_MODULES ${module})
    # If it is an Extra-module, put it in Python-Extra modules group
    elseif(${IS_PYTHON_EXTRA} GREATER -1)
        list(APPEND OPENCV_PYTHON_EXTRA_MODULES ${module})
        list(REMOVE_ITEM OPENCV_PYTHON_MODULES ${module})
    # For time-being, disable following modules. (NB:  We will enable it one-by-one)
    elseif(${module} MATCHES "ts|optim|stitching|superres|videostab")
        list(REMOVE_ITEM OPENCV_PYTHON_MODULES ${module})
    endif()
endforeach()


# -- Now collect headers for each module for Python(cv2). Extra-modules later

foreach(module ${OPENCV_PYTHON_MODULES})
    set(module_hdrs "${OPENCV_MODULE_opencv_${module}_HEADERS}")
    # Remove compatibility headers
    ocv_list_filterout(module_hdrs "\${module}/\${module}.hpp$")
    # Python bindings doesn't seem to process *.h files. So remove them, so with cuda,ios,opencl etc.
    ocv_list_filterout(module_hdrs ".h$")
    ocv_list_filterout(module_hdrs "^.*(cuda|ios|opencl).*$")
    ocv_list_filterout(module_hdrs "detection_based_tracker.hpp$")
    list(APPEND opencv_hdrs ${module_hdrs})
endforeach()

foreach(i ${opencv_hdrs})
message("opencv_hdrs" : ${i})
endforeach()
#set(opencv_hdrs
#    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core.hpp"
#    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/base.hpp"
#    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/types.hpp"
#    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/persistence.hpp"
#    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/utility.hpp"
#    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/ocl.hpp"
#    "${OPENCV_MODULE_opencv_flann_LOCATION}/include/opencv2/flann/miniflann.hpp"
#    "${OPENCV_MODULE_opencv_imgproc_LOCATION}/include/opencv2/imgproc.hpp"
#    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/background_segm.hpp"
#    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/tracking.hpp"
#    "${OPENCV_MODULE_opencv_photo_LOCATION}/include/opencv2/photo.hpp"
#    "${OPENCV_MODULE_opencv_highgui_LOCATION}/include/opencv2/highgui.hpp"
#    "${OPENCV_MODULE_opencv_ml_LOCATION}/include/opencv2/ml.hpp"
#    "${OPENCV_MODULE_opencv_features2d_LOCATION}/include/opencv2/features2d.hpp"
#    "${OPENCV_MODULE_opencv_calib3d_LOCATION}/include/opencv2/calib3d.hpp"
#    "${OPENCV_MODULE_opencv_objdetect_LOCATION}/include/opencv2/objdetect.hpp"
#    "${OPENCV_MODULE_opencv_softcascade_LOCATION}/include/opencv2/softcascade.hpp"
#    "${OPENCV_MODULE_opencv_contrib_LOCATION}/include/opencv2/contrib.hpp")

#if(HAVE_opencv_nonfree)
#  list(APPEND opencv_hdrs     "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/features2d.hpp"
#                              "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree.hpp")
#endif()

set(cv2_generated_hdrs
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_include.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_funcs.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_func_tab.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_types.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_type_reg.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_const_reg.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_typedefs.h")

add_custom_command(
   OUTPUT ${cv2_generated_hdrs}
   COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src2/gen2.py" ${CMAKE_CURRENT_BINARY_DIR} ${opencv_hdrs}
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/gen2.py
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/hdr_parser.py
   DEPENDS ${opencv_hdrs})


add_library(${the_module} SHARED src2/cv2.cpp ${cv2_generated_hdrs})
set_target_properties(${the_module} PROPERTIES COMPILE_DEFINITIONS OPENCV_NOSTL)

if(PYTHON_DEBUG_LIBRARIES AND NOT PYTHON_LIBRARIES MATCHES "optimized.*debug")
  target_link_libraries(${the_module} debug ${PYTHON_DEBUG_LIBRARIES} optimized ${PYTHON_LIBRARIES})
else()
  target_link_libraries(${the_module} ${PYTHON_LIBRARIES})
endif()

# -- Target linking dependencies.
set(TEMP_OPENCV_PYTHON_DEPS ${OPENCV_PYTHON_MODULES})
ocv_list_add_prefix(TEMP_OPENCV_PYTHON_DEPS "opencv_")
target_link_libraries(${the_module} ${TEMP_OPENCV_PYTHON_DEPS})

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SO'))"
                RESULT_VARIABLE PYTHON_CVPY_PROCESS
                OUTPUT_VARIABLE CVPY_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set_target_properties(${the_module} PROPERTIES
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

#-------------------------------------------------------------------------------------------------
#                   Python bindings for External contrib module --> cv2_contrib
#-------------------------------------------------------------------------------------------------
if(BUILD_PYTHON_CONTRIB)

    foreach(module ${OPENCV_PYTHON_EXTRA_MODULES})
        set(extra_module_hdrs "${OPENCV_MODULE_opencv_${module}_HEADERS}")
        #ocv_list_filterout(extra_module_hdrs "^.*\${module}/\${module}.hpp$")
        ocv_list_filterout(extra_module_hdrs "^.*cuda.*$")
       list(APPEND opencv_contrib_hdrs ${extra_module_hdrs})
    endforeach()

    list(LENGTH opencv_contrib_hdrs number)

    set(TEMP_OPENCV_PYTHON_EXTRA_DEPS ${OPENCV_PYTHON_EXTRA_MODULES})
    ocv_list_add_prefix(TEMP_OPENCV_PYTHON_EXTRA_DEPS "opencv_")

    set(cv2_generated_contrib_hdrs
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_contrib_include.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_contrib_funcs.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_contrib_func_tab.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_contrib_types.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_contrib_type_reg.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_contrib_const_reg.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_contrib_typedefs.h")

    add_custom_command(
       OUTPUT ${cv2_generated_contrib_hdrs}
       COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src2/gen2_contrib.py" ${CMAKE_CURRENT_BINARY_DIR} ${opencv_contrib_hdrs}
       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/gen2_contrib.py
       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/hdr_parser.py
       DEPENDS ${opencv_contrib_hdrs})

    add_library(contrib_module SHARED src2/cv2_contrib.cpp ${cv2_generated_contrib_hdrs})

    if(PYTHON_DEBUG_LIBRARIES AND NOT PYTHON_LIBRARIES MATCHES "optimized.*debug")
      target_link_libraries(contrib_module debug ${PYTHON_DEBUG_LIBRARIES} optimized ${PYTHON_LIBRARIES})
    else()
      target_link_libraries(contrib_module ${PYTHON_LIBRARIES})
    endif()

    set_target_properties(contrib_module PROPERTIES COMPILE_DEFINITIONS OPENCV_NOSTL)

    #target_link_libraries(contrib_module ${OPENCV_MODULE_${the_module}_DEPS})
    target_link_libraries(contrib_module ${the_module} ${TEMP_OPENCV_PYTHON_EXTRA_DEPS})

    set_target_properties(contrib_module PROPERTIES
                          PREFIX ""
                          OUTPUT_NAME cv2_contrib
                          SUFFIX ${CVPY_SUFFIX})

    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(contrib_module PROPERTIES FOLDER "bindings")
    endif()

    if(MSVC AND NOT BUILD_SHARED_LIBS)
      set_target_properties(contrib_module PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
    endif()

    if(NOT INSTALL_CREATE_DISTRIB)
      install(TARGETS contrib_module
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
      install(TARGETS contrib_module
              CONFIGURATIONS Release
              RUNTIME DESTINATION python/${__ver}/${OpenCV_ARCH} COMPONENT python
              LIBRARY DESTINATION python/${__ver}/${OpenCV_ARCH} COMPONENT python
              )
    endif()

endif()
