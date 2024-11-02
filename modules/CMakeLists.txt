add_definitions(-D__OPENCV_BUILD=1)

if(NOT OPENCV_MODULES_PATH)
  set(OPENCV_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

ocv_glob_modules(${OPENCV_MODULES_PATH} ${OPENCV_EXTRA_MODULES_PATH})

# build lists of modules to be documented
set(OPENCV_MODULES_MAIN "")
set(OPENCV_MODULES_EXTRA "")

foreach(mod ${OPENCV_MODULES_BUILD} ${OPENCV_MODULES_DISABLED_USER} ${OPENCV_MODULES_DISABLED_AUTO} ${OPENCV_MODULES_DISABLED_FORCE})
  string(REGEX REPLACE "^opencv_" "" mod "${mod}")
  if("${OPENCV_MODULE_opencv_${mod}_LOCATION}" STREQUAL "${OpenCV_SOURCE_DIR}/modules/${mod}")
    list(APPEND OPENCV_MODULES_MAIN ${mod})
  else()
    list(APPEND OPENCV_MODULES_EXTRA ${mod})
  endif()
endforeach()
ocv_list_sort(OPENCV_MODULES_MAIN)
ocv_list_sort(OPENCV_MODULES_EXTRA)
set(FIXED_ORDER_MODULES core imgproc imgcodecs videoio highgui video 3d stereo features calib objdetect dnn flann photo stitching)
list(REMOVE_ITEM OPENCV_MODULES_MAIN ${FIXED_ORDER_MODULES})
set(OPENCV_MODULES_MAIN ${FIXED_ORDER_MODULES} ${OPENCV_MODULES_MAIN})

set(OPENCV_MODULES_MAIN ${OPENCV_MODULES_MAIN} CACHE INTERNAL "List of main modules" FORCE)
set(OPENCV_MODULES_EXTRA ${OPENCV_MODULES_EXTRA} CACHE INTERNAL "List of extra modules" FORCE)
