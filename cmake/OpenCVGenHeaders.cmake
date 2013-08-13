# platform-specific config file
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/cvconfig.h.cmake" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/cvconfig.h")

# ----------------------------------------------------------------------------
#  opencv_modules.hpp based on actual modules list
# ----------------------------------------------------------------------------
set(OPENCV_MODULE_DEFINITIONS_CONFIGMAKE "")

set(OPENCV_MOD_LIST ${OPENCV_MODULES_PUBLIC})
ocv_list_sort(OPENCV_MOD_LIST)
foreach(m ${OPENCV_MOD_LIST})
  string(TOUPPER "${m}" m)
  set(OPENCV_MODULE_DEFINITIONS_CONFIGMAKE "${OPENCV_MODULE_DEFINITIONS_CONFIGMAKE}#define HAVE_${m}\n")
endforeach()

set(OPENCV_MODULE_DEFINITIONS_CONFIGMAKE "${OPENCV_MODULE_DEFINITIONS_CONFIGMAKE}\n")

#set(OPENCV_MOD_LIST ${OPENCV_MODULES_DISABLED_USER} ${OPENCV_MODULES_DISABLED_AUTO} ${OPENCV_MODULES_DISABLED_FORCE})
#ocv_list_sort(OPENCV_MOD_LIST)
#foreach(m ${OPENCV_MOD_LIST})
#  string(TOUPPER "${m}" m)
#  set(OPENCV_MODULE_DEFINITIONS_CONFIGMAKE "${OPENCV_MODULE_DEFINITIONS_CONFIGMAKE}#undef HAVE_${m}\n")
#endforeach()

configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/opencv_modules.hpp.in" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/opencv2/opencv_modules.hpp")
install(FILES "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/opencv2/opencv_modules.hpp" DESTINATION ${OPENCV_INCLUDE_INSTALL_PATH}/opencv2 COMPONENT main)
