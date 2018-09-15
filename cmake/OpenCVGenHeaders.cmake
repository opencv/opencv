# platform-specific config file
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/cvconfig.h.in" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/cvconfig.h")
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/cvconfig.h.in" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/opencv2/cvconfig.h")
install(FILES "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/cvconfig.h" DESTINATION ${OPENCV_INCLUDE_INSTALL_PATH}/opencv2 COMPONENT dev)

# platform-specific config file
ocv_compiler_optimization_fill_cpu_config()
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/cv_cpu_config.h.in" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/cv_cpu_config.h")

# ----------------------------------------------------------------------------
#  opencv_modules.hpp based on actual modules list
# ----------------------------------------------------------------------------
set(OPENCV_MOD_LIST ${OPENCV_MODULES_PUBLIC})
ocv_list_sort(OPENCV_MOD_LIST)

ocv_assert(OPENCV_PACKAGES)
foreach(package ${OPENCV_PACKAGES})
  ocv_clear_vars(OPENCV_${package}_MODULE_DEFINITIONS_CONFIGMAKE)
endforeach()

foreach(module ${OPENCV_MOD_LIST})
  string(TOUPPER "${module}" m)
  if(OPENCV_MODULE_${module}_PACKAGE)
    set(package "${OPENCV_MODULE_${module}_PACKAGE}")
    set(OPENCV_${package}_MODULE_DEFINITIONS_CONFIGMAKE "${OPENCV_${package}_MODULE_DEFINITIONS_CONFIGMAKE}#define HAVE_${m}\n")
  else()
    ocv_target_compile_definitions(${module} PUBLIC HAVE_${m}=1)
  endif()
endforeach()

ocv_assert(OPENCV_PACKAGES)
foreach(package ${OPENCV_PACKAGES})
  if(package STREQUAL "main")
    set(header_template_name "opencv_modules.hpp.in")
    set(header_name "opencv_modules.hpp")
  else()
    set(header_template_name "opencv_package_modules.hpp.in")
    set(header_name "opencv_${package}_modules.hpp")
  endif()
  set(OPENCV_PACKAGE_NAME "${package}")

  set(OPENCV_MODULE_DEFINITIONS_CONFIGMAKE "${OPENCV_${package}_MODULE_DEFINITIONS_CONFIGMAKE}")

  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/${header_template_name}" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/opencv2/${header_name}")
  install(FILES "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/opencv2/${header_name}" DESTINATION ${OPENCV_INCLUDE_INSTALL_PATH}/opencv2 COMPONENT dev)
endforeach()
