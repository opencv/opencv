if(NOT PYTHON3_INCLUDE_PATH OR NOT PYTHON3_NUMPY_INCLUDE_DIRS)
  ocv_module_disable(python3)
endif()

# Problem in numpy >=1.15 <1.17
if(PYTHON3_LIMITED_API
    AND NOT PYTHON3_NUMPY_VERSION VERSION_LESS "1.15"
    AND PYTHON3_NUMPY_VERSION VERSION_LESS "1.17"
  )
  message(WARNING "Current NUMPY version (${PYTHON3_NUMPY_VERSION}) is not compatible with LIMITED_API.")
  set(PYTHON3_LIMITED_API OFF)
endif()

set(the_description "The python3 bindings")
set(MODULE_NAME python3)
set(MODULE_INSTALL_SUBDIR python3)

set(_ocv_extra_modules_path ${CMAKE_CURRENT_LIST_DIR}/../package/extra_modules)
set(_old_ocv_python_extra_modules_path ${OCV_PYTHON_EXTRA_MODULES_PATH})

if("${OCV_PYTHON_EXTRA_MODULES_PATH}" STREQUAL "")
  set(OCV_PYTHON_EXTRA_MODULES_PATH ${_ocv_extra_modules_path})
else()
  list(APPEND OCV_PYTHON_EXTRA_MODULES_PATH ${_ocv_extra_modules_path})
endif()

unset(_ocv_extra_modules_path)

set(PYTHON PYTHON3)

include(../common.cmake)

set(OCV_PYTHON_EXTRA_MODULES_PATH ${_old_ocv_python_extra_modules_path})

unset(_old_ocv_python_extra_modules_path)
unset(MODULE_NAME)
unset(MODULE_INSTALL_SUBDIR)
