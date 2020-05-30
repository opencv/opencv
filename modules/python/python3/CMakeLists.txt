if(NOT PYTHON3_INCLUDE_PATH OR NOT PYTHON3_NUMPY_INCLUDE_DIRS)
  ocv_module_disable(python3)
endif()

# Problem in numpy >=1.15 <1.17
if(PYTHON3_LIMITED_API
    AND NOT PYTHON3_NUMPY_VERSION VERSION_LESS "1.15"
    AND PYTHON3_NUMPY_VERSION VERSION_LESS "1.17"
  )
  set(PYTHON3_LIMITED_API OFF)
endif()

set(the_description "The python3 bindings")
set(MODULE_NAME python3)
set(MODULE_INSTALL_SUBDIR python3)

set(PYTHON PYTHON3)

include(../common.cmake)

unset(MODULE_NAME)
unset(MODULE_INSTALL_SUBDIR)
