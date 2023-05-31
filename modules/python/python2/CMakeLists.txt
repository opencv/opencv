if(NOT PYTHON2_INCLUDE_PATH OR NOT PYTHON2_NUMPY_INCLUDE_DIRS)
  ocv_module_disable(python2)
endif()

set(the_description "The python2 bindings")
set(MODULE_NAME python2)
# Buildbot requires Python 2 to be in root lib dir
set(MODULE_INSTALL_SUBDIR "")

set(PYTHON PYTHON2)

include(../common.cmake)

unset(MODULE_NAME)
unset(MODULE_INSTALL_SUBDIR)
