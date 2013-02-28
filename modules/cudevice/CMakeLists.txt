set(the_description "The Header-only CUDA Device Primitives")

if(NOT HAVE_CUDA)
  ocv_module_disable(opencv_${the_module})
  return()
endif()

set(OPENCV_MODULE_TYPE STATIC)
ocv_define_module(cudevice)
set_target_properties(${the_module} PROPERTIES LINKER_LANGUAGE CXX)