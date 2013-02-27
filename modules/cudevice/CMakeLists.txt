set(the_description "The Headers-only CUDA Device Primitives")

if(NOT HAVE_CUDA)
  ocv_module_disable(opencv_${the_module})
  return()
endif()

ocv_add_module(cudevice )
ocv_glob_module_sources()
ocv_module_include_directories()
set(OPENCV_MODULE_TYPE STATIC)
ocv_create_module()
set_target_properties(${the_module} PROPERTIES LINKER_LANGUAGE CXX)