set(the_description "The Headers-only CUDA Device Primitives")

if(NOT HAVE_CUDA)
  ocv_module_disable(opencv_${the_module})
endif()

ocv_add_module(cudevice)
ocv_glob_module_sources()

add_library(${the_module} STATIC ${OPENCV_MODULE_${the_module}_HEADERS})
set_target_properties(${the_module} PROPERTIES LINKER_LANGUAGE CXX)