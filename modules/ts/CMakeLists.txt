if(IOS)
  ocv_module_disable(ts)
endif()

if(MINGW)
    set(OPENCV_MODULE_TYPE STATIC)
endif()
    
set(the_description "The ts module")
ocv_add_module(ts opencv_core)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

if(BUILD_SHARED_LIBS AND NOT MINGW)
    add_definitions(-DGTEST_CREATE_SHARED_LIBRARY=1)
    if (MSVC)
        add_definitions( "/wd4251 /wd4275")
    endif()
else()
    add_definitions(-DGTEST_CREATE_SHARED_LIBRARY=0)
endif()

ocv_add_precompiled_headers(${the_module})
