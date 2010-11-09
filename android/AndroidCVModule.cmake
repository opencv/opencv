macro(define_android_manual name lib_srcs includes)
set(android_module_name ${name})
set(android_srcs "")
set(include_dirs "${includes}")
foreach(f ${lib_srcs})
		string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" n_f ${f})	
		set(android_srcs "${android_srcs} ${n_f}")
endforeach()
configure_file("${CMAKE_SOURCE_DIR}/Android.mk.in" "${CMAKE_CURRENT_BINARY_DIR}/Android.mk")
endmacro()


macro(define_3rdparty_module name)
	file(GLOB lib_srcs "*.c" "*.cpp")
	file(GLOB lib_int_hdrs "*.h*")
	define_android_manual(${name} "${lib_srcs}" "$(LOCAL_PATH)/../include")		
endmacro()

macro(define_opencv_module name)
	file(GLOB lib_srcs "src/*.cpp")
	file(GLOB lib_int_hdrs "src/*.h*")
	define_android_manual(opencv_${name} "${lib_srcs}" "$(LOCAL_PATH)/src  $(OPENCV_INCLUDES)")
endmacro()





