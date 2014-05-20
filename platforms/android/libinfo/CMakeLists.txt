project(libopencv_info)
if(NOT ANDROID_PACKAGE_RELEASE)
  set(ANDROID_PACKAGE_RELEASE 1)
endif()

if(NOT ANDROID_PACKAGE_PLATFORM)
  if(ARMEABI_V7A)
    if(NEON)
      set(ANDROID_PACKAGE_PLATFORM armv7a_neon)
    else()
      set(ANDROID_PACKAGE_PLATFORM armv7a)
    endif()
  elseif(ARMEABI_V6)
    set(ANDROID_PACKAGE_PLATFORM armv6)
  elseif(ARMEABI)
    set(ANDROID_PACKAGE_PLATFORM armv5)
  elseif(X86)
    set(ANDROID_PACKAGE_PLATFORM x86)
  elseif(MIPS)
    set(ANDROID_PACKAGE_PLATFORM mips)
  else()
    message(ERROR "Can not automatically determine the value for ANDROID_PACKAGE_PLATFORM")
  endif()
endif()

add_definitions(-DANDROID_PACKAGE_RELEASE=${ANDROID_PACKAGE_RELEASE} -DANDROID_PACKAGE_PLATFORM="${ANDROID_PACKAGE_PLATFORM}")

include_directories(jni/BinderComponent jni/include "${OpenCV_SOURCE_DIR}/modules/core/include")

add_library(opencv_info SHARED info.c)

set_target_properties(${the_module} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    INSTALL_NAME_DIR lib
  )

get_filename_component(lib_name "libopencv_info.so" NAME)
install(FILES "${LIBRARY_OUTPUT_PATH}/${lib_name}" DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT libs)
