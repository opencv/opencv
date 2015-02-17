set(engine OpenCVEngine)
set(JNI_LIB_NAME ${engine} ${engine}_jni)

unset(__android_project_chain CACHE)
add_android_project(opencv_engine "${CMAKE_CURRENT_SOURCE_DIR}" SDK_TARGET 9 ${ANDROID_SDK_TARGET} IGNORE_JAVA ON IGNORE_MANIFEST ON )

set(ANDROID_PLATFORM_VERSION_CODE "0")

if(ARMEABI_V7A)
  if (ANDROID_NATIVE_API_LEVEL LESS 9)
    set(ANDROID_PLATFORM_VERSION_CODE "2")
  else()
    set(ANDROID_PLATFORM_VERSION_CODE "3")
  endif()
elseif(ARM64_V8A)
  set(ANDROID_PLATFORM_VERSION_CODE "4")
elseif(ARMEABI_V6)
  set(ANDROID_PLATFORM_VERSION_CODE "1")
elseif(ARMEABI)
  set(ANDROID_PLATFORM_VERSION_CODE "1")
elseif(X86)
  set(ANDROID_PLATFORM_VERSION_CODE "5")
elseif(MIPS)
  set(ANDROID_PLATFORM_VERSION_CODE "6")
else()
  message(WARNING "Can not automatically determine the value for ANDROID_PLATFORM_VERSION_CODE")
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${ANDROID_MANIFEST_FILE}" "${OpenCV_BINARY_DIR}/platforms/android/service/engine/.build/${ANDROID_MANIFEST_FILE}"  @ONLY)

link_directories(
  "${ANDROID_SOURCE_TREE}/out/target/product/generic/system/lib"
  "${ANDROID_SOURCE_TREE}/out/target/product/${ANDROID_PRODUCT}/system/lib"
  "${ANDROID_SOURCE_TREE}/bin/${ANDROID_ARCH_NAME}")

file(GLOB engine_files "jni/BinderComponent/*.cpp" "jni/BinderComponent/*.h" "jni/include/*.h")
set(engine_libs "z" "binder" "log" "utils")

if (TEGRA_DETECTOR)
  if (ANDROID_NATIVE_API_LEVEL GREATER 8)
    add_definitions(-DUSE_TEGRA_HW_DETECTOR)
    list(APPEND engine_libs ${TEGRA_DETECTOR} GLESv2 EGL)
  else()
    message(FATAL_ERROR "Tegra detector required native api level 9 or above")
  endif()
endif()

# -D__SUPPORT_ARMEABI_FEATURES key is also available
add_definitions(-DPLATFORM_ANDROID -D__SUPPORT_ARMEABI_V7A_FEATURES -D__SUPPORT_TEGRA3 -D__SUPPORT_MIPS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-allow-shlib-undefined")

include_directories("jni/BinderComponent" "jni/include")
include_directories(SYSTEM "${ANDROID_SOURCE_TREE}/frameworks/base/include" "${ANDROID_SOURCE_TREE}/system/core/include")
add_library(${engine} SHARED ${engine_files})
target_link_libraries(${engine} ${engine_libs})

set_target_properties(${engine} PROPERTIES
    OUTPUT_NAME ${engine}
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.build/libs/${ANDROID_NDK_ABI_NAME}"
    )

get_target_property(engine_lib_location ${engine} LOCATION)
add_custom_command(TARGET ${engine} POST_BUILD COMMAND ${CMAKE_STRIP} --strip-unneeded "${engine_lib_location}")

file(GLOB engine_jni_files "jni/JNIWrapper/*.cpp" "jni/JNIWrapper/*.h" "jni/include/*.h")
list(APPEND engine_jni_files jni/NativeService/CommonPackageManager.cpp jni/NativeService/PackageInfo.cpp)

include_directories(
  jni/include jni/JNIWrapper
  jni/NativeService
  jni/BinderComponent
  "${ANDROID_SOURCE_TREE}/frameworks/base/include"
  "${ANDROID_SOURCE_TREE}/system/core/include"
  "${ANDROID_SOURCE_TREE}/frameworks/base/core/jni"
  )

add_library(${engine}_jni SHARED ${engine_jni_files})
target_link_libraries(${engine}_jni z binder log utils android_runtime ${engine})

set_target_properties(${engine}_jni PROPERTIES
    OUTPUT_NAME ${engine}_jni
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.build/libs/${ANDROID_NDK_ABI_NAME}"
    )

get_target_property(engine_lib_location ${engine}_jni LOCATION)
add_custom_command(TARGET ${engine}_jni POST_BUILD COMMAND ${CMAKE_STRIP} --strip-unneeded "${engine_lib_location}")

# native tests
add_definitions(-DGTEST_HAS_CLONE=0 -DANDROID -DGTEST_HAS_TR1_TUPLE=0)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-allow-shlib-undefined")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/jni/Tests)
file(GLOB engine_test_files "jni/Tests/*.cpp")

add_executable(opencv_test_engine ${engine_test_files} jni/Tests/gtest/gtest-all.cpp)
target_link_libraries(opencv_test_engine z binder log utils android_runtime ${engine} ${engine}_jni)
