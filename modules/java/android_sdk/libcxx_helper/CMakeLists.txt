cmake_minimum_required(VERSION 3.6)

project(opencv_jni_shared)

if (ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES AND (ANDROID_ABI STREQUAL arm64-v8a OR ANDROID_ABI STREQUAL x86_64))
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
endif()

# dummy target to bring libc++_shared.so into packages
add_library(opencv_jni_shared STATIC dummy.cpp)
