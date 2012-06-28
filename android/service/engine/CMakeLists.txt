set(engine OpenCVEngine)
set(JNI_LIB_NAME ${engine} ${engine}_jni)
add_android_project(opencv_engine "${CMAKE_CURRENT_SOURCE_DIR}" SDK_TARGET 8 ${ANDROID_SDK_TARGET} IGNORE_JAVA ON)

link_directories("${ANDROID_SOURCE_TREE}/out/target/product/generic/system/lib" "${ANDROID_SOURCE_TREE}/bin_${ANDROID_ARCH_NAME}/system/lib")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-allow-shlib-undefined")



file(GLOB engine_files "jni/BinderComponent/*.cpp" "jni/BinderComponent/*.h" "jni/include/*.h")

include_directories(jni/BinderComponent jni/include "${ANDROID_SOURCE_TREE}/frameworks/base/include" "${ANDROID_SOURCE_TREE}/system/core/include")
add_library(${engine} SHARED ${engine_files})
target_link_libraries(${engine} z binder log utils)

set_target_properties(${engine} PROPERTIES
    OUTPUT_NAME ${engine}
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.build/libs/${ANDROID_NDK_ABI_NAME}"
    )

get_target_property(engine_lib_location ${engine} LOCATION)
add_custom_command(TARGET ${engine} POST_BUILD COMMAND ${CMAKE_STRIP} --strip-unneeded "${engine_lib_location}")

file(GLOB engine_jni_files "jni/JNIWrapper/*.cpp" "jni/JNIWrapper/*.h" "jni/include/*.h")
list(APPEND engine_jni_files jni/NativeService/CommonPackageManager.cpp jni/NativeService/PackageInfo.cpp)

include_directories(jni/include jni/JNIWrapper jni/NativeService jni/BinderComponent "${ANDROID_SOURCE_TREE}/frameworks/base/include" "${ANDROID_SOURCE_TREE}/system/core/include" "${ANDROID_SOURCE_TREE}/frameworks/base/core/jni")
add_library(${engine}_jni SHARED ${engine_jni_files})
target_link_libraries(${engine}_jni z binder log utils android_runtime ${engine})

set_target_properties(${engine}_jni PROPERTIES
    OUTPUT_NAME ${engine}_jni
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.build/libs/${ANDROID_NDK_ABI_NAME}"
    )

get_target_property(engine_lib_location ${engine}_jni LOCATION)
add_custom_command(TARGET ${engine}_jni POST_BUILD COMMAND ${CMAKE_STRIP} --strip-unneeded "${engine_lib_location}")
