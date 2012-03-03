SET (the_target native_camera_r${ANDROID_VERSION})

project(${the_target})

link_directories("${ANDROID_SOURCE_TREE}/out/target/product/generic/system/lib")

INCLUDE_DIRECTORIES(BEFORE
	${ANDROID_SOURCE_TREE}
	${ANDROID_SOURCE_TREE}/frameworks/base/include/ui
	${ANDROID_SOURCE_TREE}/frameworks/base/include/surfaceflinger
	${ANDROID_SOURCE_TREE}/frameworks/base/include/camera
	${ANDROID_SOURCE_TREE}/frameworks/base/include/media
	${ANDROID_SOURCE_TREE}/frameworks/base/include/camera
	${ANDROID_SOURCE_TREE}/frameworks/base/include
	${ANDROID_SOURCE_TREE}/system/core/include
	${ANDROID_SOURCE_TREE}/hardware/libhardware/include
	${ANDROID_SOURCE_TREE}/frameworks/base/native/include
	)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
SET(CMAKE_C_FLAGS_RELEASE "")
SET(CMAKE_CXX_FLAGS_RELEASE "")

string(REPLACE "-O3" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-O3" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os -fno-strict-aliasing -finline-limit=64 -fuse-cxa-atexit" )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -fno-strict-aliasing -finline-limit=64 -fuse-cxa-atexit")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,noexecstack")

ADD_LIBRARY(${the_target} SHARED camera_wrapper.h camera_wrapper.cpp)

string(REGEX REPLACE "[.]" "_" LIBRARY_DEF ${ANDROID_VERSION})
add_definitions(-DANDROID_r${LIBRARY_DEF})

target_link_libraries(${the_target} c m dl utils camera_client binder log)

if(NOT ANDROID_VERSION VERSION_LESS "3.0.0")
    target_link_libraries(${the_target} gui )
endif()

SET_TARGET_PROPERTIES(${the_target} PROPERTIES
                      OUTPUT_NAME "${the_target}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      )

# force strip library after the build command
ADD_CUSTOM_COMMAND( TARGET ${the_target} POST_BUILD COMMAND ${CMAKE_STRIP} "${LIBRARY_OUTPUT_PATH}/lib${the_target}.so" )


install(TARGETS ${the_target} LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main)
