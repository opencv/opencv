SET (the_target native_camera_r${ANDROID_VERSION})

project(${the_target})

link_directories("${ANDROID_SOURCE_TREE}/out/target/product/generic/system/lib")

INCLUDE_DIRECTORIES(
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

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-rtti -fno-exceptions" )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

ADD_LIBRARY(${the_target} MODULE camera_wrapper.h camera_wrapper.cpp)

string(REGEX REPLACE "[.]" "_" LIBRARY_DEF ${ANDROID_VERSION})
add_definitions(-DANDROID_r${LIBRARY_DEF})

target_link_libraries(${the_target} dl m stdc++ utils camera_client ui media binder cutils surfaceflinger_client supc++ log )

SET_TARGET_PROPERTIES(${the_target} PROPERTIES
                      OUTPUT_NAME "${the_target}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      )

install(TARGETS ${the_target}
  RUNTIME DESTINATION bin COMPONENT main
  ARCHIVE DESTINATION lib COMPONENT main
  LIBRARY DESTINATION lib COMPONENT main
  )