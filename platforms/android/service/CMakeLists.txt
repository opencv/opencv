if(NOT ANDROID_PROJECTS_BUILD_TYPE STREQUAL "ANT")
  message(STATUS "Android OpenCV Manager is ignored")
  return()
endif()

if(BUILD_ANDROID_SERVICE)
  add_subdirectory(engine)
endif()

install(FILES "readme.txt" DESTINATION "apk/" COMPONENT libs)
