if(BUILD_ANDROID_SERVICE)
  add_subdirectory(engine)
endif()

install(FILES "readme.txt" DESTINATION "apk/" COMPONENT libs)
