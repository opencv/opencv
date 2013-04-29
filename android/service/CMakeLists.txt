if(BUILD_ANDROID_SERVICE)
  add_subdirectory(engine)
  #add_subdirectory(engine_test)
endif()

install(FILES "readme.txt" DESTINATION "apk/" COMPONENT main)
