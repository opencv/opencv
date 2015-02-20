if(INSTALL_PYTHON_EXAMPLES)
  file(GLOB install_list *.py )
  install(FILES ${install_list}
          DESTINATION ${OPENCV_SAMPLES_SRC_INSTALL_PATH}/python2
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()
