find_package(ALSA)
if(ALSA_FOUND)
  include_directories(${ALSA_INCLUDE_DIRS})
endif(ALSA_FOUND)
