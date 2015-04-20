subdirs(Cxx)

if(GDCM_WRAP_PYTHON)
  subdirs(Python)
endif()

if(GDCM_WRAP_JAVA)
  subdirs(Java)
endif()
