subdirs(Cxx)

if(VTK_WRAP_PYTHON)
  subdirs(Python)
endif()

if(GDCM_WRAP_JAVA)
  if(VTK_WRAP_JAVA)
    subdirs(Java)
  endif()
endif()
