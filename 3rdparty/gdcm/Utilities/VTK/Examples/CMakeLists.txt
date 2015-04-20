#subdirs(Cxx Python)
subdirs(Cxx)

if(GDCM_WRAP_CSHARP)
  subdirs(Csharp)
endif()

if(GDCM_WRAP_JAVA)
  if(VTK_WRAP_JAVA)
    subdirs(Java)
  endif()
endif()
