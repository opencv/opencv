# python
if(GDCM_WRAP_PYTHON)
  subdirs(Python)
endif()
# java
if(GDCM_WRAP_JAVA)
  subdirs(Java)
endif()
# php
if(GDCM_WRAP_PHP)
  subdirs(PHP)
endif()
# perl
if(GDCM_WRAP_PERL)
  subdirs(Perl)
endif()
# C#
if(GDCM_WRAP_CSHARP)
#  subdirs(Csharp)
endif()
