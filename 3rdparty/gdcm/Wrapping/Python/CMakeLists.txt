# Try to rebuild wrapping a little more often:
include_regular_expression("^(gdcm).*$")
# TODO:
# SWIG is really a pain in the neck to use, a better alternative is Py++ which is using
# gccxml for the C++ parser and allow a full ANSI C++ support
# Note gcc has some issue with RTTI stuff:
# http://groups.google.com/group/comp.lang.c++.moderated/browse_thread/thread/ac889a7d9eac902f
# http://gcc.gnu.org/ml/gcc-help/2007-10/msg00239.html
# http://wiki.python.org/moin/boost.python/CrossExtensionModuleDependencies
# http://www.boost-consulting.com/writing/bpl.html
# http://gcc.gnu.org/ml/gcc/2002-05/msg00866.html
# http://mail.python.org/pipermail/c++-sig/2002-May/001021.html
# http://mail.python.org/pipermail/python-dev/2002-May/023923.html

# 2.0.5 is required here to solve the issue with wrapping of vector<int>::size_type
# see full thread at:
# http://sourceforge.net/mailarchive/message.php?msg_id=29217941
# 2.0.5 cannot be used because of the error
#  error: invalid initialization of reference of type 'ptrdiff_t& {aka int&}' from expression of type 'long int'
# http://sourceforge.net/mailarchive/message.php?msg_id=29294773
# 2.0.6 cannot be used because of a serious typemap bug in 2.0.5
# http://sourceforge.net/mailarchive/message.php?msg_id=29305946
find_package(SWIG 2.0.7 REQUIRED)
mark_as_advanced(SWIG_DIR SWIG_EXECUTABLE SWIG_VERSION)
include(${SWIG_USE_FILE})

#
# Do not cover this lib
#
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/.NoDartCoverage
                ${CMAKE_CURRENT_BINARY_DIR}/.NoDartCoverage)

# Note:
# python -c "from struct import pack; print  pack('5b', (41*len('99')), pow(8,2)+20, 4900**0.5, range(78)[-1], 10)"

include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/MessageExchangeDefinition"
  ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
# TODO Need to check consistency python interp and python libs...
mark_as_advanced(PYTHON_LIBRARY PYTHON_INCLUDE_PATH)
# Lamest excuse ever:
# http://mail.python.org/pipermail/python-list/2002-April/141189.html
# So here come craziest hack ever, since I cannot control the output of swig,
# I need to fake a Python.h file only for MSVC compilers...insane !
if(MSVC)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Python.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Python.h @ONLY
  )
  include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
else()
# just plain including pyconfig.h is working...until one crazy python dev decide otherwise...
  include_directories(
    ${PYTHON_INCLUDE_PATH}
  )
endif()
set_source_files_properties(gdcmswig.i PROPERTIES CPLUSPLUS ON)

set(GDCM_PYTHON_IMPLEMENTATION_NAME gdcmswig)
set(MODULE_NAME gdcmswig)
# BUG: DO NOT USE -interface flag it is NOT supported in cmake and in cmake > 2.6 will cause infinite rebuild
# Ref: http://www.cmake.org/pipermail/cmake/2008-August/023237.html
# UseSWIG and -interface flag (was: Re: CMake 2.6.1 available for download)
if(${PYTHON_VERSION_MAJOR} EQUAL 3)
  # http://swig.org/Doc2.0/SWIGDocumentation.html#Python_python3support
  set(CMAKE_SWIG_FLAGS "-py3")
endif()
# TODO: PythonInterp and PythonLibs are not working together well
# see: http://bugs.debian.org/677598
#message(${PYTHONLIBS_VERSION_STRING})
#message(${PYTHON_VERSION_MAJOR})
#separate_arguments(CMAKE_SWIG_FLAGS)

# While trying to get rid of the compilation warning in swig generated c++ code, I thought I could
# simply do the following:
#set(CMAKE_CXX_FLAGS "")
# well no, you cannot, it get rid of some important flags, and make the _gdcm.so incompatible with
# the other gdcm lib. bad !!!
#set (SWIG_MODULE_${MODULE_NAME}_EXTRA_DEPS ${SWIG_MODULE_${MODULE_NAME}_EXTRA_DEPS} ${CMAKE_CURRENT_SOURCE_DIR}/docstrings.i)
SWIG_ADD_MODULE(${GDCM_PYTHON_IMPLEMENTATION_NAME} python gdcmswig.i gdcmPythonFilter.cxx)
SWIG_LINK_LIBRARIES(${GDCM_PYTHON_IMPLEMENTATION_NAME} gdcmMEXD gdcmMSFF gdcmIOD)
# Apparently on my UNIX, python module (/usr/lib/pyshared/pythonX.Y/*/*.so) do not explicitely
# link to python libraries...Leave default to always link to python libraries since
# this is required at least on Apple & Win32, but leave the option to advanced user to explicitely
# refuse linking to python libs (set GDCM_NO_PYTHON_LIBS_LINKING to ON):
if(NOT GDCM_NO_PYTHON_LIBS_LINKING)
  SWIG_LINK_LIBRARIES(${GDCM_PYTHON_IMPLEMENTATION_NAME} ${PYTHON_LIBRARY})
endif()
set_property(TARGET ${SWIG_MODULE_${GDCM_PYTHON_IMPLEMENTATION_NAME}_REAL_NAME} PROPERTY NO_SONAME 1)

# Python extension modules on Windows must have the extension ".pyd"
# instead of ".dll" as of Python 2.5.  Older python versions do support
# this suffix.
# http://docs.python.org/whatsnew/ports.html#SECTION0001510000000000000000
# <quote>
# Windows: .dll is no longer supported as a filename extension for extension modules.
# .pyd is now the only filename extension that will be searched for.
# </quote>
if(WIN32 AND NOT CYGWIN)
  set_target_properties(${SWIG_MODULE_${GDCM_PYTHON_IMPLEMENTATION_NAME}_REAL_NAME} PROPERTIES SUFFIX ".pyd")
  # shared libs on windows needs to be fully resolved
  SWIG_LINK_LIBRARIES(${GDCM_PYTHON_IMPLEMENTATION_NAME} ${PYTHON_LIBRARIES})
endif()
set_target_properties(${SWIG_MODULE_${GDCM_PYTHON_IMPLEMENTATION_NAME}_REAL_NAME} PROPERTIES LINK_INTERFACE_LIBRARIES "")

# swig generates a _gdcm.so and a gdcm.py, we need to copy gdcm.py to the proper place:
# gdcm.py is the interface name != implementation name, so we need to keep 'gdcm', so that 'import gdcm'
# from a python script always work
add_custom_command(
  TARGET    ${SWIG_MODULE_${GDCM_PYTHON_IMPLEMENTATION_NAME}_REAL_NAME}
  POST_BUILD
  COMMAND   ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/gdcmswig.py ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}
  COMMAND   ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/gdcm.py ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}
  DEPENDS ${swig_generated_file_fullname} ${CMAKE_CURRENT_BINARY_DIR}/gdcmswig.py ${CMAKE_CURRENT_SOURCE_DIR}/gdcm.py
  COMMENT   "Copy gdcmswig.py into ${LIBRARY_OUTPUT_PATH}"
)

#Module are always place in the library destination
#but for poor win32 user I decided to place them
# right next to the other dlls
if(NOT GDCM_INSTALL_NO_LIBRARIES)
  install_swig_module(${GDCM_PYTHON_IMPLEMENTATION_NAME} Python)
  # the python file is not a dev file, but part of the gdcm module...
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gdcmswig.py
    ${CMAKE_CURRENT_SOURCE_DIR}/gdcm.py
    DESTINATION ${GDCM_INSTALL_PYTHONMODULE_DIR} COMPONENT PythonModule
  )
endif()

# Test that will try to load any class in the target language: python
# it make sure swig was not broken accidentally
if(BUILD_TESTING)
  ADD_PYTHON_TEST(TestWrapPython TestWrap.py ${GDCM_SOURCE_DIR}/Source)
  if(GDCM_DOCUMENTATION)
    ADD_PYTHON_TEST(TestDoxy2SWIGPython doxy2swig.py ${GDCM_BINARY_DIR}/Utilities/doxygen/xml/index.xml ${GDCM_BINARY_DIR}/generated_docstrings.i)
  endif()
endif()

# TODO
# python -c "from distutils import sysconfig; print sysconfig.get_python_lib()"
# /usr/lib/python2.4/site-packages
