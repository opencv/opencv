# TODO:
# SWIG is really a pain in the neck to use, a better alternative is Py++ which is using
# gccxml for the C++ parser and allow a full ANSI C++ support
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})
option(GDCM_AUTOLOAD_GDCMJNI "Automatically load gdcmjni" ON)

include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/MessageExchangeDefinition"
)

# $ export JAVA_HOME=/usr/lib/j2sdk1.6-sun/
# $ export JAVA_HOME=/usr/lib/jvm/java-1.5.0-sun/
find_package(Java 1.5 REQUIRED) # javac, jar
find_package(JNI REQUIRED)
include_directories(
  ${JNI_INCLUDE_DIRS}
  )
set_source_files_properties(gdcm.i PROPERTIES CPLUSPLUS ON)
#include(${GDCM_SOURCE_DIR}/CMake/UseSWIGExtra.cmake)

set(GDCM_JAVA_HOME ${CMAKE_CURRENT_BINARY_DIR}/java/gdcm)
file(MAKE_DIRECTORY ${GDCM_JAVA_HOME})
set(CMAKE_SWIG_OUTDIR "${GDCM_JAVA_HOME}")

set(CMAKE_SWIG_FLAGS "-package gdcm")
separate_arguments(CMAKE_SWIG_FLAGS)
# http://www.debian.org/doc/packaging-manuals/java-policy/x105.html
SWIG_ADD_MODULE(gdcmjni java gdcm.i)
SWIG_LINK_LIBRARIES(gdcmjni gdcmMSFF
  gdcmMEXD
  # There is no real point in linking explicitely to jni libraries. This is a java module
  # it could contains undefined symbols after all.
  #${JNI_LIBRARIES}
)
set_target_properties(${SWIG_MODULE_gdcmjni_REAL_NAME} PROPERTIES LINK_INTERFACE_LIBRARIES "")
set_property(TARGET ${SWIG_MODULE_gdcmjni_REAL_NAME} PROPERTY NO_SONAME 1)

# swig-java dummy run:
# the following execute_process is a dummy run and generate all *.java files in a directory
# we can then glob all *.java to build rules in case the java wrapper becomes broken
#execute_process(COMMAND ${SWIG_EXECUTABLE}
#  -java
#  -I${GDCM_BINARY_DIR}/Source/Common
#  -I${GDCM_SOURCE_DIR}/Source/Common
#  -I${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition
#  -I${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat
#  -I${GDCM_SOURCE_DIR}/Source/DataDictionary
#  -c++
#  -o dummy.o
#  ${CMAKE_CURRENT_SOURCE_DIR}/gdcm.i
#  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#  RESULT_VARIABLE res
#)
#message("res:${res}")
#file(GLOB javafiles "${CMAKE_CURRENT_BINARY_DIR}/*.java")
#message("javafile:${javafiles}")

if(CMAKE_COMPILER_IS_GNUCXX)
  # http://www.swig.org/Doc2.0/Java.html#Java_typemaps_c_to_java_types
  set_source_files_properties(
    ${swig_generated_file_fullname}
    PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing")
endif()

add_custom_command(
  OUTPUT ${LIBRARY_OUTPUT_PATH}/gdcm.jar
# 1. run the custom command only when the gdcmJAVA_wrap.cxx has been generated
# (which means all *.java should be there) Some compiler are picky about Java
# version. For instance covariant return type was only added to Java 5, so to
# please compiler such as ecj (Eclipse Java Compiler 0.894_R34x) explicitly
# state we want 1.5 version to compile gdcm::ImageToImageFilter::GetOutput()
# const as covariant return type
  COMMAND ${Java_JAVAC_EXECUTABLE} ARGS -source 1.5 -target 1.5 "gdcm/*.java" #${javafiles}
# 2. now that the *.class have been generated construct the jar file. We can
# only rely on the gdcm.java / gdcm.class to build dependencie, I am pretty
# sure it will break parallel builds... oh well
  COMMAND ${Java_JAR_EXECUTABLE} ARGS cvf ${LIBRARY_OUTPUT_PATH}/gdcm.jar gdcm/*.class
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/java
  DEPENDS "${swig_generated_file_fullname}"
  COMMENT "javac *.java; jar cvf -> gdcm.jar"
)

# 3. ok now add the target
add_custom_target(GDCMJavaJar ALL
  DEPENDS ${LIBRARY_OUTPUT_PATH}/gdcm.jar
  COMMENT "building gdcm.jar"
)

if(NOT GDCM_INSTALL_NO_LIBRARIES)
  install_swig_module(gdcmjni Java)
  # because gdcm.jar is constructed with custom commands, it need the INSTALL(FILES signature:
  install(FILES ${LIBRARY_OUTPUT_PATH}/gdcm.jar
    DESTINATION ${GDCM_INSTALL_JARMODULE_DIR} COMPONENT JavaModule
  )
endif()
