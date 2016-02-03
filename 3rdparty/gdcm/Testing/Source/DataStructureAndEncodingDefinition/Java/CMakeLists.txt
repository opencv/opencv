# Define the tests for gdcm-java
set(GDCM_JAVA_TEST_SRCS
  TestReader
  )

# Do not change anything after here:
find_package(Java REQUIRED) # javac, jar
find_package(JNI REQUIRED)
include_directories(
  ${JNI_INCLUDE_PATH}
  )

set(classfilesdep)

foreach(example ${GDCM_JAVA_TEST_SRCS})
  add_custom_command(
    OUTPUT ${EXECUTABLE_OUTPUT_PATH}/${example}.class
    COMMAND ${Java_JAVAC_EXECUTABLE} ARGS ${CMAKE_CURRENT_SOURCE_DIR}/${example}.java -d ${EXECUTABLE_OUTPUT_PATH} -source 1.5 -target 1.5 -classpath ${LIBRARY_OUTPUT_PATH}/gdcm.jar
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${example}.java
    COMMENT "javac ${example}.java"
    )
  list(APPEND classfilesdep ${EXECUTABLE_OUTPUT_PATH}/${example}.class)
  set_source_files_properties(${EXECUTABLE_OUTPUT_PATH}/${example}.class PROPERTIES CLASSPATH "${EXECUTABLE_OUTPUT_PATH}/gdcm.jar")
  ADD_JAVA_TEST(${example}Java ${EXECUTABLE_OUTPUT_PATH}/${example})
endforeach()

# 3. ok now add the target
add_custom_target(GDCMJavaDSEDTests ALL
  DEPENDS ${classfilesdep}
  COMMENT "building gdcm java dsed tests"
)
# make sure gdcm.jar is built
add_dependencies(GDCMJavaDSEDTests GDCMJavaJar)
