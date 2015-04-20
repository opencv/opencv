find_package(Java REQUIRED) # javac, jar
find_package(JNI REQUIRED)
include_directories(
  #${JNI_INCLUDE_PATH}
  ${JAVA_INCLUDE_PATH}
  ${JAVA_INCLUDE_PATH2}
  ${JAVA_AWT_INCLUDE_PATH}
  )

set(classfilesdep)
set(examples
  HelloSimple
  ExtractImageRegion
  DecompressPixmap
  DecompressImage
  ScanDirectory
  ReadFiles
  FileAnonymize
  )
foreach(example ${examples})
  add_custom_command(
    OUTPUT ${EXECUTABLE_OUTPUT_PATH}/${example}.class
    COMMAND ${Java_JAVAC_EXECUTABLE} ARGS -source 1.5 -target 1.5 ${CMAKE_CURRENT_SOURCE_DIR}/${example}.java -d ${EXECUTABLE_OUTPUT_PATH} -classpath ${LIBRARY_OUTPUT_PATH}/gdcm.jar
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${example}.java
    COMMENT "javac ${example}.java"
    )
  list(APPEND classfilesdep ${EXECUTABLE_OUTPUT_PATH}/${example}.class)
endforeach()

# 3. ok now add the target
add_custom_target(GDCMJavaExample ALL
  DEPENDS ${classfilesdep}
  COMMENT "building gdcm java example"
)
# make sure gdcm.jar is built
add_dependencies(GDCMJavaExample GDCMJavaJar)

if(BUILD_TESTING)
  if(GDCM_DATA_ROOT)
    set_source_files_properties(${EXECUTABLE_OUTPUT_PATH}/HelloSimple.class PROPERTIES CLASSPATH "${EXECUTABLE_OUTPUT_PATH}/gdcm.jar")
    ADD_JAVA_TEST(TestHelloSimpleJava ${EXECUTABLE_OUTPUT_PATH}/HelloSimple ${GDCM_DATA_ROOT}/012345.002.050.dcm)
  endif()
endif()
