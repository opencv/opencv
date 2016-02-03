set(classpath ${LIBRARY_OUTPUT_PATH}/vtkgdcm.jar${JavaProp_PATH_SEPARATOR}${LIBRARY_OUTPUT_PATH}/gdcm.jar${JavaProp_PATH_SEPARATOR}${VTK_JAVA_JAR}${JavaProp_PATH_SEPARATOR}${GDCM_VTK_JAVA_JAR})

set(classfilesdep)
set(examples
  HelloVTKWorld
  ReadSeriesIntoVTK
  #AWTMedical3
  MPRViewer2
  )
if(VTK_HAS_IMAGERESLICEMAPPER)
  set(examples
    ${examples}
    MPRViewer
    )
endif()
if(VTK_HAS_SMARTVOLUMEMAPPER)
  set(examples
    ${examples}
    MIPViewer
    )
endif()
foreach(example ${examples})
  add_custom_command(
    OUTPUT ${EXECUTABLE_OUTPUT_PATH}/${example}.class
    COMMAND ${Java_JAVAC_EXECUTABLE} ARGS -source 1.5 -target 1.5 -classpath "${classpath}"
    ${CMAKE_CURRENT_SOURCE_DIR}/${example}.java -d ${EXECUTABLE_OUTPUT_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${example}.java
    COMMENT "javac ${example}.java"
    )
  list(APPEND classfilesdep ${EXECUTABLE_OUTPUT_PATH}/${example}.class)
endforeach()

# 3. ok now add the target
add_custom_target(GDCMJavaVTKExample ALL
  DEPENDS ${classfilesdep}
  COMMENT "building gdcm java example"
)
add_dependencies(GDCMJavaVTKExample VTKGDCMJavaJar)

if(BUILD_TESTING)
  if(GDCM_DATA_ROOT)
    #set_source_files_properties(${EXECUTABLE_OUTPUT_PATH}/${test}.class PROPERTIES RUNTIMEPATH "${EXECUTABLE_OUTPUT_PATH}")
    set_source_files_properties(${EXECUTABLE_OUTPUT_PATH}/HelloVTKWorld.class PROPERTIES CLASSPATH "${EXECUTABLE_OUTPUT_PATH}/vtkgdcm.jar:${GDCM_VTK_JAVA_JAR}")
    ADD_JAVA_TEST(TestHelloVTKWorldJava ${EXECUTABLE_OUTPUT_PATH}/HelloVTKWorld ${GDCM_DATA_ROOT}/012345.002.050.dcm ${GDCM_TEMP_DIRECTORY}/HelloVTKWorld.dcm)
    set_source_files_properties(${EXECUTABLE_OUTPUT_PATH}/ReadSeriesIntoVTK.class PROPERTIES CLASSPATH "${EXECUTABLE_OUTPUT_PATH}/vtkgdcm.jar:${GDCM_VTK_JAVA_JAR}")
    ADD_JAVA_TEST(TestReadSeriesIntoVTKJava ${EXECUTABLE_OUTPUT_PATH}/ReadSeriesIntoVTK)
    #set_source_files_properties(${EXECUTABLE_OUTPUT_PATH}/AWTMedical3.class PROPERTIES CLASSPATH "${EXECUTABLE_OUTPUT_PATH}/vtkgdcm.jar:${GDCM_VTK_JAVA_JAR}")
    #ADD_JAVA_TEST(TestAWTMedical3Java ${EXECUTABLE_OUTPUT_PATH}/AWTMedical3)
  endif()
endif()
