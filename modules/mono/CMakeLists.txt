# ----------------------------------------------------------------------------
#  CMake file for mono support
# ----------------------------------------------------------------------------
if(NOT IKVMC_EXECUTABLE)
  ocv_module_disable(mono)
endif()

set(the_description "The mono bindings")
ocv_add_module(mono BINDINGS opencv_java)

# Is there a possibility to include this?
set(JAR_NAME opencv-${OPENCV_VERSION}.jar)
set(JAR_FILE "${OpenCV_BINARY_DIR}/bin/${JAR_NAME}")

set(ASSEMBLY_NAME "opencv-mono.dll")
set(ASSEMBLY_VERSION "${OPENCV_VERSION}.0")
set(ASSEMBLY_FILE "${OpenCV_BINARY_DIR}/bin/${ASSEMBLY_NAME}")

if(WIN32)
  set(ASSEMBLY_INSTALL_DIR mono)
else(WIN32)
  set(ASSEMBLY_INSTALL_DIR share/OpenCV/mono)
endif(WIN32)

add_custom_command(
   OUTPUT ${ASSEMBLY_FILE}
   COMMAND ${IKVMC_EXECUTABLE} -out:${ASSEMBLY_FILE} -target:library -version:${ASSEMBLY_VERSION} ${JAR_FILE}
   DEPENDS ${JAR_FILE}
   COMMENT "Generating ${ASSEMBLY_NAME}")
   
install(FILES ${ASSEMBLY_FILE} DESTINATION ${ASSEMBLY_INSTALL_DIR} COMPONENT main)

# copy ikvm assemblies
# TODO: ikvm-native-* dlls are not managed assemblies. How to deal with them?
set(ikvm_assembly_names "IKVM.Runtime.dll" "IKVM.Runtime.JNI.dll" "IKVM.OpenJDK.Core.dll" "ikvm-native-win32-x86.dll" "ikvm-native-win32-x64.dll")
set(ikvm_assembly_files "")
foreach(assembly_name ${ikvm_assembly_names})
  set(ikvm_assembly_src_file "${IKVM_DIR}/${assembly_name}")
  set(ikvm_assembly_dst_file "${OpenCV_BINARY_DIR}/bin/${assembly_name}")

  add_custom_command(OUTPUT "${ikvm_assembly_dst_file}"
                     COMMAND ${CMAKE_COMMAND} -E copy "${ikvm_assembly_src_file}" "${ikvm_assembly_dst_file}"
                     MAIN_DEPENDENCY "${ikvm_assembly_src_file}"
                     COMMENT "Generating ${ikvm_assembly_dst_file}"
                    )
  install(FILES ${ikvm_assembly_dst_file} DESTINATION ${ASSEMBLY_INSTALL_DIR} COMPONENT main)
  
  list(APPEND ikvm_assembly_files "${ikvm_assembly_dst_file}")
endforeach()

add_custom_target(${the_module} ALL DEPENDS ${ASSEMBLY_FILE} ${ikvm_assembly_files})

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_module} PROPERTIES FOLDER "bindings")
endif()

if(BUILD_TESTS)
  add_subdirectory(test)
endif()