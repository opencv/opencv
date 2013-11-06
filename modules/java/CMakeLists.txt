# ----------------------------------------------------------------------------
#  CMake file for java support
# ----------------------------------------------------------------------------
if(IOS OR NOT PYTHON_EXECUTABLE OR NOT ANT_EXECUTABLE OR NOT (JNI_FOUND OR (ANDROID AND ANDROID_NATIVE_API_LEVEL GREATER 7)))
  ocv_module_disable(java)
endif()

set(the_description "The java bindings")
ocv_add_module(java BINDINGS opencv_core opencv_imgproc OPTIONAL opencv_objdetect opencv_features2d opencv_video opencv_highgui opencv_ml opencv_calib3d opencv_photo opencv_nonfree opencv_contrib)
ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/generator/src/cpp")

if(NOT ANDROID)
  include_directories(${JNI_INCLUDE_DIRS})
endif()

# output locations
set(JAVA_INSTALL_ROOT "sdk/java")
set(JNI_INSTALL_ROOT  "sdk/native")

# get list of modules to wrap
string(REPLACE "opencv_" "" OPENCV_JAVA_MODULES "${OPENCV_MODULE_${the_module}_REQ_DEPS};${OPENCV_MODULE_${the_module}_OPT_DEPS}")
foreach(module ${OPENCV_JAVA_MODULES})
  if(NOT HAVE_opencv_${module})
    list(REMOVE_ITEM OPENCV_JAVA_MODULES ${module})
  endif()
endforeach()

######################################################################################################################################

# scripts
set(scripts_gen_java "${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_java.py")
set(scripts_hdr_parser "${CMAKE_CURRENT_SOURCE_DIR}/../python/src2/hdr_parser.py")
set(scripts_gen_javadoc "${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_javadoc.py")
set(scripts_rst_parser "${CMAKE_CURRENT_SOURCE_DIR}/generator/rst_parser.py")

# handwritten C/C++ and Java sources
file(GLOB handwrittren_h_sources "${CMAKE_CURRENT_SOURCE_DIR}/generator/src/cpp/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/generator/src/cpp/*.hpp")
file(GLOB handwrittren_cpp_sources "${CMAKE_CURRENT_SOURCE_DIR}/generator/src/cpp/*.cpp")
file(GLOB handwrittren_java_sources "${CMAKE_CURRENT_SOURCE_DIR}/generator/src/java/*.java")
file(GLOB handwrittren_aidl_sources  "${CMAKE_CURRENT_SOURCE_DIR}/generator/src/java/*.aidl")
if(NOT ANDROID)
  ocv_list_filterout(handwrittren_java_sources "/(engine|android)\\\\+")
  ocv_list_filterout(handwrittren_aidl_sources "/(engine|android)\\\\+")
else()
  file(GLOB_RECURSE handwrittren_lib_project_files_rel RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/" "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/*")
  list(REMOVE_ITEM handwrittren_lib_project_files_rel "${ANDROID_MANIFEST_FILE}")
endif()

# headers of OpenCV modules
set(opencv_public_headers "")
foreach(module ${OPENCV_JAVA_MODULES})
  # get list of module headers
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generator/config/${module}.filelist")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/generator/config/${module}.filelist" module_headers)
    ocv_list_add_prefix(module_headers "${OPENCV_MODULE_opencv_${module}_LOCATION}/")
  else()
    set(module_headers "${OPENCV_MODULE_opencv_${module}_HEADERS}")
  endif()
  if(module_headers)
    # C headers must go first
    set(module_headers_cpp ${module_headers})
    ocv_list_filterout(module_headers_cpp "\\\\.h$")
    if(module_headers_cpp)
      list(REMOVE_ITEM module_headers ${module_headers_cpp})
      list(APPEND module_headers ${module_headers_cpp})
    endif()
    unset(module_headers_cpp)

    set(opencv_public_headers_${module} ${module_headers})
    list(APPEND opencv_public_headers ${module_headers})
  else()
    list(REMOVE_ITEM OPENCV_JAVA_MODULES ${module})
  endif()
endforeach()

# rst documentation used for javadoc generation
set(javadoc_rst_sources "")
foreach(module ${OPENCV_JAVA_MODULES})
  file(GLOB_RECURSE refman_rst_headers "${OPENCV_MODULE_opencv_${module}_LOCATION}/*.rst")
  list(APPEND javadoc_rst_sources ${refman_rst_headers})
endforeach()

# generated cpp files
set(generated_cpp_sources "")
foreach(module ${OPENCV_JAVA_MODULES})
  list(APPEND generated_cpp_sources "${CMAKE_CURRENT_BINARY_DIR}/${module}.cpp")
endforeach()

# IMPORTANT: add dependencies to cmake (we should rerun cmake if any of these files is modified)
configure_file("${scripts_gen_java}"   "${OpenCV_BINARY_DIR}/junk/gen_java.junk" COPYONLY)
configure_file("${scripts_hdr_parser}" "${OpenCV_BINARY_DIR}/junk/hdr_parser.junk" COPYONLY)
foreach(header ${opencv_public_headers})
  get_filename_component(header_name "${header}" NAME)
  configure_file("${header}" "${OpenCV_BINARY_DIR}/junk/${header_name}.junk" COPYONLY)
endforeach()

# generated java files
set(generated_java_sources "")
foreach(module ${OPENCV_JAVA_MODULES})
  # first run of gen_java.py (to get list of generated files)
  file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out/")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out")
  execute_process(COMMAND ${PYTHON_EXECUTABLE} "${scripts_gen_java}" "${scripts_hdr_parser}" ${module} ${opencv_public_headers_${module}}
                  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out"
                  OUTPUT_QUIET ERROR_QUIET)
  unset(generated_java_sources_${module})
  file(GLOB_RECURSE generated_java_sources_${module} RELATIVE "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out/" "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out/*.java")
  ocv_list_add_prefix(generated_java_sources_${module} "${CMAKE_CURRENT_BINARY_DIR}/")

  list(APPEND generated_java_sources ${generated_java_sources_${module}})
endforeach()

# generated java files with javadoc
set(documented_java_files "")
foreach(java_file ${generated_java_sources} ${handwrittren_java_sources})
  get_filename_component(java_file_name "${java_file}" NAME_WE)
  list(APPEND documented_java_files "${CMAKE_CURRENT_BINARY_DIR}/${java_file_name}-jdoc.java")
endforeach()

######################################################################################################################################

# step 1: generate .cpp/.java from OpenCV headers
set(step1_depends "${scripts_gen_java}" "${scripts_hdr_parser}" ${opencv_public_headers})
foreach(module ${OPENCV_JAVA_MODULES})
  # second run of gen_java.py (at build time)
  add_custom_command(OUTPUT ${generated_java_sources_${module}} "${CMAKE_CURRENT_BINARY_DIR}/${module}.cpp"
                     COMMAND ${PYTHON_EXECUTABLE} "${scripts_gen_java}" "${scripts_hdr_parser}" ${module} ${opencv_public_headers_${module}}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS "${scripts_gen_java}" "${scripts_hdr_parser}" ${opencv_public_headers_${module}}
                    )
endforeach()

# step 2: generate javadoc comments
set(step2_depends ${step1_depends} ${scripts_gen_javadoc} ${scripts_rst_parser} ${javadoc_rst_sources} ${generated_java_sources} ${handwrittren_java_sources})
string(REPLACE ";" "," OPENCV_JAVA_MODULES_STR "${OPENCV_JAVA_MODULES}")
add_custom_command(OUTPUT ${documented_java_files}
                   COMMAND ${PYTHON_EXECUTABLE} "${scripts_gen_javadoc}" --modules ${OPENCV_JAVA_MODULES_STR} "${CMAKE_CURRENT_SOURCE_DIR}/generator/src/java" "${CMAKE_CURRENT_BINARY_DIR}" 2>"${CMAKE_CURRENT_BINARY_DIR}/get_javadoc_errors.log"
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   DEPENDS ${step2_depends}
                  )

# step 3: copy files to destination
set(step3_input_files ${documented_java_files} ${handwrittren_aidl_sources})
set(copied_files "")
foreach(java_file ${step3_input_files})
  get_filename_component(java_file_name "${java_file}" NAME)
  string(REPLACE "-jdoc.java" ".java" java_file_name "${java_file_name}")
  string(REPLACE "+" "/" java_file_name "${java_file_name}")
  set(output_name "${OpenCV_BINARY_DIR}/src/org/opencv/${java_file_name}")

  add_custom_command(OUTPUT "${output_name}"
                     COMMAND ${CMAKE_COMMAND} -E copy "${java_file}" "${output_name}"
                     MAIN_DEPENDENCY "${java_file}"
                     DEPENDS ${step2_depends}
                     COMMENT "Generating src/org/opencv/${java_file_name}"
                    )
  list(APPEND copied_files "${output_name}")

  if(ANDROID)
    get_filename_component(install_subdir "${java_file_name}" PATH)
    install(FILES "${output_name}" DESTINATION "${JAVA_INSTALL_ROOT}/src/org/opencv/${install_subdir}" COMPONENT main)
  endif()
endforeach()

if(ANDROID)
  set(android_copied_files "")
  set(android_step3_input_files "")
  foreach(file ${handwrittren_lib_project_files_rel})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${file}" "${OpenCV_BINARY_DIR}/${file}" @ONLY)
    list(APPEND android_copied_files "${OpenCV_BINARY_DIR}/${file}")
    list(APPEND android_step3_input_files "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${file}")

    if(NOT file MATCHES "jni/.+")
      get_filename_component(install_subdir "${file}" PATH)
      install(FILES "${OpenCV_BINARY_DIR}/${file}" DESTINATION "${JAVA_INSTALL_ROOT}/${install_subdir}" COMPONENT main)
    endif()
  endforeach()

  # library project jni sources (nothing really depends on them so we will not add them to step3_input_files)
  foreach(jni_file ${handwrittren_cpp_sources} ${handwrittren_h_sources} ${generated_cpp_sources})
    get_filename_component(jni_file_name "${jni_file}" NAME)
    add_custom_command(OUTPUT "${OpenCV_BINARY_DIR}/jni/${jni_file_name}"
                       COMMAND ${CMAKE_COMMAND} -E copy "${jni_file}" "${OpenCV_BINARY_DIR}/jni/${jni_file_name}"
                       DEPENDS "${jni_file}" ${java_hdr_deps}
                       COMMENT "Generating jni/${jni_file_name}"
                      )
    list(APPEND android_copied_files "${OpenCV_BINARY_DIR}/jni/${jni_file_name}")
  endforeach()
endif(ANDROID)

# step 3.5: generate Android library project
if(ANDROID AND ANDROID_EXECUTABLE)
  set(lib_target_files ${ANDROID_LIB_PROJECT_FILES})
  ocv_list_add_prefix(lib_target_files "${OpenCV_BINARY_DIR}/")

  android_get_compatible_target(lib_target_sdk_target ${ANDROID_NATIVE_API_LEVEL} ${ANDROID_SDK_TARGET} 14)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${ANDROID_MANIFEST_FILE}" "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" @ONLY)

  add_custom_command(OUTPUT ${lib_target_files} "${OpenCV_BINARY_DIR}/${ANDROID_MANIFEST_FILE}"
                     COMMAND ${CMAKE_COMMAND} -E remove ${lib_target_files}
                     COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" "${OpenCV_BINARY_DIR}/${ANDROID_MANIFEST_FILE}"
                     COMMAND ${ANDROID_EXECUTABLE} --silent create lib-project --path \"${OpenCV_BINARY_DIR}\" --target \"${lib_target_sdk_target}\" --name OpenCV --package org.opencv 2>\"${CMAKE_CURRENT_BINARY_DIR}/create_lib_project.log\"
                     COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" "${OpenCV_BINARY_DIR}/${ANDROID_MANIFEST_FILE}"
                     MAIN_DEPENDENCY "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}"
                     DEPENDS ${android_step3_input_files} ${android_copied_files}
                     COMMENT "Generating OpenCV Android library project. SDK target: ${lib_target_sdk_target}"
                    )
  list(APPEND copied_files ${lib_target_files} "${OpenCV_BINARY_DIR}/${ANDROID_MANIFEST_FILE}")
  list(APPEND step3_input_files "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}")

  install(FILES "${OpenCV_BINARY_DIR}/${ANDROID_PROJECT_PROPERTIES_FILE}" DESTINATION ${JAVA_INSTALL_ROOT} COMPONENT main)
  install(FILES "${OpenCV_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" DESTINATION ${JAVA_INSTALL_ROOT} COMPONENT main)
  # creating empty 'gen' and 'res' folders
  install(CODE "MAKE_DIRECTORY(\"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${JAVA_INSTALL_ROOT}/gen\")" COMPONENT main)
  install(CODE "MAKE_DIRECTORY(\"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${JAVA_INSTALL_ROOT}/res\")" COMPONENT main)
endif(ANDROID AND ANDROID_EXECUTABLE)

set(step3_depends ${step2_depends} ${step3_input_files} ${copied_files})

if(ANDROID)
  set(LIB_NAME_SUFIX "")
else()
  set(LIB_NAME_SUFIX "${OPENCV_VERSION_MAJOR}${OPENCV_VERSION_MINOR}${OPENCV_VERSION_PATCH}")
endif()

# step 4: build jar
if(ANDROID)
  set(JAR_FILE "${OpenCV_BINARY_DIR}/bin/classes.jar")
  if(ANDROID_TOOLS_Pkg_Revision GREATER 13)
    # build the library project
    # normally we should do this after a native part, but for a library project we can build the java part first
    add_custom_command(OUTPUT "${JAR_FILE}" "${JAR_FILE}.dephelper"
                       COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug
                       COMMAND ${CMAKE_COMMAND} -E touch "${JAR_FILE}.dephelper" # can not rely on classes.jar because different versions of SDK update timestamp at different times
                       WORKING_DIRECTORY "${OpenCV_BINARY_DIR}"
                       DEPENDS ${step3_depends}
                       COMMENT "Building OpenCV Android library project"
                      )
  else()
    # ditto
    add_custom_command(OUTPUT "${JAR_FILE}" "${JAR_FILE}.dephelper"
                       COMMAND ${CMAKE_COMMAND} -E touch "${JAR_FILE}"
                       COMMAND ${CMAKE_COMMAND} -E touch "${JAR_FILE}.dephelper"
                       WORKING_DIRECTORY "${OpenCV_BINARY_DIR}"
                       DEPENDS ${step3_depends}
                       COMMENT ""
                      )
  endif()
else(ANDROID)
  set(JAR_NAME opencv-${LIB_NAME_SUFIX}.jar)
  set(JAR_FILE "${OpenCV_BINARY_DIR}/bin/${JAR_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/build.xml.in" "${OpenCV_BINARY_DIR}/build.xml" IMMEDIATE @ONLY)
  list(APPEND step3_depends "${OpenCV_BINARY_DIR}/build.xml")

  add_custom_command(OUTPUT "${JAR_FILE}" "${JAR_FILE}.dephelper"
                     COMMAND ${ANT_EXECUTABLE} -q -noinput -k jar
                     COMMAND ${CMAKE_COMMAND} -E touch "${JAR_FILE}.dephelper"
                     WORKING_DIRECTORY "${OpenCV_BINARY_DIR}"
                     DEPENDS ${step3_depends}
                     COMMENT "Generating ${JAR_NAME}"
                    )

  if(WIN32)
    set(JAR_INSTALL_DIR java)
  else(WIN32)
    set(JAR_INSTALL_DIR share/OpenCV/java)
  endif(WIN32)
  install(FILES ${JAR_FILE} DESTINATION ${JAR_INSTALL_DIR} COMPONENT main)
endif(ANDROID)

# step 5: build native part

# workarounding lack of `__attribute__ ((visibility("default")))` in jni_md.h/JNIEXPORT
string(REPLACE "-fvisibility=hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(${the_module} SHARED ${handwrittren_h_sources} ${handwrittren_cpp_sources} ${generated_cpp_sources}
                                 ${copied_files}
                                "${JAR_FILE}" "${JAR_FILE}.dephelper")
set_target_properties(${the_module} PROPERTIES COMPILE_DEFINITIONS OPENCV_NOSTL)

if(BUILD_FAT_JAVA_LIB)
  set(__deps ${OPENCV_MODULE_${the_module}_DEPS} ${OPENCV_MODULES_BUILD})
  foreach(m ${OPENCV_MODULES_BUILD}) # filterout INTERNAL (like opencv_ts) and BINDINGS (like opencv_python) modules
    ocv_assert(DEFINED OPENCV_MODULE_${m}_CLASS)
    if(NOT OPENCV_MODULE_${m}_CLASS STREQUAL "PUBLIC")
      list(REMOVE_ITEM __deps ${m})
    endif()
  endforeach()
  ocv_list_unique(__deps)
  set(__extradeps ${__deps})
  ocv_list_filterout(__extradeps "^opencv_")
  if(__extradeps)
    list(REMOVE_ITEM __deps ${__extradeps})
  endif()
  if(APPLE)
    foreach(_dep ${__deps})
      target_link_libraries(${the_module} -Wl,-force_load "${_dep}")
    endforeach()
  else()
    target_link_libraries(${the_module} -Wl,-whole-archive ${__deps} -Wl,-no-whole-archive)
  endif()
  target_link_libraries(${the_module} ${__extradeps} ${OPENCV_LINKER_LIBS})
else()
  target_link_libraries(${the_module} ${OPENCV_MODULE_${the_module}_DEPS} ${OPENCV_LINKER_LIBS})
endif()

if(ANDROID)
  target_link_libraries(${the_module} jnigraphics) # for Mat <=> Bitmap converters

  # force strip library after the build command
  # because samples and tests will make a copy of the library before install
  get_target_property(__opencv_java_location ${the_module} LOCATION)
  # Turn off stripping in debug build
  if ( NOT (CMAKE_BUILD_TYPE MATCHES "Debug"))
    add_custom_command(TARGET ${the_module} POST_BUILD COMMAND ${CMAKE_STRIP} --strip-unneeded "${__opencv_java_location}")
  endif()
endif()

# Additional target properties
set_target_properties(${the_module} PROPERTIES
    OUTPUT_NAME "${the_module}${LIB_NAME_SUFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    INSTALL_NAME_DIR ${OPENCV_LIB_INSTALL_PATH}
    LINK_INTERFACE_LIBRARIES ""
    )

if(WIN32)
  set_target_properties(${the_module} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_module} PROPERTIES FOLDER "bindings")
endif()

if(ANDROID)
  ocv_install_target(${the_module} EXPORT OpenCVModules
          LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main
          ARCHIVE DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main)
else()
  if(NOT INSTALL_CREATE_DISTRIB)
    ocv_install_target(${the_module} EXPORT OpenCVModules
            RUNTIME DESTINATION ${JAR_INSTALL_DIR} COMPONENT main
            LIBRARY DESTINATION ${JAR_INSTALL_DIR} COMPONENT main)
  else()
    ocv_install_target(${the_module} EXPORT OpenCVModules
            RUNTIME DESTINATION ${JAR_INSTALL_DIR}/${OpenCV_ARCH} COMPONENT main
            LIBRARY DESTINATION ${JAR_INSTALL_DIR}/${OpenCV_ARCH} COMPONENT main)
  endif()
endif()

######################################################################################################################################

if(BUILD_TESTS)
  if(ANDROID)
    add_subdirectory(android_test)
  else()
    add_subdirectory(test)
  endif()
endif()
