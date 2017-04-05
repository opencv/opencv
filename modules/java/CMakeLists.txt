# ----------------------------------------------------------------------------
#  CMake file for java support
# ----------------------------------------------------------------------------
if(APPLE_FRAMEWORK OR WINRT OR NOT PYTHON_DEFAULT_AVAILABLE OR NOT ANT_EXECUTABLE
    OR NOT (JNI_FOUND OR (ANDROID AND ANDROID_NATIVE_API_LEVEL GREATER 7))
    OR BUILD_opencv_world
    )
  ocv_module_disable(java)
endif()

if(EXISTS ${CMAKE_BINARY_DIR}/src)
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_BINARY_DIR}/src")
endif()
if(EXISTS ${CMAKE_BINARY_DIR}/gen)
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_BINARY_DIR}/gen")
endif()

set(the_description "The java bindings")
ocv_add_module(java BINDINGS opencv_core opencv_imgproc)
ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/generator/src/cpp")
ocv_module_include_directories("${OpenCV_SOURCE_DIR}/include")

if(NOT ANDROID)
  include_directories(${JNI_INCLUDE_DIRS})
endif()

# output locations
set(JAVA_INSTALL_ROOT "sdk/java")
set(JNI_INSTALL_ROOT  "sdk/native")

# get list of modules to wrap
# message(STATUS "Wrapped in java:")
set(OPENCV_JAVA_MODULES)
foreach(m ${OPENCV_MODULES_BUILD})
  if (";${OPENCV_MODULE_${m}_WRAPPERS};" MATCHES ";java;" AND HAVE_${m})
    string(REPLACE "opencv_" "" m ${m})
    list(APPEND OPENCV_JAVA_MODULES ${m})
    # message(STATUS "\topencv_${m}")
  endif()
endforeach()

######################################################################################################################################

# UTILITY: make C headers go first
macro(sort_headers_c_cpp __lst)
  set(__cpp ${${__lst}})
  ocv_list_filterout(__cpp "\\\\.h$")
  if(__cpp)
    list(REMOVE_ITEM ${__lst} ${__cpp})
    list(APPEND ${__lst} ${__cpp})
  endif()
  unset(__cpp)
endmacro()

# UTILITY: force cmake to rerun when files from list have changed
macro(add_cmake_dependencies)
  foreach (f ${ARGN})
    get_filename_component(f_name "${f}" NAME)
    configure_file(${f} ${OpenCV_BINARY_DIR}/junk/${f_name}.junk COPYONLY)
  endforeach()
endmacro()

# UTILITY: glob specific sources and append them to list (type is in H, CPP, JAVA, AIDL)
macro(glob_more_specific_sources _type _root _output)
  unset(_masks)
  if(${_type} STREQUAL "H")
    set(_masks "${_root}/src/cpp/*.h" "${root}/src/cpp/*.hpp")
  elseif(${_type} STREQUAL "CPP")
    set(_masks "${_root}/src/cpp/*.cpp")
  elseif(${_type} STREQUAL "JAVA")
    set(_masks "${_root}/src/java/*.java")
  elseif(${_type} STREQUAL "AIDL")
    set(_masks "${_root}/src/java/*.aidl")
  endif()
  if (_masks)
    file(GLOB _result ${_masks})
    list(APPEND ${_output} ${_result})
  else()
    message(WARNING "Bad argument passed to macro: skipped")
  endif()
endmacro()

# UTILITY: copy common java test files and add them to _deps
# copy_common_tests(<source-folder> <destination-folder> <variable-to-store-deps>)
macro(copy_common_tests _src_location _dst_location _deps)
  set(_src ${${_src_location}})
  set(_dst ${${_dst_location}})
  file(GLOB_RECURSE _files RELATIVE "${_src}" "${_src}/res/*" "${_src}/src/*")
  foreach(f ${_files})
    add_custom_command(
        OUTPUT "${_dst}/${f}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}/${f}" "${_dst}/${f}"
        MAIN_DEPENDENCY "${_src}/${f}"
        COMMENT "Copying ${f}")
    list(APPEND ${_deps} "${_src}/${f}" "${_dst}/${f}")
  endforeach()
  unset(_files)
  unset(_src)
  unset(_dst)
endmacro()

# UTILITY: copy all java tests for specific module and add them to _deps
# copy_modules_tests(<modules-list> <destination-folder> <variable-to-store-deps>)
macro(copy_modules_tests _modules _dst_location _deps)
  set(_dst ${${_dst_location}})
  foreach(module ${${_modules}})
    set(_src "${OPENCV_MODULE_opencv_${module}_LOCATION}/misc/java/test")
    set(_tree "src/org/opencv/test/${module}")
    file(GLOB _files RELATIVE "${_src}" "${_src}/*.java")
    foreach (f ${_files})
      add_custom_command(
          OUTPUT "${_dst}/${_tree}/${f}"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}/${f}" "${_dst}/${_tree}/${f}"
          MAIN_DEPENDENCY "${_src}/${f}"
          COMMENT "Copying ${f}")
      list(APPEND ${_deps} "${_src}/${f}" "${_dst}/${_tree}/${f}")
    endforeach()
    unset(_files)
    unset(_src)
    unset(_tree)
  endforeach()
  unset(_dst)
endmacro()

######################################################################################################################################

# scripts
set(scripts_gen_java "${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_java.py")
set(scripts_hdr_parser "${CMAKE_CURRENT_SOURCE_DIR}/../python/src2/hdr_parser.py")

# directory to store temporary files generated on first gen_java.py run
set(probe_dir "${CMAKE_CURRENT_BINARY_DIR}/test_gen")

# handwritten C/C++ and Java sources
glob_more_specific_sources(H "${CMAKE_CURRENT_SOURCE_DIR}/generator" handwritten_h_sources)
glob_more_specific_sources(CPP "${CMAKE_CURRENT_SOURCE_DIR}/generator" handwritten_cpp_sources)
glob_more_specific_sources(JAVA "${CMAKE_CURRENT_SOURCE_DIR}/generator" handwritten_java_sources)
glob_more_specific_sources(AIDL "${CMAKE_CURRENT_SOURCE_DIR}/generator" handwritten_aidl_sources)

# headers of OpenCV modules
set(opencv_public_headers "")
set(generated_cpp_sources "")
set(generated_java_sources "")
foreach(module ${OPENCV_JAVA_MODULES})
  set(module_java_dir "${OPENCV_MODULE_opencv_${module}_LOCATION}/misc/java")
  set(custom_header_list "${module_java_dir}/filelist")
  if(EXISTS "${custom_header_list}")
    file(STRINGS "${custom_header_list}" module_headers)
    ocv_list_add_prefix(module_headers "${OPENCV_MODULE_opencv_${module}_LOCATION}/")
  else()
    set(module_headers "${OPENCV_MODULE_opencv_${module}_HEADERS}")
  endif()

  if(NOT module_headers)
    message(WARNING "Module ${module} does not have headers to wrap for java")
  endif()

  sort_headers_c_cpp(module_headers)

  set(opencv_public_headers_${module} ${module_headers})
  list(APPEND opencv_public_headers ${module_headers})
  list(APPEND generated_cpp_sources "${CMAKE_CURRENT_BINARY_DIR}/gen/${module}.cpp")

  include_directories("${module_java_dir}/src/cpp")

  foreach(m ${OPENCV_MODULE_opencv_${module}_DEPS})
    set(common_header_list "${OPENCV_MODULE_${m}_LOCATION}/misc/java/filelist_common")
    if(EXISTS "${common_header_list}")
      file(STRINGS "${common_header_list}" __headers)
      ocv_list_add_prefix(__headers "${OPENCV_MODULE_${m}_LOCATION}/")
      list(APPEND opencv_java_common_headers_${module} ${__headers})
    endif()
  endforeach()

  glob_more_specific_sources(H "${module_java_dir}" handwritten_h_sources)
  glob_more_specific_sources(CPP "${module_java_dir}" handwritten_cpp_sources)
  glob_more_specific_sources(JAVA "${module_java_dir}" handwritten_java_sources)
  glob_more_specific_sources(AIDL "${module_java_dir}" handwritten_aidl_sources)

  # first run of gen_java.py (to get list of generated files)
  file(REMOVE_RECURSE "${probe_dir}")
  file(MAKE_DIRECTORY "${probe_dir}")
  execute_process(COMMAND ${PYTHON_DEFAULT_EXECUTABLE} "${scripts_gen_java}" "${scripts_hdr_parser}" ${module} ${opencv_public_headers_${module}} "--common" ${opencv_java_common_headers_${module}}
                  WORKING_DIRECTORY "${probe_dir}"
                  OUTPUT_QUIET ERROR_QUIET)
  file(GLOB_RECURSE generated_java_sources_${module} RELATIVE "${probe_dir}" "${probe_dir}/*.java")
  ocv_list_add_prefix(generated_java_sources_${module} "${CMAKE_CURRENT_BINARY_DIR}/gen/")
  list(APPEND generated_java_sources ${generated_java_sources_${module}})
endforeach()

file(REMOVE_RECURSE "${probe_dir}")

if(NOT ANDROID)
  ocv_list_filterout(handwritten_java_sources "/(engine|android)\\\\+")
  ocv_list_filterout(handwritten_aidl_sources "/(engine|android)\\\\+")
else()
  file(GLOB_RECURSE handwrittren_lib_project_files_rel RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/" "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/*")
  list(REMOVE_ITEM handwrittren_lib_project_files_rel "${ANDROID_MANIFEST_FILE}")

  #  calc default SDK Target
  android_get_compatible_target(android_sdk_target ${ANDROID_NATIVE_API_LEVEL} ${ANDROID_SDK_TARGET} 11)
  string(REGEX REPLACE "android-" "" android_sdk_target_num ${android_sdk_target})
  # filter out
  if( (ANDROID_SDK_TARGET AND ANDROID_SDK_TARGET LESS 21) OR (android_sdk_target_num LESS 21) )
    message(STATUS "[OpenCV for Android SDK]: A new OpenGL Camera Bridge (CameraGLSurfaceView, CameraGLRendererBase, CameraRenderer, Camera2Renderer) is disabled, because ANDROID_SDK_TARGET (${android_sdk_target_num}) < 21")
    ocv_list_filterout(handwritten_java_sources "android\\\\+CameraGL")
    ocv_list_filterout(handwritten_java_sources "android\\\\+Camera.?Renderer")
  endif()
endif()

# IMPORTANT: add dependencies to cmake (we should rerun cmake if any of these files is modified)
add_cmake_dependencies(${scripts_gen_java} ${scripts_hdr_parser} ${opencv_public_headers})

######################################################################################################################################

# step 1: generate .cpp/.java from OpenCV headers
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gen")
set(step1_depends "${scripts_gen_java}" "${scripts_hdr_parser}" ${opencv_public_headers})
foreach(module ${OPENCV_JAVA_MODULES})
  # second run of gen_java.py (at build time)
  add_custom_command(OUTPUT ${generated_java_sources_${module}} "${CMAKE_CURRENT_BINARY_DIR}/gen/${module}.cpp"
                     COMMAND ${PYTHON_DEFAULT_EXECUTABLE} "${scripts_gen_java}" "${scripts_hdr_parser}" ${module} ${opencv_public_headers_${module}} "--common" ${opencv_java_common_headers_${module}}
                     WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gen/"
                     DEPENDS "${scripts_gen_java}" "${scripts_hdr_parser}" ${opencv_public_headers_${module}} ${opencv_java_common_headers_${module}}
                    )
endforeach()

# step 2: TODO: generate documentation somewhere

# step 3: copy files to destination
set(step3_input_files ${generated_java_sources} ${handwritten_java_sources} ${handwritten_aidl_sources})
set(copied_files "")
foreach(java_file ${step3_input_files})
  get_filename_component(java_file_name "${java_file}" NAME)
  string(REPLACE "-jdoc.java" ".java" java_file_name "${java_file_name}")
  string(REPLACE "+" "/" java_file_name "${java_file_name}")
  set(output_name "${OpenCV_BINARY_DIR}/src/org/opencv/${java_file_name}")
  add_custom_command(OUTPUT "${output_name}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${java_file}" "${output_name}"
                     MAIN_DEPENDENCY "${java_file}"
                     DEPENDS ${step1_depends} ${generated_java_sources} ${handwritten_java_sources}
                     COMMENT "Generating src/org/opencv/${java_file_name}"
                    )
  list(APPEND copied_files "${output_name}")

  if(ANDROID)
    get_filename_component(install_subdir "${java_file_name}" PATH)
    install(FILES "${output_name}" DESTINATION "${JAVA_INSTALL_ROOT}/src/org/opencv/${install_subdir}" COMPONENT java)
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
      install(FILES "${OpenCV_BINARY_DIR}/${file}" DESTINATION "${JAVA_INSTALL_ROOT}/${install_subdir}" COMPONENT java)
    endif()
  endforeach()

  # library project jni sources (nothing really depends on them so we will not add them to step3_input_files)
  foreach(jni_file ${handwritten_cpp_sources} ${handwritten_h_sources} ${generated_cpp_sources})
    get_filename_component(jni_file_name "${jni_file}" NAME)
    add_custom_command(OUTPUT "${OpenCV_BINARY_DIR}/jni/${jni_file_name}"
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different "${jni_file}" "${OpenCV_BINARY_DIR}/jni/${jni_file_name}"
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
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" "${OpenCV_BINARY_DIR}/${ANDROID_MANIFEST_FILE}"
                     COMMAND ${ANDROID_EXECUTABLE} --silent create lib-project --path \"${OpenCV_BINARY_DIR}\" --target \"${lib_target_sdk_target}\" --name OpenCV --package org.opencv 2>\"${CMAKE_CURRENT_BINARY_DIR}/create_lib_project.log\"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" "${OpenCV_BINARY_DIR}/${ANDROID_MANIFEST_FILE}"
                     MAIN_DEPENDENCY "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}"
                     DEPENDS ${android_step3_input_files} ${android_copied_files}
                     COMMENT "Generating OpenCV Android library project. SDK target: ${lib_target_sdk_target}"
                    )
  list(APPEND copied_files ${lib_target_files} "${OpenCV_BINARY_DIR}/${ANDROID_MANIFEST_FILE}")
  list(APPEND step3_input_files "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}")

  install(FILES "${OpenCV_BINARY_DIR}/${ANDROID_PROJECT_PROPERTIES_FILE}" DESTINATION ${JAVA_INSTALL_ROOT} COMPONENT java)
  install(FILES "${OpenCV_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" DESTINATION ${JAVA_INSTALL_ROOT} COMPONENT java)

  # creating empty 'gen' and 'res' folders
  file(MAKE_DIRECTORY ${OpenCV_BINARY_DIR}/sdk/java/gen ${OpenCV_BINARY_DIR}/sdk/java/res )

  install(DIRECTORY ${OpenCV_BINARY_DIR}/sdk/java/gen DESTINATION ${JAVA_INSTALL_ROOT}/ COMPONENT java)
  install(DIRECTORY ${OpenCV_BINARY_DIR}/sdk/java/res DESTINATION ${JAVA_INSTALL_ROOT}/ COMPONENT java)
endif(ANDROID AND ANDROID_EXECUTABLE)

set(step3_depends ${step2_depends} ${step3_input_files} ${copied_files})

if(ANDROID)
  set(LIB_NAME_SUFIX "${OPENCV_VERSION_MAJOR}")
else()
  set(LIB_NAME_SUFIX "${OPENCV_VERSION_MAJOR}${OPENCV_VERSION_MINOR}${OPENCV_VERSION_PATCH}")
endif()

file(MAKE_DIRECTORY "${OpenCV_BINARY_DIR}/bin")

# step 4: build jar
if(ANDROID)
  set(JAR_FILE "${OpenCV_BINARY_DIR}/bin/classes.jar")
  if(ANDROID_TOOLS_Pkg_Revision GREATER 13)
    # build the library project
    # normally we should do this after a native part, but for a library project we can build the java part first
    add_custom_command(OUTPUT "${JAR_FILE}" "${JAR_FILE}.dephelper"
                       COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug -Djava.target=1.6 -Djava.source=1.6
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
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/build.xml.in" "${OpenCV_BINARY_DIR}/build.xml" @ONLY)
  list(APPEND step3_depends "${OpenCV_BINARY_DIR}/build.xml")

  add_custom_command(OUTPUT "${JAR_FILE}" "${JAR_FILE}.dephelper"
                     COMMAND ${ANT_EXECUTABLE} -q -noinput -k jar javadoc
                     COMMAND ${CMAKE_COMMAND} -E touch "${JAR_FILE}.dephelper"
                     WORKING_DIRECTORY "${OpenCV_BINARY_DIR}"
                     DEPENDS ${step3_depends}
                     COMMENT "Generating ${JAR_NAME}"
                    )

  install(FILES ${JAR_FILE} OPTIONAL DESTINATION ${OPENCV_JAR_INSTALL_PATH} COMPONENT java)
endif(ANDROID)

# step 5: build native part

# workarounding lack of `__attribute__ ((visibility("default")))` in jni_md.h/JNIEXPORT
string(REPLACE "-fvisibility=hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-const-variable)

ocv_add_library(${the_module} SHARED ${handwritten_h_sources} ${handwritten_cpp_sources} ${generated_cpp_sources}
                                 ${copied_files}
                                "${JAR_FILE}" "${JAR_FILE}.dephelper")

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
      ocv_target_link_libraries(${the_module} -Wl,-force_load "${_dep}")
    endforeach()
  else()
    ocv_target_link_libraries(${the_module} -Wl,-whole-archive ${__deps} -Wl,-no-whole-archive)
  endif()
  ocv_target_link_libraries(${the_module} ${__extradeps} ${OPENCV_LINKER_LIBS})
else()
  ocv_target_link_libraries(${the_module} ${OPENCV_MODULE_${the_module}_DEPS} ${OPENCV_LINKER_LIBS})
endif()

# Additional target properties
set_target_properties(${the_module} PROPERTIES
    OUTPUT_NAME "${the_module}${LIB_NAME_SUFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    LINK_INTERFACE_LIBRARIES ""
    )

if(ANDROID)
  ocv_target_link_libraries(${the_module} jnigraphics) # for Mat <=> Bitmap converters
  ocv_target_link_libraries(${the_module} LINK_INTERFACE_LIBRARIES ${OPENCV_LINKER_LIBS} jnigraphics)

  # force strip library after the build command
  # because samples and tests will make a copy of the library before install
  get_target_property(__opencv_java_location ${the_module} LOCATION)
  # Turn off stripping in debug build
  if ( NOT (CMAKE_BUILD_TYPE MATCHES "Debug"))
    add_custom_command(TARGET ${the_module} POST_BUILD COMMAND ${CMAKE_STRIP} --strip-unneeded "${__opencv_java_location}")
  endif()
endif()

if(WIN32)
  set_target_properties(${the_module} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_module} PROPERTIES FOLDER "bindings")
endif()

if(ANDROID)
  ocv_install_target(${the_module} OPTIONAL EXPORT OpenCVModules
          LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT java
          ARCHIVE DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT java)
else()
  if(NOT INSTALL_CREATE_DISTRIB)
    ocv_install_target(${the_module} OPTIONAL EXPORT OpenCVModules
            RUNTIME DESTINATION ${OPENCV_JAR_INSTALL_PATH} COMPONENT java
            LIBRARY DESTINATION ${OPENCV_JAR_INSTALL_PATH} COMPONENT java)
  else()
    ocv_install_target(${the_module} OPTIONAL EXPORT OpenCVModules
            RUNTIME DESTINATION ${OPENCV_JAR_INSTALL_PATH}/${OpenCV_ARCH} COMPONENT java
            LIBRARY DESTINATION ${OPENCV_JAR_INSTALL_PATH}/${OpenCV_ARCH} COMPONENT java)
  endif()
endif()

######################################################################################################################################

if(BUILD_TESTS)
  if(ANDROID)
    add_subdirectory(android_test)
  else()
    add_subdirectory(pure_test)
  endif()
endif()
