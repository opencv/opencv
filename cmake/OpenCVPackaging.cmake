# Use patched version of CPACK to build accurate set of Debian packages
# https://github.com/asmorkalov/CMake/tree/deb_generator_improvement

if(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
set(CPACK_set_DESTDIR "on")

if(NOT OPENCV_CUSTOM_PACKAGE_INFO)
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open Computer Vision Library")
  set(CPACK_PACKAGE_DESCRIPTION
"OpenCV (Open Source Computer Vision Library) is an open source computer vision
and machine learning software library. OpenCV was built to provide a common
infrastructure for computer vision applications and to accelerate the use of
machine perception in the commercial products. Being a BSD-licensed product,
OpenCV makes it easy for businesses to utilize and modify the code.")
  set(CPACK_PACKAGE_VENDOR "OpenCV Foundation")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  set(CPACK_PACKAGE_CONTACT "OpenCV Developers <admin@opencv.org>")
  set(CPACK_PACKAGE_VERSION_MAJOR "${OPENCV_VERSION_MAJOR}")
  set(CPACK_PACKAGE_VERSION_MINOR "${OPENCV_VERSION_MINOR}")
  set(CPACK_PACKAGE_VERSION_PATCH "${OPENCV_VERSION_PATCH}")
  set(CPACK_PACKAGE_VERSION "${OPENCV_VCSVERSION}")
  if (NOT "${OPENCV_VCSVERSION}" MATCHES "^${OPENCV_VERSION}.*")
    message(WARNING "CPACK_PACKAGE_VERSION does not match version provided by version.hpp header!")
  endif()
  set(OPENCV_DEBIAN_COPYRIGHT_FILE "")
endif(NOT OPENCV_CUSTOM_PACKAGE_INFO)

set(CPACK_STRIP_FILES 1)

#arch
if(X86)
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "i686")
elseif(X86_64)
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
elseif(ARM)
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "armhf")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "armhf")
elseif(AARCH64)
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
else()
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
  set(CPACK_RPM_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
endif()

if(CPACK_GENERATOR STREQUAL "DEB")
  set(OPENCV_PACKAGE_ARCH_SUFFIX ${CPACK_DEBIAN_PACKAGE_ARCHITECTURE})
elseif(CPACK_GENERATOR STREQUAL "RPM")
  set(OPENCV_PACKAGE_ARCH_SUFFIX ${CPACK_RPM_PACKAGE_ARCHITECTURE})
else()
  set(OPENCV_PACKAGE_ARCH_SUFFIX ${CMAKE_SYSTEM_PROCESSOR})
endif()

set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${OPENCV_VCSVERSION}-${OPENCV_PACKAGE_ARCH_SUFFIX}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${OPENCV_VCSVERSION}-${OPENCV_PACKAGE_ARCH_SUFFIX}")

#rpm options
set(CPACK_RPM_COMPONENT_INSTALL TRUE)
set(CPACK_RPM_PACKAGE_SUMMARY ${CPACK_PACKAGE_DESCRIPTION_SUMMARY})
set(CPACK_RPM_PACKAGE_DESCRIPTION ${CPACK_PACKAGE_DESCRIPTION})
set(CPACK_RPM_PACKAGE_URL "http://opencv.org")
set(CPACK_RPM_PACKAGE_LICENSE "BSD")

#deb options
set(CPACK_DEB_COMPONENT_INSTALL TRUE)
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "http://opencv.org")

#display names
set(CPACK_COMPONENT_DEV_DISPLAY_NAME     "Development files")
set(CPACK_COMPONENT_DOCS_DISPLAY_NAME    "Documentation")
set(CPACK_COMPONENT_JAVA_DISPLAY_NAME    "Java bindings")
set(CPACK_COMPONENT_LIBS_DISPLAY_NAME    "Libraries and data")
set(CPACK_COMPONENT_PYTHON_DISPLAY_NAME  "Python bindings")
set(CPACK_COMPONENT_SAMPLES_DISPLAY_NAME "Samples")
set(CPACK_COMPONENT_TESTS_DISPLAY_NAME   "Tests")

#depencencies
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS TRUE)
set(CPACK_COMPONENT_LIBS_REQUIRED TRUE)
set(CPACK_COMPONENT_SAMPLES_DEPENDS libs dev)
set(CPACK_COMPONENT_DEV_DEPENDS libs)
set(CPACK_COMPONENT_DOCS_DEPENDS libs)
set(CPACK_COMPONENT_JAVA_DEPENDS libs)
set(CPACK_COMPONENT_PYTHON_DEPENDS libs)
set(CPACK_DEB_PYTHON_PACKAGE_DEPENDS "python-numpy (>=${PYTHON_NUMPY_VERSION}), python${PYTHON_VERSION_MAJOR_MINOR}")
set(CPACK_COMPONENT_TESTS_DEPENDS libs)
if (HAVE_opencv_python)
  set(CPACK_DEB_TESTS_PACKAGE_DEPENDS "python-numpy (>=${PYTHON_NUMPY_VERSION}), python${PYTHON_VERSION_MAJOR_MINOR}")
endif()

if(HAVE_CUDA)
  string(REPLACE "." "-" cuda_version_suffix ${CUDA_VERSION})
  if(CUDA_VERSION VERSION_LESS "6.5")
    set(CPACK_DEB_LIBS_PACKAGE_DEPENDS "cuda-core-libs-${cuda_version_suffix}, cuda-extra-libs-${cuda_version_suffix}")
    set(CPACK_DEB_DEV_PACKAGE_DEPENDS "cuda-headers-${cuda_version_suffix}")
  else()
    set(CPACK_DEB_LIBS_PACKAGE_DEPENDS "cuda-cudart-${cuda_version_suffix}, cuda-npp-${cuda_version_suffix}")
    set(CPACK_DEB_DEV_PACKAGE_DEPENDS "cuda-cudart-dev-${cuda_version_suffix}, cuda-npp-dev-${cuda_version_suffix}")
    if(HAVE_CUFFT)
      set(CPACK_DEB_LIBS_PACKAGE_DEPENDS "${CPACK_DEB_LIBS_PACKAGE_DEPENDS}, cuda-cufft-${cuda_version_suffix}")
      set(CPACK_DEB_DEV_PACKAGE_DEPENDS "${CPACK_DEB_DEV_PACKAGE_DEPENDS}, cuda-cufft-dev-${cuda_version_suffix}")
    endif()
    if(HAVE_HAVE_CUBLAS)
      set(CPACK_DEB_LIBS_PACKAGE_DEPENDS "${CPACK_DEB_LIBS_PACKAGE_DEPENDS}, cuda-cublas-${cuda_version_suffix}")
      set(CPACK_DEB_DEV_PACKAGE_DEPENDS "${CPACK_DEB_DEV_PACKAGE_DEPENDS}, cuda-cublas-dev-${cuda_version_suffix}")
    endif()
  endif()
endif()

if(HAVE_TBB AND NOT BUILD_TBB)
  if(CPACK_DEB_DEV_PACKAGE_DEPENDS)
    set(CPACK_DEB_DEV_PACKAGE_DEPENDS "${CPACK_DEB_DEV_PACKAGE_DEPENDS}, libtbb-dev")
  else()
    set(CPACK_DEB_DEV_PACKAGE_DEPENDS "libtbb-dev")
  endif()
endif()

set(STD_OPENCV_LIBS opencv-data)
set(STD_OPENCV_DEV libopencv-dev)

foreach(module calib3d contrib core features2d flann gpu highgui imgproc legacy
               ml objdetect ocl photo stitching superres ts video videostab)
  if(HAVE_opencv_${module})
    list(APPEND STD_OPENCV_LIBS "libopencv-${module}2.4")
    list(APPEND STD_OPENCV_DEV "libopencv-${module}-dev")
  endif()
endforeach()

list(APPEND STD_OPENCV_DEV "libhighgui-dev" "libcv-dev" "libcvaux-dev")

string(REPLACE ";" ", " CPACK_COMPONENT_LIBS_CONFLICTS "${STD_OPENCV_LIBS}")
string(REPLACE ";" ", " CPACK_COMPONENT_LIBS_PROVIDES "${STD_OPENCV_LIBS}")
string(REPLACE ";" ", " CPACK_COMPONENT_LIBS_REPLACES "${STD_OPENCV_LIBS}")

string(REPLACE ";" ", " CPACK_COMPONENT_DEV_CONFLICTS "${STD_OPENCV_DEV}")
string(REPLACE ";" ", " CPACK_COMPONENT_DEV_PROVIDES "${STD_OPENCV_DEV}")
string(REPLACE ";" ", " CPACK_COMPONENT_DEV_REPLACES "${STD_OPENCV_DEV}")

set(CPACK_COMPONENT_PYTHON_CONFLICTS python-opencv)
set(CPACK_COMPONENT_PYTHON_PROVIDES python-opencv)
set(CPACK_COMPONENT_PYTHON_REPLACES python-opencv)

set(CPACK_COMPONENT_JAVA_CONFLICTS "libopencv2.4-java, libopencv2.4-jni")
set(CPACK_COMPONENT_JAVA_PROVIDES "libopencv2.4-java, libopencv2.4-jni")
set(CPACK_COMPONENT_JAVA_REPLACES "libopencv2.4-java, libopencv2.4-jni")

set(CPACK_COMPONENT_DOCS_CONFLICTS opencv-doc)
set(CPACK_COMPONENT_SAMPLES_CONFLICTS opencv-doc)

if(NOT OPENCV_CUSTOM_PACKAGE_INFO)
  set(CPACK_COMPONENT_LIBS_DESCRIPTION "Open Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_LIBS_NAME "libopencv")
  set(CPACK_DEBIAN_COMPONENT_LIBS_SECTION "libs")

  set(CPACK_COMPONENT_PYTHON_DESCRIPTION "Python bindings for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_PYTHON_NAME "libopencv-python")
  set(CPACK_DEBIAN_COMPONENT_PYTHON_SECTION "python")

  set(CPACK_COMPONENT_JAVA_DESCRIPTION "Java bindings for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_JAVA_NAME "libopencv-java")
  set(CPACK_DEBIAN_COMPONENT_JAVA_SECTION "java")

  set(CPACK_COMPONENT_DEV_DESCRIPTION "Development files for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_DEV_NAME "libopencv-dev")
  set(CPACK_DEBIAN_COMPONENT_DEV_SECTION "libdevel")

  set(CPACK_COMPONENT_DOCS_DESCRIPTION "Documentation for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_DOCS_NAME "libopencv-docs")
  set(CPACK_DEBIAN_COMPONENT_DOCS_SECTION "doc")

  set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "Samples for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_SAMPLES_NAME "libopencv-samples")
  set(CPACK_DEBIAN_COMPONENT_SAMPLES_SECTION "devel")

  set(CPACK_COMPONENT_TESTS_DESCRIPTION "Accuracy and performance tests for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_TESTS_NAME "libopencv-tests")
  set(CPACK_DEBIAN_COMPONENT_TESTS_SECTION "misc")
endif(NOT OPENCV_CUSTOM_PACKAGE_INFO)

set(CPACK_DEBIAN_COMPONENT_DOCS_ARCHITECTURE "all")

# lintian stuff

#
# ocv_generate_lintian_overrides_file: generates lintian overrides file for
#   the specified component (deb-package). It's assumed that <comp>_LINTIAN_OVERRIDES
#   variable with suppressed tags is defined.
#
# Usage: ocv_generate_lintian_overrides_file(<component name>)
#

function(ocv_generate_lintian_overrides_file comp)
    string(TOUPPER ${comp} comp_upcase)

    set(package_name ${CPACK_DEBIAN_COMPONENT_${comp_upcase}_NAME})
    set(suppressions ${${comp_upcase}_LINTIAN_OVERRIDES})

    if(suppressions)
        if(NOT package_name)
            message(FATAL_ERROR "Package name for the ${comp} component is not defined")
        endif()

        # generate content of lintian overrides file

        foreach(suppression ${suppressions})
            set(line "${package_name}: ${suppression}")
            if(content)
                set(content "${content}\n${line}")
            else()
                set(content "${line}")
            endif()
        endforeach()

        # create file and install it
        set(cpack_tmp_dir "${CMAKE_BINARY_DIR}/deb-packages-gen/${comp}")
        set(overrides_filename "${cpack_tmp_dir}/${package_name}")

        file(WRITE "${overrides_filename}" "${content}")

        # install generated file
        install(FILES "${overrides_filename}"
            DESTINATION share/lintian/overrides/
            COMPONENT ${comp})

        unset(content)
    endif()
endfunction()

function(ocv_get_lintian_version version)
    find_program(LINTIAN_EXECUTABLE lintian)

    if(NOT LINTIAN_EXECUTABLE)
        return()
    endif()

    execute_process(COMMAND ${LINTIAN_EXECUTABLE} --version
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
              RESULT_VARIABLE LINTIAN_EXITCODE
              OUTPUT_VARIABLE LINTIAN_VERSION
              ERROR_QUIET)

    if(NOT LINTIAN_EXITCODE EQUAL 0)
        return()
    endif()

    if(LINTIAN_VERSION MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(LINTIAN_VERSION "${CMAKE_MATCH_1}" CACHE INTERNAL "Lintian version")
    endif()

    set("${version}" "${LINTIAN_VERSION}" PARENT_SCOPE)
endfunction()

ocv_get_lintian_version(LINTIAN_VERSION)

set(LIBS_LINTIAN_OVERRIDES "binary-or-shlib-defines-rpath" # usr/lib/libopencv_core.so.2.4.12
                           "package-name-doesnt-match-sonames") # libopencv-calib3d2.4 libopencv-contrib2.4

if(HAVE_opencv_python)
    set(PYTHON_LINTIAN_OVERRIDES "binary-or-shlib-defines-rpath" # usr/lib/python2.7/dist-packages/cv2.so
                                 "missing-dependency-on-numpy-abi")
else()
    set(PYTHON_LINTIAN_OVERRIDES "empty-binary-package") # python module is off
endif()

if(NOT HAVE_opencv_java)
    set(JAVA_LINTIAN_OVERRIDES "empty-binary-package") # Java is off
else()
    # TODO: add smht here
endif()

set(DEV_LINTIAN_OVERRIDES "binary-or-shlib-defines-rpath" # usr/bin/opencv_traincascade
                          "binary-without-manpage") # usr/bin/opencv_traincascade

if(LINTIAN_VERSION VERSION_GREATER "2.5.30" OR
    LINTIAN_VERSION VERSION_EQUAL "2.5.30")
    list(APPEND DEV_LINTIAN_OVERRIDES "pkg-config-bad-directive") # usr/lib/pkgconfig/opencv.pc -L/usr/local/cuda-7.0/lib64
endif()

if(NOT INSTALL_C_EXAMPLES)
    set(SAMPLES_LINTIAN_OVERRIDES "empty-binary-package") # samples are not installed
endif()

if(INSTALL_TESTS)
    set(TESTS_LINTIAN_OVERRIDES "arch-dependent-file-in-usr-share" # usr/share/OpenCV/bin/opencv_test_ml
                                "binary-or-shlib-defines-rpath" # usr/share/OpenCV/bin/opencv_test_ml
                                "python-script-but-no-python-dep") # usr/share/OpenCV/bin/calchist.py
else()
    set(TESTS_LINTIAN_OVERRIDES "empty-binary-package") # there is no tests
endif()

set(ALL_COMPONENTS "libs" "dev" "docs" "python" "java" "samples" "tests")

foreach (comp ${ALL_COMPONENTS})
    string(TOUPPER ${comp} comp_upcase)
    list(APPEND ${comp_upcase}_LINTIAN_OVERRIDES "misplaced-extra-member-in-deb") # for signed packages
endforeach()

if(CPACK_GENERATOR STREQUAL "DEB")
  find_program(GZIP_TOOL NAMES "gzip" PATHS "/bin" "/usr/bin" "/usr/local/bin")
  if(NOT GZIP_TOOL)
    message(FATAL_ERROR "Unable to find 'gzip' program")
  endif()

  execute_process(COMMAND "date" "-R"
                  OUTPUT_VARIABLE CHANGELOG_PACKAGE_DATE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(CHANGELOG_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")
  foreach (comp ${ALL_COMPONENTS})
    string(TOUPPER "${comp}" comp_upcase)

    set(DEBIAN_CHANGELOG_OUT_FILE    "${CMAKE_BINARY_DIR}/deb-packages-gen/${comp}/changelog.Debian")
    set(DEBIAN_CHANGELOG_OUT_FILE_GZ "${CMAKE_BINARY_DIR}/deb-packages-gen/${comp}/changelog.Debian.gz")
    set(CHANGELOG_PACKAGE_NAME "${CPACK_DEBIAN_COMPONENT_${comp_upcase}_NAME}")
    configure_file("${CMAKE_SOURCE_DIR}/cmake/templates/changelog.Debian.in" "${DEBIAN_CHANGELOG_OUT_FILE}" @ONLY)

    execute_process(COMMAND "${GZIP_TOOL}" "-cf9" "${DEBIAN_CHANGELOG_OUT_FILE}"
                    OUTPUT_FILE "${DEBIAN_CHANGELOG_OUT_FILE_GZ}"
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

    install(FILES "${DEBIAN_CHANGELOG_OUT_FILE_GZ}"
            DESTINATION "share/doc/${CPACK_DEBIAN_COMPONENT_${comp_upcase}_NAME}"
            COMPONENT "${comp}")

    set(CHANGELOG_OUT_FILE "${CMAKE_BINARY_DIR}/deb-packages-gen/${comp}/changelog")
    set(CHANGELOG_OUT_FILE_GZ "${CMAKE_BINARY_DIR}/deb-packages-gen/${comp}/changelog.gz")
    file(WRITE ${CHANGELOG_OUT_FILE} "Upstream changelog stub. See https://github.com/Itseez/opencv/wiki/ChangeLog")

    execute_process(COMMAND "${GZIP_TOOL}" "-cf9" "${CHANGELOG_OUT_FILE}"
                    OUTPUT_FILE "${CHANGELOG_OUT_FILE_GZ}"
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

    install(FILES "${CHANGELOG_OUT_FILE_GZ}"
            DESTINATION "share/doc/${CPACK_DEBIAN_COMPONENT_${comp_upcase}_NAME}"
            COMPONENT "${comp}")

    if(OPENCV_DEBIAN_COPYRIGHT_FILE)
        install(FILES "${OPENCV_DEBIAN_COPYRIGHT_FILE}"
                DESTINATION "share/doc/${CPACK_DEBIAN_COMPONENT_${comp_upcase}_NAME}"
                COMPONENT "${comp}")
    endif()

    ocv_generate_lintian_overrides_file("${comp}")

  endforeach()
endif()

include(CPack)

ENDif(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
