# create .deb file
# You need to setup CPack first !
# UGLY: I reuse CPACK_NSIS_CONTACT to get the contact name for the debian package...
# TODO: How do I transmit the 'Depends' line ?

# DOCUMENTATION; You need to fill these values to set the control file:
# "Package: ${DEBIAN_PACKAGE_NAME}
# Version: ${DEBIAN_PACKAGE_VERSION}
# Architecture: ${DEBIAN_ARCHITECTURE}
# Depends: ${DEBIAN_PACKAGE_DEPENDS}
# Maintainer: ${CPACK_NSIS_CONTACT}
# Description: ${CPACK_PACKAGE_DESCRIPTION_SUMMARY}

# Thanks:
# Eric Noulard for initial UseRpmTools used as template
#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

if(UNIX)
  if (NOT CMAKE_AR)
    message(STATUS "No ar, cannot proceed")
    set(DEBIAN_FOUND FALSE)
  else ()
    set(DEBIAN_FOUND TRUE)
    # Detect if CPack was included or not
    if (NOT DEFINED "CPACK_PACKAGE_NAME")
      message(FATAL_ERROR "CPack was not included, you should include CPack before Using UseDebian")
    endif ()

    macro(ADD_DEBIAN_TARGETS DEBNAME)
      # $ ar tv cmake_2.4.5-1_i386.deb
      # rw-r--r-- 0/0      4 Dec  4 22:58 2006 debian-binary
      # rw-r--r-- 0/0   8981 Dec  4 22:58 2006 control.tar.gz
      # rw-r--r-- 0/0 4893146 Dec  4 22:58 2006 data.tar.gz

      # Need a newline:
      # dpkg-deb: archive has no newlines in header
      add_custom_command(
        OUTPUT    ${CMAKE_BINARY_DIR}/debian-binary
        COMMAND   ${CMAKE_COMMAND}
        ARGS      -E echo "2.0" > ${CMAKE_BINARY_DIR}/debian-binary
        COMMENT   "Generating debian-binary"
        VERBATIM)

      # seems better to ADD_CUSTOM_COMMAND this way debian-binary may
      # be regenerated when make is called.
      #      file(WRITE ${CMAKE_BINARY_DIR}/debian-binary "2.0
      #" )

      # debian policy enforce lower case for package name
      if(NOT DEBIAN_PACKAGE_NAME)
        string(TOLOWER
          ${CPACK_PACKAGE_NAME}
          DEBIAN_PACKAGE_NAME
          )
      endif()
      if(NOT DEBIAN_PACKAGE_DEPENDS)
        set(DEBIAN_PACKAGE_DEPENDS
          "libc6 (>= 2.3.1-6), libgcc1 (>= 1:3.4.2-12)"
          )
      endif()
      if(NOT DEBIAN_ARCHITECTURE)
        # There is no such thing as i686 architecture on debian, you should use i386 instead
        # $ dpkg --print-architecture
        set(DEBIAN_ARCHITECTURE i386)
      endif()
      if(NOT DEBIAN_PACKAGE_VERSION)
        set(DEBIAN_PACKAGE_VERSION
          ${CPACK_PACKAGE_VERSION})
      endif()

      #message(${CMAKE_SYSTEM_PROCESSOR})

      file(WRITE ${CMAKE_BINARY_DIR}/control
        "Package: ${DEBIAN_PACKAGE_NAME}
        Version: ${CPACK_PACKAGE_VERSION}
        Section: devel
        Priority: optional
        Architecture: ${DEBIAN_ARCHITECTURE}
        Depends: ${DEBIAN_PACKAGE_DEPENDS}
        Maintainer: ${CPACK_NSIS_CONTACT}
        Description: ${CPACK_PACKAGE_DESCRIPTION_SUMMARY}
        .
        ${DEBIAN_PACKAGE_NAME} was packaged by UseDebian and CMake.
        .
        ")


      # FIXME:
      # I have no friggin clue how cpack works, let's reinvent the wheel instead

      #include(${CMAKE_BINARY_DIR}/CPackConfig.cmake)
      #add_custom_target(data_tgz
      #  COMMAND cpack -G TGZ --config CPackConfig.cmake
      ## TODO: How to get the cpack package name ?
      #  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/${CPACK_PACKAGE_FILE_NAME}.tar.gz ${CMAKE_BINARY_DIR}/data.tar.gz
      #)

      # let's create a temp directory to call 'DESTDIR=... make install' into:
      # cleanup
      file(REMOVE ${CMAKE_BINARY_DIR}/debian_package)
      # make dir:
      file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/debian_package)

      # calling cmake -P cmake_install.cmake is the same as calling make install:
      add_custom_target(deb_destdir_install
        COMMAND ${CMAKE_MAKE_PROGRAM} DESTDIR=${CMAKE_BINARY_DIR}/debian_package install
        DEPENDS ${CMAKE_BINARY_DIR}/cmake_install.cmake
        COMMENT "Building debian_package directory with DESTDIR"
        )
      add_dependencies(deb_destdir_install all preinstall)

      # create data.tar.gz from the make install stuff
      # all files starts with: ./usr
      add_custom_command(
        OUTPUT    ${CMAKE_BINARY_DIR}/data.tar.gz
        COMMAND   cmake -E tar
        ARGS      cfz ${CMAKE_BINARY_DIR}/data.tar.gz .
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/debian_package
        DEPENDS   ${CMAKE_BINARY_DIR}/debian_package
        COMMENT   "Generating data.tar.gz"
        )


      # get all the files to be installed:
      find_package(Md5sum REQUIRED)
      COMPUTE_MD5SUMS(
        ${CMAKE_BINARY_DIR}/debian_package
        ${CMAKE_BINARY_DIR}/md5sums
        )

      # create a tarball (control.tar.gz) of control and md5sums
      # files need to be in relative path: ./md5sums ./control ...
      add_custom_command(
        OUTPUT    ${CMAKE_BINARY_DIR}/control.tar.gz
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND   cmake -E tar
        ARGS      cfz ${CMAKE_BINARY_DIR}/control.tar.gz ./control ./md5sums
        DEPENDS   ${CMAKE_BINARY_DIR}/control ${CMAKE_BINARY_DIR}/md5sums
        COMMENT   "Generating control.tar.gz"
        )



      # Warning order is important:
      # ar -r your-package-name.deb debian-binary control.tar.gz data.tar.gz
      # eg: cmake_2.4.5-1_i386.deb
      add_custom_command(
        OUTPUT    ${CMAKE_BINARY_DIR}/${DEBIAN_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-1_${DEBIAN_ARCHITECTURE}.deb
        COMMAND   ${CMAKE_AR}
        ARGS      -r ${CMAKE_BINARY_DIR}/${DEBIAN_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-1_${DEBIAN_ARCHITECTURE}.deb
        ${CMAKE_BINARY_DIR}/debian-binary
        ${CMAKE_BINARY_DIR}/control.tar.gz ${CMAKE_BINARY_DIR}/data.tar.gz
        DEPENDS   ${CMAKE_BINARY_DIR}/debian-binary ${CMAKE_BINARY_DIR}/control.tar.gz ${CMAKE_BINARY_DIR}/data.tar.gz
        COMMENT   "Generating deb package"
        )

      # the final target:
      add_custom_target(${DEBNAME}_deb
        DEPENDS ${CMAKE_BINARY_DIR}/${DEBIAN_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-1_${DEBIAN_ARCHITECTURE}.deb
        )
      add_dependencies(${DEBNAME}_deb deb_destdir_install)


      # BUG: debian_package is not removed during a 'make clean':
      set_directory_properties(PROPERTIES
        ADDITIONAL_MAKE_CLEAN_FILES "debian-binary;control;md5sums;debian_package;")

    endmacro()
  endif ()
endif()
