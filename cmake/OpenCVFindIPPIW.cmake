#
# The script to detect Intel(R) Integrated Performance Primitives Integration Wrappers (IPP IW)
# installation/package
#
#
# On return this will define:
#
# HAVE_IPP_IW       - True if Intel IPP found
# IPP_IW_PATH       - Root of Intel IPP IW directory
# IPP_IW_LIBRARIES  - Intel IPP IW libraries
# IPP_IW_INCLUDES   - Intel IPP IW include folder
#

unset(HAVE_IPP_IW CACHE)
unset(IPP_IW_PATH)
unset(IPP_IW_LIBRARIES)
unset(IPP_IW_INCLUDES)
unset(IW_CONFIG_DEBUG)
#set(IW_CONFIG_DEBUG 1)

if(NOT HAVE_IPP)
  return()
endif()

macro(ippiw_debugmsg MESSAGE)
  if(DEFINED IW_CONFIG_DEBUG)
    message(STATUS "${MESSAGE}")
  endif()
endmacro()
file(TO_CMAKE_PATH "${IPPROOT}" IPPROOT)

# This function detects Intel IPP IW version by analyzing .h file
macro(ippiw_setup PATH BUILD)
  set(FILE "${PATH}/include/iw/iw_ll.h") # check if Intel IPP IW is OpenCV specific
  ippiw_debugmsg("Checking path: ${PATH}")
  if(EXISTS "${FILE}")
    set(FILE "${PATH}/include/iw/iw_version.h")
    ippiw_debugmsg("vfile\tok")
    file(STRINGS "${FILE}" IW_VERSION_MAJOR  REGEX "IW_VERSION_MAJOR")
    file(STRINGS "${FILE}" IW_VERSION_MINOR  REGEX "IW_VERSION_MINOR")
    file(STRINGS "${FILE}" IW_VERSION_UPDATE REGEX "IW_VERSION_UPDATE")

    file(STRINGS "${FILE}" IW_MIN_COMPATIBLE_IPP_MAJOR  REGEX "IW_MIN_COMPATIBLE_IPP_MAJOR")
    file(STRINGS "${FILE}" IW_MIN_COMPATIBLE_IPP_MINOR  REGEX "IW_MIN_COMPATIBLE_IPP_MINOR")
    file(STRINGS "${FILE}" IW_MIN_COMPATIBLE_IPP_UPDATE REGEX "IW_MIN_COMPATIBLE_IPP_UPDATE")

    string(REGEX MATCH "[0-9]+" IW_MIN_COMPATIBLE_IPP_MAJOR  ${IW_MIN_COMPATIBLE_IPP_MAJOR})
    string(REGEX MATCH "[0-9]+" IW_MIN_COMPATIBLE_IPP_MINOR  ${IW_MIN_COMPATIBLE_IPP_MINOR})
    string(REGEX MATCH "[0-9]+" IW_MIN_COMPATIBLE_IPP_UPDATE ${IW_MIN_COMPATIBLE_IPP_UPDATE})

    string(REGEX MATCH "[0-9]+" IW_VERSION_MAJOR  ${IW_VERSION_MAJOR})
    string(REGEX MATCH "[0-9]+" IW_VERSION_MINOR  ${IW_VERSION_MINOR})
    string(REGEX MATCH "[0-9]+" IW_VERSION_UPDATE ${IW_VERSION_UPDATE})

    math(EXPR IPP_VERSION_EXP           "${IPP_VERSION_MAJOR}*10000 + ${IPP_VERSION_MINOR}*100 + ${IPP_VERSION_BUILD}")
    math(EXPR IW_MIN_COMPATIBLE_IPP_EXP "${IW_MIN_COMPATIBLE_IPP_MAJOR}*10000 + ${IW_MIN_COMPATIBLE_IPP_MINOR}*100 + ${IW_MIN_COMPATIBLE_IPP_UPDATE}")

    if((IPP_VERSION_EXP GREATER IW_MIN_COMPATIBLE_IPP_EXP) OR (IPP_VERSION_EXP EQUAL IW_MIN_COMPATIBLE_IPP_EXP))
      ippiw_debugmsg("version\tok")
      if(${BUILD})
        # check sources
        if(EXISTS "${PATH}/src/iw_core.c")
          ippiw_debugmsg("sources\tok")
          set(IPP_IW_PATH "${PATH}")
          message(STATUS "found Intel IPP IW sources: ${IW_VERSION_MAJOR}.${IW_VERSION_MINOR}.${IW_VERSION_UPDATE}")
          message(STATUS "at: ${IPP_IW_PATH}")

          set(IPP_IW_LIBRARY ippiw)
          set(IPP_IW_INCLUDES "${IPP_IW_PATH}/include")
          set(IPP_IW_LIBRARIES ${IPP_IW_LIBRARY})
          execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OpenCV_SOURCE_DIR}/3rdparty/ippicv/CMakeLists.txt" "${IPP_IW_PATH}/")
          add_subdirectory("${IPP_IW_PATH}/" ${OpenCV_BINARY_DIR}/3rdparty/ippiw)

          set(HAVE_IPP_IW 1)
          return()
        endif()
      else()
        # check binaries
        if(IPP_X64)
          set(FILE "${PATH}/lib/intel64/${CMAKE_STATIC_LIBRARY_PREFIX}ipp_iw${CMAKE_STATIC_LIBRARY_SUFFIX}")
        else()
          set(FILE "${PATH}/lib/ia32/${CMAKE_STATIC_LIBRARY_PREFIX}ipp_iw${CMAKE_STATIC_LIBRARY_SUFFIX}")
        endif()
        if(EXISTS ${FILE})
          ippiw_debugmsg("binaries\tok (64=${IPP_X64})")
          set(IPP_IW_PATH "${PATH}")
          message(STATUS "found Intel IPP IW binaries: ${IW_VERSION_MAJOR}.${IW_VERSION_MINOR}.${IW_VERSION_UPDATE}")
          message(STATUS "at: ${IPP_IW_PATH}")

          add_library(ippiw STATIC IMPORTED)
          set_target_properties(ippiw PROPERTIES
            IMPORTED_LINK_INTERFACE_LIBRARIES ""
            IMPORTED_LOCATION "${FILE}"
          )
          if (NOT BUILD_SHARED_LIBS)
            # CMake doesn't support "install(TARGETS ${name} ...)" command with imported targets
            install(FILES "${FILE}"
                    DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
            set(IPPIW_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/${OPENCV_3P_LIB_INSTALL_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}ipp_iw${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE INTERNAL "" FORCE)
            set(IPPIW_LOCATION_PATH "${FILE}" CACHE INTERNAL "" FORCE)
          endif()

          set(IPP_IW_INCLUDES "${IPP_IW_PATH}/include")
          set(IPP_IW_LIBRARIES ippiw)

          set(HAVE_IPP_IW 1)
          set(BUILD_IPP_IW 0)
          return()
        endif()
      endif()
    endif()
  endif()
  set(HAVE_IPP_IW 0)
endmacro()

# check os and architecture
if(APPLE)
  set(IW_PACKAGE_SUBDIR "ippiw_mac")
elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
  set(IW_PACKAGE_SUBDIR "ippiw_lnx")
elseif(WIN32 AND NOT ARM)
  set(IW_PACKAGE_SUBDIR "ippiw_win")
else()
  message(SEND_ERROR "Improper system for Intel IPP Integrations Wrappers. This message shouldn't appear. Check Intel IPP configurations steps")
  return()
endif()

# check build options first
if(BUILD_IPP_IW)
  # custom path
  if(DEFINED IPPIWROOT)
    ippiw_setup("${IPPIWROOT}/" 1)
    message(STATUS "Can't find Intel IPP IW sources at: ${IPPIWROOT}")
  endif()

  # local sources
  ippiw_setup("${OpenCV_SOURCE_DIR}/3rdparty/ippiw" 1)

  # Package sources
  get_filename_component(__PATH "${IPPROOT}/../${IW_PACKAGE_SUBDIR}/" ABSOLUTE)
  ippiw_setup("${__PATH}" 1)
endif()


# custom binaries
if(DEFINED IPPIWROOT)
  ippiw_setup("${IPPIWROOT}/" 0)
  message(STATUS "Can't find Intel IPP IW sources at: ${IPPIWROOT}")
endif()

# check binaries in IPP folder
ippiw_setup("${IPPROOT}/" 0)

# check binaries near IPP folder
ippiw_setup("${IPPROOT}/../${IW_PACKAGE_SUBDIR}/" 0)


# take Intel IPP IW from ICV package
if(NOT HAVE_IPP_ICV AND BUILD_IPP_IW)
  message(STATUS "Cannot find Intel IPP IW. Checking \"Intel IPP for OpenCV\" package")
  set(TEMP_ROOT 0)
  include("${OpenCV_SOURCE_DIR}/3rdparty/ippicv/ippicv.cmake")
  download_ippicv(TEMP_ROOT)

  # Package sources. Only sources are compatible with regular Intel IPP
  ippiw_setup("${TEMP_ROOT}/../${IW_PACKAGE_SUBDIR}/" 1)
endif()


set(HAVE_IPP_IW 0)
message(STATUS "Cannot find Intel IPP Integration Wrappers, optimizations will be limited. Use IPPIWROOT to set custom location")
return()
