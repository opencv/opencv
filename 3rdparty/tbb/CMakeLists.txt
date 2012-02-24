#build TBB for Android from source
if(NOT ANDROID)
  message(FATAL_ERROR "The script is designed for Android only!")
endif()

project(tbb)

set(tbb_ver "tbb40_20111003oss")
set(tbb_url "http://threadingbuildingblocks.org/uploads/77/177/4.0%20update%201/tbb40_20111003oss_src.tgz")
set(tbb_md5 "7b5d94eb35a563b29ef402e0fd8f15c9")

set(tbb_tarball "${CMAKE_CURRENT_SOURCE_DIR}/${tbb_ver}_src.tgz")
set(tbb_src_dir "${CMAKE_CURRENT_BINARY_DIR}/${tbb_ver}")

macro(getMD5 filename varname)
  if(CMAKE_VERSION VERSION_GREATER 2.8.6)
    file(MD5 "${filename}" ${varname})
  else()
    execute_process(COMMAND ${CMAKE_COMMAND} -E md5sum "${filename}"
              RESULT_VARIABLE getMD5_RESULT
              OUTPUT_VARIABLE getMD5_OUTPUT
              OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(getMD5_RESULT EQUAL 0)
      string(REGEX MATCH "^[a-z0-9]+" ${varname} "${getMD5_OUTPUT}")
    else()
      set(${varname} "invalid_md5")
    endif()
  endif()
endmacro()

if(EXISTS "${tbb_tarball}")
  getMD5("${tbb_tarball}" tbb_local_md5)
  if(NOT tbb_local_md5 STREQUAL tbb_md5)
    message(WARNING "Local copy of TBB source tarball has invalid MD5 hash: ${tbb_local_md5} (expected: ${tbb_md5})")
    file(REMOVE "${tbb_tarball}")
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${tbb_ver}")
      file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/${tbb_ver}")
    endif()
  endif()
endif()

if(NOT EXISTS "${tbb_tarball}")
  message(STATUS "Downloading ${tbb_ver}_src.tgz")
  file(DOWNLOAD "${tbb_url}" "${tbb_tarball}" TIMEOUT 600 STATUS __statvar)
  if(NOT __statvar EQUAL 0)
    message(FATAL_ERROR "Failed to download TBB sources: ${tbb_url}")
  endif()
  getMD5("${tbb_tarball}" tbb_local_md5)
  if(NOT tbb_local_md5 STREQUAL tbb_md5)
    file(REMOVE "${tbb_tarball}")
    message(FATAL_ERROR "Downloaded TBB source tarball has invalid MD5 hash: ${tbb_local_md5} (expected: ${tbb_md5})")
  endif()
  
  if(EXISTS "${tbb_src_dir}")
    file(REMOVE_RECURSE "${tbb_src_dir}")
  endif()
endif()

# untar TBB sources
if(NOT EXISTS "${tbb_src_dir}")
  message(STATUS "Unpacking ${tbb_ver}_src.tgz to ${tbb_src_dir}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xz "${tbb_tarball}"
              WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
              RESULT_VARIABLE tbb_untar_RESULT)

  if(NOT tbb_untar_RESULT EQUAL 0 OR NOT EXISTS "${tbb_src_dir}")
    message(FATAL_ERROR "Failed to unpack TBB sources")
  endif()
endif()

set(TBB_INCLUDE_DIRS "${tbb_src_dir}/include" CACHE PATH "TBB headers location")
mark_as_advanced(TBB_INCLUDE_DIRS)

include_directories("${tbb_src_dir}/include"
                    "${tbb_src_dir}/src/"
                    "${tbb_src_dir}/src/rml/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB lib_srcs "${tbb_src_dir}/src/tbb/*.cpp")
file(GLOB lib_hdrs "${tbb_src_dir}/src/tbb/*.h")

list(APPEND lib_srcs "${tbb_src_dir}/src/rml/client/rml_tbb.cpp")

add_definitions(-D__TBB_DYNAMIC_LOAD_ENABLED=0
                -D__TBB_NO_DLOPEN
                -D__TBB_WEAK_SYMBOLS
                -DTBB_USE_GCC_BUILTINS=1
                -DUSE_PTHREAD
                -D__TBB_USE_GENERIC_DWORD_LOAD_STORE=1
                -D__TBB_GCC_BUILTIN_ATOMICS_PRESENT=1
                -D__TBB_BUILD=1
                -DDO_ITT_NOTIFY
               )

add_library(tbb STATIC ${lib_srcs} ${lib_hdrs})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -include \"${CMAKE_CURRENT_SOURCE_DIR}/android_additional.h\"")

set_target_properties(tbb
	PROPERTIES OUTPUT_NAME tbb
	DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/3rdparty/${OPENCV_LIB_INSTALL_PATH}"
	)
	
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(tbb PROPERTIES FOLDER "3rdparty")
endif()
    
if(NOT BUILD_SHARED_LIBS)
  install(TARGETS tbb ARCHIVE DESTINATION share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH} COMPONENT main)
endif()
