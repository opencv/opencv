#Cross compile TBB from source
project(tbb)

if (WIN32 AND NOT ARM)
  message(FATAL_ERROR "BUILD_TBB option supports Windows on ARM only!\nUse regular official TBB build instead of the BUILD_TBB option!")
endif()

set(tbb_ver "tbb43_20141204oss")
set(tbb_url "http://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb43_20141204oss_src.tgz")
set(tbb_md5 "e903dd92d9433701f097fa7ca29a3c1f")
set(tbb_version_file "version_string.ver")
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4702)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow)

# 4.1 update 4 - The first release that supports Windows RT. Hangs on some Android devices
#set(tbb_ver "tbb41_20130613oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb41_20130613oss_src.tgz")
#set(tbb_md5 "108c8c1e481b0aaea61878289eb28b6a")
#set(tbb_version_file "version_string.ver")
#ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4702)

# 4.1 update 2 - works fine
#set(tbb_ver "tbb41_20130116oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb41_20130116oss_src.tgz")
#set(tbb_md5 "3809790e1001a1b32d59c9fee590ee85")
#set(tbb_version_file "version_string.ver")
#ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow)

# 4.1 update 3 dev - Hangs on some Android devices
#set(tbb_ver "tbb41_20130401oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb41_20130401oss_src.tgz")
#set(tbb_md5 "f2f591a0d2ca8f801e221ce7d9ea84bb")
#set(tbb_version_file "version_string.ver")
#ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow)

# 4.1 update 2 - works fine
#set(tbb_ver "tbb41_20130116oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb41_20130116oss_src.tgz")
#set(tbb_md5 "3809790e1001a1b32d59c9fee590ee85")
#set(tbb_version_file "version_string.ver")
#ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow)

# 4.1 update 1 - works fine
#set(tbb_ver "tbb41_20121003oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb41_20121003oss_src.tgz")
#set(tbb_md5 "2a684fefb855d2d0318d1ef09afa75ff")
#set(tbb_version_file "version_string.ver")

# 4.1 - works fine
#set(tbb_ver "tbb41_20120718oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb41_20120718oss_src.tgz")
#set(tbb_md5 "31b9ec300f3d09da2504d5d882788dd4")
#set(tbb_version_file "version_string.ver")

# 4.0 update 5 - works fine
#set(tbb_ver "tbb40_20120613oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb40_20120613oss_src.tgz")
#set(tbb_md5 "da01ed74944ec5950cfae3476901a172")
#set(tbb_version_file "version_string.ver")

# 4.0 update 4 - works fine
#set(tbb_ver "tbb40_20120408oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb40_20120408oss_src.tgz")
#set(tbb_md5 "734b356da7fe0ed308741f3e6018251e")
#set(tbb_version_file "version_string.ver")

# 4.0 update 3 - build broken
#set(tbb_ver "tbb40_20120201oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb40_20120201oss_src.tgz")
#set(tbb_md5 "4669e7d4adee018de7a7b8b972987218")
#set(tbb_version_file "version_string.tmp")

# 4.0 update 2 - works fine
#set(tbb_ver "tbb40_20111130oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb40_20111130oss_src.tgz")
#set(tbb_md5 "1e6926b21e865e79772119cd44fc3ad8")
#set(tbb_version_file "version_string.tmp")
#set(tbb_need_GENERIC_DWORD_LOAD_STORE TRUE)

# 4.0 update 1 - works fine
#set(tbb_ver "tbb40_20111003oss")
#set(tbb_url "http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb40_20111003oss_src.tgz")
#set(tbb_md5 "7b5d94eb35a563b29ef402e0fd8f15c9")
#set(tbb_version_file "version_string.tmp")
#set(tbb_need_GENERIC_DWORD_LOAD_STORE TRUE)

set(tbb_tarball "${CMAKE_CURRENT_SOURCE_DIR}/${tbb_ver}_src.tgz")
set(tbb_src_dir "${CMAKE_CURRENT_BINARY_DIR}/${tbb_ver}")

if(EXISTS "${tbb_tarball}")
  file(MD5 "${tbb_tarball}" tbb_local_md5)
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
    message(FATAL_ERROR "Failed to download TBB sources (${__statvar}): ${tbb_url}")
  endif()
  file(MD5 "${tbb_tarball}" tbb_local_md5)
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
    message(FATAL_ERROR "Failed to unpack TBB sources from ${tbb_tarball} to ${tbb_src_dir} with error ${tbb_untar_RESULT}")
  endif()
endif()

set(TBB_INCLUDE_DIRS "${tbb_src_dir}/include" PARENT_SCOPE)

ocv_include_directories("${tbb_src_dir}/include"
                        "${tbb_src_dir}/src/"
                        "${tbb_src_dir}/src/rml/include"
                        "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB lib_srcs "${tbb_src_dir}/src/tbb/*.cpp")
file(GLOB lib_hdrs "${tbb_src_dir}/src/tbb/*.h")
list(APPEND lib_srcs "${tbb_src_dir}/src/rml/client/rml_tbb.cpp")

if (WIN32)
  add_definitions(/D__TBB_DYNAMIC_LOAD_ENABLED=0
                  /D__TBB_BUILD=1
                  /DTBB_NO_LEGACY=1
                  /D_UNICODE
                  /DUNICODE
                  /DWINAPI_FAMILY=WINAPI_FAMILY_APP
                  /DDO_ITT_NOTIFY=0
                  /DUSE_WINTHREAD
               ) # defines were copied from windows.cl.inc

  if (ARM)
    add_definitions(/D_WIN32_WINNT=0x0602
                    /D__TBB_WIN32_USE_CL_BUILTINS
                   )
  endif()

set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} /APPCONTAINER")
else()
  add_definitions(-D__TBB_DYNAMIC_LOAD_ENABLED=0         #required
                  -D__TBB_WEAK_SYMBOLS_PRESENT=0         #required for 4.3
                  -D__TBB_BUILD=1                        #required
                  -D__TBB_SURVIVE_THREAD_SWITCH=0        #no cilk support
                  -DTBB_USE_DEBUG=0                      #just to be sure
                  -DTBB_NO_LEGACY=1                      #don't need backward compatibility
                  -DDO_ITT_NOTIFY=0                      #it seems that we don't need these notifications
                 )
endif()

if (HAVE_LIBPTHREAD)
  add_definitions(-DUSE_PTHREAD) #required for Unix
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-DTBB_USE_GCC_BUILTINS=1) #required for ARM GCC
endif()

if(ANDROID_COMPILER_IS_CLANG)
  add_definitions(-D__TBB_GCC_BUILTIN_ATOMICS_PRESENT=1)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wmissing-prototypes)
endif()

if(tbb_need_GENERIC_DWORD_LOAD_STORE)
  #needed by TBB 4.0 update 1,2; fixed in TBB 4.0 update 3 but it has 2 new problems
  add_definitions(-D__TBB_USE_GENERIC_DWORD_LOAD_STORE=1)
  set(tbb_need_GENERIC_DWORD_LOAD_STORE ON PARENT_SCOPE)
endif()

set(TBB_SOURCE_FILES ${lib_srcs} ${lib_hdrs})

if (ARM AND NOT WIN32)
  if (NOT ANDROID)
    set(TBB_SOURCE_FILES ${TBB_SOURCE_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/arm_linux_stub.cpp")
  endif()
  set(TBB_SOURCE_FILES ${TBB_SOURCE_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/android_additional.h")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include \"${CMAKE_CURRENT_SOURCE_DIR}/android_additional.h\"")
endif()

set(TBB_SOURCE_FILES ${TBB_SOURCE_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/${tbb_version_file}")

add_library(tbb ${TBB_SOURCE_FILES})

if (WIN32)
  if (ARM)
    set(platform_macro /D_M_ARM=1)
  endif()

  add_custom_command(TARGET tbb
                     PRE_BUILD
                     COMMAND ${CMAKE_C_COMPILER} /nologo /TC /EP ${tbb_src_dir}\\src\\tbb\\win32-tbb-export.def /DTBB_NO_LEGACY=1 /D_CRT_SECURE_NO_DEPRECATE /D__TBB_BUILD=1 ${platform_macro} /I${tbb_src_dir}\\src /I${tbb_src_dir}\\include > "${tbb_src_dir}\\src\\tbb\\tbb.def"
                     WORKING_DIRECTORY ${tbb_src_dir}\\src\\tbb
                     COMMENT "Generating tbb.def file" VERBATIM
                    )

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DEF:${tbb_src_dir}/src/tbb/tbb.def /DLL /MAP /fixed:no /INCREMENTAL:NO")
else()
  target_link_libraries(tbb c m dl)
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

# filter out flags that are not handled well by the TBB code
foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
  string(REPLACE "-Werror=non-virtual-dtor" "" ${var} "${${var}}")
  string(REPLACE "-fvisibility=hidden" "" ${var} "${${var}}")
  string(REPLACE "-fvisibility-inlines-hidden" "" ${var} "${${var}}")
endforeach()

if (WIN32)
  set(tbb_debug_postfix "_debug") # to fit pragmas in _windef.h inside TBB
else()
  set(tbb_debug_postfix ${OPENCV_DEBUG_POSTFIX})
endif()

set_target_properties(tbb
  PROPERTIES OUTPUT_NAME tbb
  DEBUG_POSTFIX "${tbb_debug_postfix}"
  COMPILE_PDB_NAME tbb
  COMPILE_PDB_NAME_DEBUG "tbb${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(tbb PROPERTIES FOLDER "3rdparty")
endif()

ocv_install_target(tbb EXPORT OpenCVModules
    RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT libs
    LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT libs
    ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev
    )

# get TBB version
ocv_parse_header("${tbb_src_dir}/include/tbb/tbb_stddef.h" TBB_VERSION_LINES TBB_VERSION_MAJOR TBB_VERSION_MINOR TBB_INTERFACE_VERSION CACHE)
