#Cross compile TBB from source
project(tbb CXX)

if (WIN32 AND NOT ARM)
  message(FATAL_ERROR "BUILD_TBB option supports Windows on ARM only!\nUse regular official TBB build instead of the BUILD_TBB option!")
endif()

ocv_update(OPENCV_TBB_RELEASE "v2020.2")
ocv_update(OPENCV_TBB_RELEASE_MD5 "5af6f6c2a24c2043e62e47205e273b1f")
ocv_update(OPENCV_TBB_FILENAME "${OPENCV_TBB_RELEASE}.tar.gz")
string(REGEX REPLACE "^v" "" OPENCV_TBB_RELEASE_ "${OPENCV_TBB_RELEASE}")
#ocv_update(OPENCV_TBB_SUBDIR ...)

set(tbb_src_dir "${OpenCV_BINARY_DIR}/3rdparty/tbb")
ocv_download(FILENAME ${OPENCV_TBB_FILENAME}
             HASH ${OPENCV_TBB_RELEASE_MD5}
             URL
               "${OPENCV_TBB_URL}"
               "$ENV{OPENCV_TBB_URL}"
               "https://github.com/01org/tbb/archive/"
             DESTINATION_DIR ${tbb_src_dir}
             ID TBB
             STATUS res
             UNPACK RELATIVE_URL)
if(NOT res)
  return()
endif()
if(OPENCV_TBB_SUBDIR)
  # use current value
  ocv_assert(EXISTS "${tbb_src_dir}/${OPENCV_TBB_SUBDIR}")
elseif(EXISTS "${tbb_src_dir}/oneTBB-${OPENCV_TBB_RELEASE_}")
  set(OPENCV_TBB_SUBDIR "oneTBB-${OPENCV_TBB_RELEASE_}")
elseif(EXISTS "${tbb_src_dir}/tbb-${OPENCV_TBB_RELEASE_}")
  set(OPENCV_TBB_SUBDIR "oneTBB-${OPENCV_TBB_RELEASE_}")
else()
  message(FATAL_ERROR "TBB: Can't configure TBB. Specify OPENCV_TBB_SUBDIR through command-line.")
endif()
set(tbb_src_dir "${tbb_src_dir}/${OPENCV_TBB_SUBDIR}")

ocv_include_directories("${tbb_src_dir}/include"
                        "${tbb_src_dir}/src/"
                        "${tbb_src_dir}/src/rml/include"
                        "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB lib_srcs "${tbb_src_dir}/src/tbb/*.cpp")
file(GLOB lib_hdrs "${tbb_src_dir}/src/tbb/*.h")
list(APPEND lib_srcs "${tbb_src_dir}/src/rml/client/rml_tbb.cpp")
ocv_list_filterout(lib_srcs "${tbb_src_dir}/src/tbb/tbbbind.cpp")  # hwloc.h requirement
ocv_list_filterout(lib_srcs "${tbb_src_dir}/src/tbb/tbb_bind.cpp")  # hwloc.h requirement 2020.1+

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

if(HAVE_PTHREAD)
  add_definitions(-DUSE_PTHREAD) #required for Unix
endif()

if(CV_GCC)
  add_definitions(-DTBB_USE_GCC_BUILTINS=1) #required for ARM GCC
  if(NOT CMAKE_CXX_COMPILER_VERSION LESS 6.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flifetime-dse=1") # workaround for GCC 6.x
  endif()
endif()

if(ANDROID_COMPILER_IS_CLANG)
  add_definitions(-D__TBB_GCC_BUILTIN_ATOMICS_PRESENT=1)
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS
    /wd4702
    -Wshadow
    -Wunused-parameter
    -Wclass-memaccess                  # TBB 2018 under GCC 8+
    -Wimplicit-fallthrough             # TBB 2018 under GCC 7+
    -Wmissing-prototypes               # MacOSX, Android/Clang
    -Wundef -Wmissing-declarations     # TBB 2019
)

set(TBB_SOURCE_FILES ${lib_srcs} ${lib_hdrs})

set(tbb_version_file "version_string.ver")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${tbb_version_file}.cmakein" "${CMAKE_CURRENT_BINARY_DIR}/${tbb_version_file}" @ONLY)
list(APPEND TBB_SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/${tbb_version_file}")

add_library(tbb ${TBB_SOURCE_FILES})
target_compile_definitions(tbb PUBLIC
    TBB_USE_GCC_BUILTINS=1
    __TBB_GCC_BUILTIN_ATOMICS_PRESENT=1
    TBB_SUPPRESS_DEPRECATED_MESSAGES=1
)
target_include_directories(tbb SYSTEM PUBLIC $<BUILD_INTERFACE:${tbb_src_dir}/include>
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
)

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

ocv_install_3rdparty_licenses(tbb "${tbb_src_dir}/LICENSE" "${tbb_src_dir}/README")

ocv_tbb_read_version("${tbb_src_dir}/include")
