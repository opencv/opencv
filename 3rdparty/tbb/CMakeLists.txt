#Cross compile TBB from source
project(tbb)

if (WIN32 AND NOT ARM)
  message(FATAL_ERROR "BUILD_TBB option supports Windows on ARM only!\nUse regular official TBB build instead of the BUILD_TBB option!")
endif()

set(tbb_ver "tbb44_20160128oss")
set(tbb_filename "4.4.3.tar.gz")
set(tbb_subdir "tbb-4.4.3")
set(tbb_md5 "8e7200af3ac16e91a0d1535c606a485c")
set(tbb_version_file "version_string.ver")
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4702)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow)

set(tbb_src_dir "${OpenCV_BINARY_DIR}/3rdparty/tbb")
ocv_download(FILENAME ${tbb_filename}
             HASH ${tbb_md5}
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
set(tbb_src_dir "${tbb_src_dir}/${tbb_subdir}")

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
  if (NOT CMAKE_OPENCV_GCC_VERSION_NUM LESS 600)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flifetime-dse=1") # workaround for GCC 6.x
  endif()
endif()

if(ANDROID_COMPILER_IS_CLANG)
  add_definitions(-D__TBB_GCC_BUILTIN_ATOMICS_PRESENT=1)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wmissing-prototypes)
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
target_compile_definitions(tbb PUBLIC TBB_USE_GCC_BUILTINS=1 __TBB_GCC_BUILTIN_ATOMICS_PRESENT=1)
target_include_directories(tbb SYSTEM PUBLIC $<BUILD_INTERFACE:${tbb_src_dir}/include>)

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

ocv_tbb_read_version("${tbb_src_dir}/include")
