set(ade_src_dir "${OpenCV_BINARY_DIR}/3rdparty/ade")
set(ade_filename "v0.1.2d.zip")
set(ade_subdir "ade-0.1.2d")
set(ade_md5 "dbb095a8bf3008e91edbbf45d8d34885")
ocv_download(FILENAME ${ade_filename}
             HASH ${ade_md5}
             URL
               "${OPENCV_ADE_URL}"
               "$ENV{OPENCV_ADE_URL}"
               "https://github.com/opencv/ade/archive/"
             DESTINATION_DIR ${ade_src_dir}
             ID ADE
             STATUS res
             UNPACK RELATIVE_URL)

if (NOT res)
    return()
endif()

set(ADE_root "${ade_src_dir}/${ade_subdir}/sources/ade")
file(GLOB_RECURSE ADE_sources "${ADE_root}/source/*.cpp")
file(GLOB_RECURSE ADE_include "${ADE_root}/include/ade/*.hpp")
add_library(ade STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL}
    ${ADE_include}
    ${ADE_sources}
)

# https://github.com/opencv/ade/issues/32
if(CV_CLANG AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.1)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wdeprecated-copy)
endif()

target_include_directories(ade PUBLIC $<BUILD_INTERFACE:${ADE_root}/include>)
set_target_properties(ade PROPERTIES
  POSITION_INDEPENDENT_CODE True
  OUTPUT_NAME ade
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ade
  COMPILE_PDB_NAME_DEBUG "ade${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(ade PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(ade EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()

ocv_install_3rdparty_licenses(ade "${ade_src_dir}/${ade_subdir}/LICENSE")
