set(ade_src_dir "${OpenCV_BINARY_DIR}/3rdparty/ade")
set(ade_filename "v0.1.1f.zip")
set(ade_GITHUB_md5 "b624b995ec9c439cbc2e9e6ee940d3a2")
set(ade_GITCODE_md5 "aa2ec43abe534f8b1db13274b5ed92dd")
set(ade_GITHUB_subdir "ade-0.1.1f")
set(ade_GITCODE_subdir "ade-v0.1.1f-58b2595a1a95cc807be8bf6222f266a9a1f393a9") # suffix is the commit id

set(OPENCV_ADE_GITHUB_URL "https://github.com/opencv/ade/archive/")
set(OPENCV_ADE_GITCODE_URL "https://gitcode.net/opencv/ade/-/archive/")

ocv_download(FILENAME ${ade_filename}
             HASH ${ade_${OPENCV_DOWNLOAD_HOST}_md5}
             URL
               "${OPENCV_ADE_URL}"
               "$ENV{OPENCV_ADE_URL}"
               "${OPENCV_ADE_${OPENCV_DOWNLOAD_HOST}_URL}"
             DESTINATION_DIR ${ade_src_dir}
             ID ADE
             STATUS res
             UNPACK RELATIVE_URL)

if (NOT res)
    return()
endif()

set(ADE_root "${ade_src_dir}/${ade_${OPENCV_DOWNLOAD_HOST}_subdir}/sources/ade")
message(STATUS "OCV_DOWNLOAD: ADE ${ADE_root}")
file(GLOB_RECURSE ADE_sources "${ADE_root}/source/*.cpp")
file(GLOB_RECURSE ADE_include "${ADE_root}/include/ade/*.hpp")
add_library(ade STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL}
    ${ADE_include}
    ${ADE_sources}
)
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
