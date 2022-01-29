set(ade_commit_id "58b2595a1a95cc807be8bf6222f266a9a1f393a9") # tag: v0.1.1f
set(ade_filename "${ade_commit_id}.zip")
set(ade_src_dir "${OpenCV_BINARY_DIR}/3rdparty/ade")
set(ade_subdir "ade-${ade_commit_id}")

# Github
set(OPENCV_ADE_GITHUB_URL "https://github.com/opencv/ade/archive/")
set(ade_GITHUB_md5 "0ebc8ee5486a005050b2a31ab1104b97")
# Gitcode
set(OPENCV_ADE_GITCODE_URL "https://gitcode.net/opencv/ade/-/archive/")
set(ade_GITCODE_md5 "862c207ca796e3f9be305c0cac29cad4")
# Custom
set(OPENCV_ADE_CUSTOM_URL "https://${OPENCV_MIRROR_CUSTOM}/opencv/ade/-/archive/"
set(ade_CUSTOM_md5 "")
if(NOT ade_CUSTOM_md5)
  message(STATUS "ADE: Need to assign ade_CUSTOM_md5 before downloading from custom source. Switching back to Github.")
  set(OPENCV_ADE_CUSTOM_URL "${OPENCV_ADE_GITHUB_URL}")
  set(ade_CUSTOM_md5 "${ade_GITHUB_md5}")
endif()

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

set(ADE_root "${ade_src_dir}/${ade_subdir}/sources/ade")
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