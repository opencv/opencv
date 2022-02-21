# Tengine (Download via commit id)
ocv_update(TENGINE_GITCODE_PKG_MD5 1b5908632b557275cd6e85b0c03f9690)
ocv_update(TENGINE_GITHUB_PKG_MD5 23f61ebb1dd419f1207d8876496289c5) # same as tengine_md5sum for TENGINE commit of e89cf8870de2ff0a80cfe626c0b52b2a16fb302e

# TBB (Download from release page)
ocv_update(TBB_GITCODE_RELEASE "v2020.2")
ocv_update(TBB_GITCODE_PKG_NAME "tbb-${TBB_GITCODE_RELEASE}")
ocv_update(TBB_GITCODE_PKG_MD5 4eeafdf16a90cb66e39a31c8d6c6804e)
ocv_update(TBB_GITHUB_PKG_MD5 5af6f6c2a24c2043e62e47205e273b1f) # same as OPENCV_TBB_RELEASE_MD5 for TBB release of v2020.2

# ADE (Download from release page)
ocv_update(ADE_GITCODE_RELEASE "v0.1.1f")
ocv_update(ADE_GITCODE_PKG_NAME "ade-${ADE_GITCODE_RELEASE}")
ocv_update(ADE_GITCODE_PKG_MD5 c12909e0ccfa93138c820ba91ff37b3c)
ocv_update(ADE_GITHUB_PKG_MD5 b624b995ec9c439cbc2e9e6ee940d3a2) # same as ade_md5 for ADE release of v0.1.1f

#
# Replace download links for packages in opencv/opencv_3rdparty:
# 1. Extract repo owner and repo name from DL_URL.
# 2. Put repo owner and repo name into the placeholders of new DL_URL.
#
macro(ocv_download_url_gitcode_usercontent)
  string(REPLACE "/" ";" DL_URL_split ${DL_URL})
  list(GET DL_URL_split 5 __COMMIT_ID)
  list(GET DL_URL_split 6 __PKG_NAME)
  set(DL_URL "https://gitcode.net/opencv/opencv_3rdparty/-/raw/${__COMMIT_ID}/${__PKG_NAME}/")
endmacro()
#
# Replace download links and checksums for archives/releases in other repositories:
# 1. Check if versions matched. If not matched, download from github instead.
# 2. Extract repo owner and repo name from DL_URL.
# 3. Put repo owner and repo name into the placeholders of new DL_URL.
# 4. Replace DL_HASH with the one downloaded from gitcode.net.
#
macro(ocv_download_url_gitcode_archive_commit_id)
  if(DL_HASH STREQUAL ${${DL_ID}_GITHUB_PKG_MD5})
    string(REPLACE "/" ";" DL_URL_split ${DL_URL})
    list(GET DL_URL_split 3 __OWNER)
    list(GET DL_URL_split 4 __REPO_NAME)
    set(DL_URL "https://gitcode.net/${__OWNER}/${__REPO_NAME}/-/archive/")
    set(DL_HASH "${${DL_ID}_GITCODE_PKG_MD5}")
  else()
    message(WARNING "Package ${DL_ID} from mirror gitcode.net is outdated and will be downloaded from github.com instead.")
  endif()
endmacro()
macro(ocv_download_url_gitcode_archive_release)
  if(DL_HASH STREQUAL ${${DL_ID}_GITHUBPKG_MD5})
    string(REPLACE "/" ";" DL_URL_split ${DL_URL})
    list(GET DL_URL_split 3 __OWNER)
    list(GET DL_URL_split 4 __REPO_NAME)
    set(DL_URL "https://gitcode.net/${__OWNER}/${__REPO_NAME}/-/archive/${${DL_ID}_GITCODE_RELEASE}/${__REPO_NAME}-")
    set(DL_HASH "${${DL_ID}_GITCODE_PKG_MD5}")
  else()
    message(WARNING "Package ${DL_ID} from mirror gitcode.net is outdated and will be downloaded from github.com instead.")
endmacro()

if((DL_ID STREQUAL "FFMPEG") OR (DL_ID STREQUAL "IPPICV"))
  ocv_download_url_gitcode_usercontent()
elseif(DL_ID STREQUAL "TENGINE")
  ocv_download_url_gitcode_archive_commit_id()
elseif(DL_ID STREQUAL "TBB")
  ocv_download_url_gitcode_archive_release()
  set(OPENCV_TBB_SUBDIR "${TBB_GITCODE_PKG_NAME}" PARENT_SCOPE)
elseif(DL_ID STREQUAL "ADE")
  ocv_download_url_gitcode_archive_release()
  set(ade_subdir "${ADE_GITCODE_PKG_NAME}" PARENT_SCOPE)
else()
  message(STATUS "ocv_download: Unknown download ID ${DL_ID} for using mirror gitcode.net. Use original source instead.")
endif()
