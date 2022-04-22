# Gitlab-style mirror
# CMake scripts look for opencv/opencv_3rdparty,
#  OAID/Tengine, 01org/tbb(oneAPI/oneTBB), opencv/ade
#  from OPENCV_DOWNLOAD_MIRROR
ocv_update(OPENCV_DOWNLOAD_MIRROR_URL "")

######
# Download via commit id
######
# Tengine
ocv_update(TENGINE_PKG_MD5_CUSTOM "")
ocv_update(TENGINE_PKG_MD5_ORIGINAL 23f61ebb1dd419f1207d8876496289c5) # same as tengine_md5sum for TENGINE commit of e89cf8870de2ff0a80cfe626c0b52b2a16fb302e
# NVIDIA_OPTICAL_FLOW
ocv_update(NVIDIA_OPTICAL_FLOW_PKG_MD5_GITCODE "")
ocv_update(NVIDIA_OPTICAL_FLOW_PKG_MD5_ORIGINAL a73cd48b18dcc0cc8933b30796074191)
# TIM-VX
ocv_update(TIM-VX_PKG_MD5_GITCODE "")
ocv_update(TIM-VX_PKG_MD5_ORIGINAL 92619cc4498014ac7a09834d5e33ebd5)

######
# Download from release page
#####
# TBB
ocv_update(TBB_RELEASE_CUSTOM "")
ocv_update(TBB_PKG_NAME_CUSTOM "")
ocv_update(TBB_PKG_MD5_CUSTOM "")
ocv_update(TBB_PKG_MD5_ORIGINAL 5af6f6c2a24c2043e62e47205e273b1f) # same as OPENCV_TBB_RELEASE_MD5 for TBB release of v2020.2
# ADE
ocv_update(ADE_RELEASE_CUSTOM "")
ocv_update(ADE_PKG_NAME_CUSTOM "")
ocv_update(ADE_PKG_MD5_CUSTOM "")
ocv_update(ADE_PKG_MD5_ORIGINAL b624b995ec9c439cbc2e9e6ee940d3a2) # same as ade_md5 for ADE release of v0.1.1f

macro(ocv_download_url_custom_usercontent OWNER)
  string(REPLACE "/" ";" DL_URL_split ${DL_URL})
  list(GET DL_URL_split 5 __COMMIT_ID)
  list(GET DL_URL_split 6 __PKG_NAME)
  set(DL_URL "https://${OPENCV_DOWNLOAD_MIRROR_URL}/${OWNER}/opencv_3rdparty/-/raw/${__COMMIT_ID}/${__PKG_NAME}/")
endmacro()
macro(ocv_download_url_custom_archive_commit_id)
  if("m${${DL_ID}_PKG_MD5_CUSTOM}" STREQUAL "m")
    message(WARNING "ocv_download: specify ${DL_ID}_PKG_MD5_CUSTOM to download ${DL_ID} from custom source.")
  elseif(${DL_ID}_PKG_MD5_ORIGINAL STREQUAL "${DL_HASH}")
    string(REPLACE "/" ";" DL_URL_split ${DL_URL})
    list(GET DL_URL_split 3 __OWNER)
    list(GET DL_URL_split 4 __REPO_NAME)
    set(DL_URL "https://${OPENCV_DOWNLOAD_MIRROR_URL}/${__OWNER}/${__REPO_NAME}/-/archive/")
    set(DL_HASH "${${DL_ID}_PKG_MD5_CUSTOM}")
  else()
    message(WARNING "No information about mirrors for downloading ${DL_FILENAME} from URL='${DL_URL}' and MD5=${DL_HASH}.")
  endif()
endmacro()
macro(ocv_download_url_custom_archive_release)
    if("m${${DL_ID}_RELEASE_CUSTOM}" STREQUAL "m")
      message(WARNING "ocv_download: specify ${DL_ID}_RELEASE_CUSTOM to download ${DL_ID} from custom source.")
      return()
    endif()
    if("m${${DL_ID}_PKG_NAME_CUSTOM}" STREQUAL "m")
      message(WARNING "ocv_download: specify ${DL_ID}_PKG_NAME_CUSTOM to download ${DL_ID} from custom source.")
      return()
    endif()
    if("m${${DL_ID}_PKG_MD5_CUSTOM}" STREQUAL "m")
      message(WARNING "ocv_download: specify ${DL_ID}_PKG_MD5_CUSTOM to download ${DL_ID} from custom source.")
      return()
    endif()
    string(REPLACE "/" ";" DL_URL_split ${DL_URL})
    list(GET DL_URL_split 3 __OWNER)
    list(GET DL_URL_split 4 __REPO_NAME)
    set(DL_URL "https://${OPENCV_DOWNLOAD_MIRROR_URL}/${__OWNER}/${__REPO_NAME}/-/archive/${${DL_ID}_RELEASE_CUSTOM}/${__REPO_NAME}-")
    set(DL_HASH "${${DL_ID}_PKG_MD5_CUSTOM}")
endmacro()

if("m${OPENCV_DOWNLOAD_MIRROR_URL}" STREQUAL "m")
  message(WARNING "ocv_download: specify OPENCV_DOWNLOAD_MIRROR_URL to use custom mirror.")
else()
  if((DL_ID STREQUAL "FFMPEG") OR (DL_ID STREQUAL "IPPICV") OR (DL_ID STREQUAL "data") OR (DL_ID STREQUAL "xfeatures2d/boostdesc") OR (DL_ID STREQUAL "xfeatures2d/vgg"))
    ocv_download_url_custom_usercontent(opencv)
  elseif(DL_ID STREQUAL "wechat_qrcode")
    ocv_download_url_gitcode_usercontent(WeChatCV)
  elseif((DL_ID STREQUAL "TENGINE") OR (DL_ID STREQUAL "NVIDIA_OPTICAL_FLOW") OR (DL_ID STREQUAL "TIM-VX"))
    ocv_download_url_custom_archive_commit_id()
  elseif(DL_ID STREQUAL "TBB")
    ocv_download_url_custom_archive_release()
    set(OPENCV_TBB_SUBDIR "${TBB_PKG_NAME_CUSTOM}" PARENT_SCOPE)
  elseif(DL_ID STREQUAL "ADE")
    ocv_download_url_custom_archive_release()
    set(ade_subdir "${ADE_PKG_NAME_CUSTOM}" PARENT_SCOPE)
  else()
    message(STATUS "ocv_download: Unknown download ID ${DL_ID} for using mirror ${OPENCV_DOWNLOAD_MIRROR_URL}. Use original source instead.")
  endif()
endif()
