# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Contributors to the OpenEXR Project.

openexr_define_library(OpenEXRUtil
  PRIV_EXPORT OPENEXRUTIL_EXPORTS
  CURDIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
    ImfCheckFile.cpp
    ImfDeepImage.cpp
    ImfDeepImageChannel.cpp
    ImfDeepImageIO.cpp
    ImfDeepImageLevel.cpp
    ImfFlatImage.cpp
    ImfFlatImageChannel.cpp
    ImfFlatImageIO.cpp
    ImfFlatImageLevel.cpp
    ImfImage.cpp
    ImfImageChannel.cpp
    ImfImageDataWindow.cpp
    ImfImageIO.cpp
    ImfImageLevel.cpp
    ImfSampleCountChannel.cpp
  HEADERS
    ImfCheckFile.h
    ImfDeepImage.h
    ImfDeepImageChannel.h
    ImfDeepImageIO.h
    ImfDeepImageLevel.h
    ImfFlatImage.h
    ImfFlatImageChannel.h
    ImfFlatImageIO.h
    ImfFlatImageLevel.h
    ImfImage.h
    ImfImageChannel.h
    ImfImageChannelRenaming.h
    ImfImageDataWindow.h
    ImfImageIO.h
    ImfImageLevel.h
    ImfSampleCountChannel.h
    ImfUtilExport.h
  DEPENDENCIES
    OpenEXR::OpenEXR
    OpenEXR::OpenEXRCore
)
