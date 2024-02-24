# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Contributors to the OpenEXR Project.

openexr_define_library(Iex
  PRIV_EXPORT IEX_EXPORTS
  CURDIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
    IexMathFpu.h
    IexBaseExc.cpp
    IexMathFloatExc.cpp
    IexMathFpu.cpp
    IexThrowErrnoExc.cpp
  HEADERS
    Iex.h
    IexBaseExc.h
    IexErrnoExc.h
    IexExport.h
    IexForward.h
    IexMacros.h
    IexMathExc.h
    IexMathFloatExc.h
    IexMathIeeeExc.h
    IexNamespace.h
    IexThrowErrnoExc.h
  DEPENDENCIES
    OpenEXR::Config
  )
