# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Contributors to the OpenEXR Project.

openexr_define_library(IlmThread
  PRIV_EXPORT ILMTHREAD_EXPORTS
  CURDIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
    IlmThread.cpp
    IlmThreadPool.cpp
    IlmThreadSemaphore.cpp
    IlmThreadSemaphoreOSX.cpp
    IlmThreadSemaphorePosix.cpp
    IlmThreadSemaphorePosixCompat.cpp
    IlmThreadSemaphoreWin32.cpp
  HEADERS
    IlmThread.h
    IlmThreadExport.h
    IlmThreadForward.h
    IlmThreadMutex.h
    IlmThreadNamespace.h
    IlmThreadPool.h
    IlmThreadSemaphore.h
  DEPENDENCIES
    OpenEXR::Config
    OpenEXR::Iex
)

if(OPENEXR_ENABLE_THREADING)
  target_link_libraries(IlmThread PUBLIC Threads::Threads)
endif()

