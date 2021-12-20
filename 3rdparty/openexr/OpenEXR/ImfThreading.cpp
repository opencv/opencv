//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Threading support for the OpenEXR library
//
//-----------------------------------------------------------------------------

#include "ImfThreading.h"
#include "IlmThreadPool.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


int
globalThreadCount ()
{
    return ILMTHREAD_NAMESPACE::ThreadPool::globalThreadPool().numThreads();
}


void
setGlobalThreadCount (int count)
{
    ILMTHREAD_NAMESPACE::ThreadPool::globalThreadPool().setNumThreads (count);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
