//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfOutputPartData.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


OutputPartData::OutputPartData(OutputStreamMutex* mutex, const Header &header,
                               int partNumber, int numThreads, bool multipart):
        header(header),
        numThreads(numThreads),
        partNumber(partNumber),
        multipart(multipart),
        mutex(mutex)
{
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
