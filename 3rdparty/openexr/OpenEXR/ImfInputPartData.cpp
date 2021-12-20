//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfInputPartData.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

InputPartData::InputPartData(InputStreamMutex* mutex, const Header &header,
                             int partNumber, int numThreads, int version):
        header(header),
        numThreads(numThreads),
        partNumber(partNumber),
        version(version),       
        mutex(mutex),
        completed(false)
{
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
