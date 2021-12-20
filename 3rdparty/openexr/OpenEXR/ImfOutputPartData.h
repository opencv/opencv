//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMFOUTPUTPARTDATA_H_
#define IMFOUTPUTPARTDATA_H_

#include "ImfForward.h"

#include "ImfHeader.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

struct OutputPartData
{
    Header                  header;
    uint64_t                chunkOffsetTablePosition;
    uint64_t                previewPosition;
    int                     numThreads;
    int                     partNumber;
    bool                    multipart;
    OutputStreamMutex*      mutex;

    OutputPartData(OutputStreamMutex* mutex, const Header &header,
                   int partNumber, int numThreads, bool multipart);

};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif /* IMFOUTPUTPARTDATA_H_ */
