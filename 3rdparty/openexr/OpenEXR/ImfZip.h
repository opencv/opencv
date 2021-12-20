//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_ZIP_H
#define INCLUDED_IMF_ZIP_H

#include "ImfNamespace.h"
#include "ImfExport.h"

#include <cstddef>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class Zip
{
    public:
        explicit Zip (size_t rawMaxSize, int level);
        Zip (size_t maxScanlineSize, size_t numScanLines, int level);
        ~Zip();

        Zip (const Zip& other) = delete;
        Zip& operator = (const Zip& other) = delete;
        Zip (Zip&& other) = delete;
        Zip& operator = (Zip&& other) = delete;

        size_t maxRawSize();
        size_t maxCompressedSize();

        //
        // Compress the raw data into the provided buffer.
        // Returns the amount of compressed data.
        //
        int compress(const char *raw, int rawSize, char *compressed);

        // 
        // Uncompress the compressed data into the provided
        // buffer. Returns the amount of raw data actually decoded.
        //
        int uncompress(const char *compressed, int compressedSize,
                                                 char *raw);

    private:
        size_t _maxRawSize;
        char  *_tmpBuffer;
        int    _zipLevel;
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
