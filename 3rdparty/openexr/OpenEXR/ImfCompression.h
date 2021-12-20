//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_COMPRESSION_H
#define INCLUDED_IMF_COMPRESSION_H

//-----------------------------------------------------------------------------
//
//	enum Compression
//
//-----------------------------------------------------------------------------
#include "ImfExport.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

enum IMF_EXPORT_ENUM Compression
{
    NO_COMPRESSION  = 0,	// no compression

    RLE_COMPRESSION = 1,	// run length encoding

    ZIPS_COMPRESSION = 2,	// zlib compression, one scan line at a time

    ZIP_COMPRESSION = 3,	// zlib compression, in blocks of 16 scan lines

    PIZ_COMPRESSION = 4,	// piz-based wavelet compression

    PXR24_COMPRESSION = 5,	// lossy 24-bit float compression

    B44_COMPRESSION = 6,	// lossy 4-by-4 pixel block compression,
    				// fixed compression rate

    B44A_COMPRESSION = 7,	// lossy 4-by-4 pixel block compression,
    				// flat fields are compressed more

    DWAA_COMPRESSION = 8,       // lossy DCT based compression, in blocks
                                // of 32 scanlines. More efficient for partial
                                // buffer access.

    DWAB_COMPRESSION = 9,       // lossy DCT based compression, in blocks
                                // of 256 scanlines. More efficient space
                                // wise and faster to decode full frames
                                // than DWAA_COMPRESSION.

    NUM_COMPRESSION_METHODS	// number of different compression methods
};

/// Controls the default zip compression level used. Zip is used for
/// the 2 zip levels as well as some modes of the DWAA/B compression.
IMF_EXPORT void setDefaultZipCompressionLevel (int level);

/// Controls the default quality level for the DWA lossy compression
IMF_EXPORT void setDefaultDwaCompressionLevel (float level);

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
