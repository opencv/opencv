//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_RLE_H_
#define INCLUDED_IMF_RLE_H_

#include "ImfNamespace.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Compress an array of bytes, using run-length encoding,
// and return the length of the compressed data.
//

IMF_EXPORT
int rleCompress (int inLength, const char in[], signed char out[]);

//
// Uncompress an array of bytes compressed with rleCompress().
// Returns the length of the uncompressed data, or 0 if the
// length of the uncompressed data would be more than maxLength.
//

IMF_EXPORT
int rleUncompress (int inLength, int maxLength,
                                 const signed char in[], char out[]);

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
