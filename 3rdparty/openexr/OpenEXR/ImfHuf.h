//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_HUF_H
#define INCLUDED_IMF_HUF_H

#include "ImfExport.h"
#include "ImfNamespace.h"

//-----------------------------------------------------------------------------
//
//	16-bit Huffman compression and decompression:
//
//	hufCompress (r, nr, c)
//
//		Compresses the contents of array r (of length nr),
//		stores the compressed data in array c, and returns
//		the size of the compressed data (in bytes).
//
//		To avoid buffer overflows, the size of array c should
//		be at least 2 * nr + 65536.
//
//	hufUncompress (c, nc, r, nr)
//
//		Uncompresses the data in array c (with length nc),
//		and stores the results in array r (with length nr).
//
//-----------------------------------------------------------------------------

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


IMF_EXPORT 
int
hufCompress (const unsigned short raw[/*nRaw*/],
	     int nRaw,
	     char compressed[/*2 * nRaw + 65536*/]);

IMF_EXPORT
void
hufUncompress (const char compressed[/*nCompressed*/],
	       int nCompressed,
	       unsigned short raw[/*nRaw*/],
	       int nRaw);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
