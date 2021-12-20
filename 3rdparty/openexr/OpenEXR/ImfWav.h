//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_WAV_H
#define INCLUDED_IMF_WAV_H

//-----------------------------------------------------------------------------
//
//	16-bit Haar Wavelet encoding and decoding
//
//-----------------------------------------------------------------------------
#include "ImfNamespace.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


IMF_EXPORT 
void
wav2Encode
    (unsigned short *in, // io: values in[y][x] are transformed in place
     int     nx,	 // i : x size
     int     ox,	 // i : x offset
     int     ny,	 // i : y size
     int     oy,	 // i : y offset
     unsigned short mx); // i : maximum in[x][y] value

IMF_EXPORT
void
wav2Decode
    (unsigned short *in, // io: values in[y][x] are transformed in place
     int     nx,	 // i : x size
     int     ox,	 // i : x offset
     int     ny,	 // i : y size
     int     oy,	 // i : y offset
     unsigned short mx); // i : maximum in[x][y] value


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
