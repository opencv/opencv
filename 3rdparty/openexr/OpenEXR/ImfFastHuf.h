//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) DreamWorks Animation LLC and Contributors of the OpenEXR Project
//

#ifndef INCLUDED_IMF_FAST_HUF_H
#define INCLUDED_IMF_FAST_HUF_H

#include "ImfNamespace.h"

#include <cstdint>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Alternative Canonical Huffman decoder:
//
// Canonical Huffman decoder based on 'On the Implementation of Minimum
// Redundancy Prefix Codes' by Moffat and Turpin - highly recommended
// reading as a good description of the problem space, as well as 
// a fast decoding algorithm.
//
// The premise is that instead of working directly with the coded 
// symbols, we create a new ordering based on the frequency of symbols.
// Less frequent symbols (and thus longer codes) are ordered earler.
// We're calling the values in this ordering 'Ids', as oppsed to 
// 'Symbols' - which are the short values we eventually want decoded.
//
// With this new ordering, a few small tables can be derived ('base' 
// and 'offset') which drive the decoding. To cut down on the 
// linear scanning of these tables, you can add a small table
// to directly look up short codes (as you might in a traditional
// lookup-table driven decoder). 
//
// The decoder is meant to be compatible with the encoder (and decoder)
// in ImfHuf.cpp, just faster. For ease of implementation, this decoder
// should only be used on compressed bitstreams >= 128 bits long.
//

class FastHufDecoder
{
  public:

    //
    // Longest compressed code length that ImfHuf supports (58 bits)
    //

    static const int MAX_CODE_LEN = 58;

    //
    // Number of bits in our acceleration table. Should match all
    // codes up to TABLE_LOOKUP_BITS in length.
    //

    static const int TABLE_LOOKUP_BITS = 12;

    FastHufDecoder (const char*& table,
                    int numBytes,
                    int minSymbol,
                    int maxSymbol,
                    int rleSymbol);

    ~FastHufDecoder ();

    FastHufDecoder (const FastHufDecoder& other) = delete;
    FastHufDecoder& operator = (const FastHufDecoder& other) = delete;
    FastHufDecoder (FastHufDecoder&& other) = delete;
    FastHufDecoder& operator = (FastHufDecoder&& other) = delete;

    static bool enabled ();

    void decode (const unsigned char *src,
                 int numSrcBits,
                 unsigned short *dst,
                 int numDstElems);

  private:

    void  buildTables (uint64_t*, uint64_t*);
    void  refill (uint64_t&, int, uint64_t&, int&, const unsigned char *&, int&);
    uint64_t readBits (int, uint64_t&, int&, const char *&);

    int             _rleSymbol;        // RLE symbol written by the encoder.
                                       // This could be 65536, so beware
                                       // when you use shorts to hold things.

    int             _numSymbols;       // Number of symbols in the codebook.

    unsigned char   _minCodeLength;    // Minimum code length, in bits.
    unsigned char   _maxCodeLength;    // Maximum code length, in bits.

    int            *_idToSymbol;       // Maps Ids to symbols. Ids are a symbol
                                       // ordering sorted first in terms of 
                                       // code length, and by code within
                                       // the same length. Ids run from 0
                                       // to mNumSymbols-1.

    uint64_t _ljBase[MAX_CODE_LEN + 1];   // the 'left justified base' table.
                                       // Takes base[i] (i = code length)
                                       // and 'left justifies' it into an uint64_t

    uint64_t _ljOffset[MAX_CODE_LEN +1 ]; // There are some other terms that can 
                                       // be folded into constants when taking
                                       // the 'left justified' decode path. This
                                       // holds those constants, indexed by
                                       // code length

    //
    // We can accelerate the 'left justified' processing by running the
    // top TABLE_LOOKUP_BITS through a LUT, to find the symbol and code
    // length. These are those acceleration tables.
    //
    // Even though our evental 'symbols' are ushort's, the encoder adds
    // a symbol to indicate RLE. So with a dense code book, we could
    // have 2^16+1 codes, so both mIdToSymbol and mTableSymbol need
    // to be bigger than 16 bits.
    //

    int            _tableSymbol[1 << TABLE_LOOKUP_BITS];
    unsigned char  _tableCodeLen[1 << TABLE_LOOKUP_BITS];
    uint64_t       _tableMin;
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif 
