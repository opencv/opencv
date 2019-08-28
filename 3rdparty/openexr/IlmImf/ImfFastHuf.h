///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2009-2014 DreamWorks Animation LLC. 
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

#ifndef INCLUDED_IMF_FAST_HUF_H
#define INCLUDED_IMF_FAST_HUF_H

#include "ImfInt64.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

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

    IMF_EXPORT
    FastHufDecoder (const char*& table,
                    int numBytes,
                    int minSymbol,
                    int maxSymbol,
                    int rleSymbol);

    IMF_EXPORT
    ~FastHufDecoder ();

    IMF_EXPORT
    static bool enabled ();

    IMF_EXPORT
    void decode (const unsigned char *src,
                 int numSrcBits,
                 unsigned short *dst,
                 int numDstElems);

  private:

    void  buildTables (Int64*, Int64*);
    void  refill (Int64&, int, Int64&, int&, const unsigned char *&, int&);
    Int64 readBits (int, Int64&, int&, const char *&);

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

    Int64 _ljBase[MAX_CODE_LEN + 1];   // the 'left justified base' table.
                                       // Takes base[i] (i = code length)
                                       // and 'left justifies' it into an Int64

    Int64 _ljOffset[MAX_CODE_LEN +1 ]; // There are some other terms that can 
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
    Int64          _tableMin;
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif 
