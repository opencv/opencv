///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
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
// *       Neither the name of Industrial Light & Magic nor the names of
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



#ifndef INCLUDED_IMF_HUF_H
#define INCLUDED_IMF_HUF_H


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

namespace Imf {


int
hufCompress (const unsigned short raw[/*nRaw*/],
         int nRaw,
         char compressed[/*2 * nRaw + 65536*/]);


void
hufUncompress (const char compressed[/*nCompressed*/],
           int nCompressed,
           unsigned short raw[/*nRaw*/],
           int nRaw);


} // namespace Imf

#endif
