///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
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

#include "ImfZip.h"
#include "ImfCheckedArithmetic.h"
#include "ImfNamespace.h"
#include "Iex.h"

#include <math.h>
#include <zlib.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

Imf::Zip::Zip(size_t maxRawSize):
    _maxRawSize(maxRawSize),
    _tmpBuffer(0)
{
    _tmpBuffer = new char[_maxRawSize];
}

Imf::Zip::Zip(size_t maxScanLineSize, size_t numScanLines):
    _maxRawSize(0),
    _tmpBuffer(0)
{
    _maxRawSize = uiMult (maxScanLineSize, numScanLines);
    _tmpBuffer  = new char[_maxRawSize];
}

Imf::Zip::~Zip()
{
    if (_tmpBuffer) delete[] _tmpBuffer;
}

size_t
Imf::Zip::maxRawSize()
{
    return _maxRawSize;
}

size_t
Imf::Zip::maxCompressedSize()
{
    return uiAdd (uiAdd (_maxRawSize,
               size_t (ceil (_maxRawSize * 0.01))),
                  size_t (100));
}

int
Imf::Zip::compress(const char *raw, int rawSize, char *compressed)
{
    //
    // Reorder the pixel data.
    //

    {
        char *t1 = _tmpBuffer;
        char *t2 = _tmpBuffer + (rawSize + 1) / 2;
        const char *stop = raw + rawSize;

        while (true)
        {
            if (raw < stop)
            *(t1++) = *(raw++);
            else
            break;

            if (raw < stop)
            *(t2++) = *(raw++);
            else
            break;
        }
    }

    //
    // Predictor.
    //

    {
        unsigned char *t    = (unsigned char *) _tmpBuffer + 1;
        unsigned char *stop = (unsigned char *) _tmpBuffer + rawSize;
        int p = t[-1];

        while (t < stop)
        {
            int d = int (t[0]) - p + (128 + 256);
            p = t[0];
            t[0] = d;
            ++t;
        }
    }

    //
    // Compress the data using zlib
    //

    uLongf outSize = int(ceil(rawSize * 1.01)) + 100;

    if (Z_OK != ::compress ((Bytef *)compressed, &outSize,
                (const Bytef *) _tmpBuffer, rawSize))
    {
        throw Iex::BaseExc ("Data compression (zlib) failed.");
    }

    return outSize;
}

int
Imf::Zip::uncompress(const char *compressed, int compressedSize,
                                            char *raw)
{
    //
    // Decompress the data using zlib
    //

    uLongf outSize = _maxRawSize;

    if (Z_OK != ::uncompress ((Bytef *)_tmpBuffer, &outSize,
                     (const Bytef *) compressed, compressedSize))
    {
        throw Iex::InputExc ("Data decompression (zlib) failed.");
    }

    //
    // Predictor.
    //
    {
        unsigned char *t    = (unsigned char *) _tmpBuffer + 1;
        unsigned char *stop = (unsigned char *) _tmpBuffer + outSize;

        while (t < stop)
        {
            int d = int (t[-1]) + int (t[0]) - 128;
            t[0] = d;
            ++t;
        }
    }

    //
    // Reorder the pixel data.
    //

    {
        const char *t1 = _tmpBuffer;
        const char *t2 = _tmpBuffer + (outSize + 1) / 2;
        char *s = raw;
        char *stop = s + outSize;

        while (true)
        {
            if (s < stop)
            *(s++) = *(t1++);
            else
            break;

            if (s < stop)
            *(s++) = *(t2++);
            else
            break;
        }
    }

    return outSize;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
