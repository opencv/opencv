///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2003, Industrial Light & Magic, a division of Lucas
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


//-----------------------------------------------------------------------------
//
//	class PreviewImage
//
//-----------------------------------------------------------------------------

#include <ImfPreviewImage.h>
#include <ImfCheckedArithmetic.h>
#include "Iex.h"

namespace Imf {


PreviewImage::PreviewImage (unsigned int width,
			    unsigned int height,
			    const PreviewRgba pixels[])
{
    _width = width;
    _height = height;
    _pixels = new PreviewRgba
        [checkArraySize (uiMult (_width, _height), sizeof (PreviewRgba))];

    if (pixels)
    {
	for (unsigned int i = 0; i < _width * _height; ++i)
	    _pixels[i] = pixels[i];
    }
    else
    {
	for (unsigned int i = 0; i < _width * _height; ++i)
	    _pixels[i] = PreviewRgba();
    }
}


PreviewImage::PreviewImage (const PreviewImage &other):
    _width (other._width),
    _height (other._height),
    _pixels (new PreviewRgba [other._width * other._height])
{
    for (unsigned int i = 0; i < _width * _height; ++i)
	_pixels[i] = other._pixels[i];
}


PreviewImage::~PreviewImage ()
{
    delete [] _pixels;
}


PreviewImage &
PreviewImage::operator = (const PreviewImage &other)
{
    delete [] _pixels;

    _width = other._width;
    _height = other._height;
    _pixels = new PreviewRgba [other._width * other._height];

    for (unsigned int i = 0; i < _width * _height; ++i)
	_pixels[i] = other._pixels[i];

    return *this;
}


} // namespace Imf
