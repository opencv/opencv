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


//-----------------------------------------------------------------------------
//
//	class PreviewImageAttribute
//
//-----------------------------------------------------------------------------

#include <ImfPreviewImageAttribute.h>


namespace Imf {


template <>
const char *
PreviewImageAttribute::staticTypeName ()
{
    return "preview";
}


template <>
void
PreviewImageAttribute::writeValueTo (OStream &os, int) const
{
    Xdr::write <StreamIO> (os, _value.width());
    Xdr::write <StreamIO> (os, _value.height());

    int numPixels = _value.width() * _value.height();
    const PreviewRgba *pixels = _value.pixels();

    for (int i = 0; i < numPixels; ++i)
    {
    Xdr::write <StreamIO> (os, pixels[i].r);
    Xdr::write <StreamIO> (os, pixels[i].g);
    Xdr::write <StreamIO> (os, pixels[i].b);
    Xdr::write <StreamIO> (os, pixels[i].a);
    }
}


template <>
void
PreviewImageAttribute::readValueFrom (IStream &is, int, int)
{
    int width, height;

    Xdr::read <StreamIO> (is, width);
    Xdr::read <StreamIO> (is, height);

    PreviewImage p (width, height);

    int numPixels = p.width() * p.height();
    PreviewRgba *pixels = p.pixels();

    for (int i = 0; i < numPixels; ++i)
    {
    Xdr::read <StreamIO> (is, pixels[i].r);
    Xdr::read <StreamIO> (is, pixels[i].g);
    Xdr::read <StreamIO> (is, pixels[i].b);
    Xdr::read <StreamIO> (is, pixels[i].a);
    }

    _value = p;
}


} // namespace Imf
