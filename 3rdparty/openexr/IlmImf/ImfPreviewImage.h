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


#ifndef INCLUDED_IMF_PREVIEW_IMAGE_H
#define INCLUDED_IMF_PREVIEW_IMAGE_H

#include "ImfNamespace.h"
#include "ImfExport.h"

//-----------------------------------------------------------------------------
//
//	class PreviewImage -- a usually small, low-dynamic range image,
//	that is intended to be stored in an image file's header.
//
//	struct PreviewRgba -- holds the value of a PreviewImage pixel.
//
//-----------------------------------------------------------------------------


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


struct IMF_EXPORT PreviewRgba
{
    unsigned char	r;	// Red, green and blue components of
    unsigned char	g;	// the pixel's color; intensity is
    unsigned char	b;	// proportional to pow (x/255, 2.2),
    				// where x is r, g, or b.

    unsigned char	a;	// The pixel's alpha; 0 == transparent,
				// 255 == opaque.

    PreviewRgba (unsigned char r = 0,
		 unsigned char g = 0,
		 unsigned char b = 0,
		 unsigned char a = 255)
	: r(r), g(g), b(b), a(a) {}
};


class IMF_EXPORT PreviewImage
{
  public:

    //--------------------------------------------------------------------
    // Constructor:
    //
    // PreviewImage(w,h,p) constructs a preview image with w by h pixels
    // whose initial values are specified in pixel array p.  The x and y
    // coordinates of the pixels in p go from 0 to w-1, and from 0 to h-1.
    // The pixel with coordinates (x, y) is at address p + y*w + x.
    // Pixel (0, 0) is in the upper left corner of the preview image.
    // If p is zero, the pixels in the preview image are initialized with
    // (r = 0, b = 0, g = 0, a = 255).
    //
    //--------------------------------------------------------------------
   
     PreviewImage (unsigned int width = 0,
		   unsigned int height = 0,
		   const PreviewRgba pixels[] = 0);

    //-----------------------------------------------------
    // Copy constructor, destructor and assignment operator
    //-----------------------------------------------------

     PreviewImage (const PreviewImage &other);
    ~PreviewImage ();

    PreviewImage &	operator = (const PreviewImage &other);


    //-----------------------------------------------
    // Access to width, height and to the pixel array
    //-----------------------------------------------

    unsigned int	width () const	{return _width;}
    unsigned int	height () const	{return _height;}

    PreviewRgba *	pixels ()	{return _pixels;}
    const PreviewRgba *	pixels () const	{return _pixels;}


    //----------------------------
    // Access to individual pixels
    //----------------------------

    PreviewRgba &	pixel (unsigned int x, unsigned int y)
    					{return _pixels[y * _width + x];}

    const PreviewRgba &	pixel (unsigned int x, unsigned int y) const
    					{return _pixels[y * _width + x];}

  private:

    unsigned int	_width;
    unsigned int	_height;
    PreviewRgba *	_pixels;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
