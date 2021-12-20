//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_PREVIEW_IMAGE_H
#define INCLUDED_IMF_PREVIEW_IMAGE_H

#include "ImfForward.h"

//-----------------------------------------------------------------------------
//
//	class PreviewImage -- a usually small, low-dynamic range image,
//	that is intended to be stored in an image file's header.
//
//	struct PreviewRgba -- holds the value of a PreviewImage pixel.
//
//-----------------------------------------------------------------------------


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


struct IMF_EXPORT_TYPE PreviewRgba
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


class IMF_EXPORT_TYPE PreviewImage
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
   
    IMF_EXPORT
     PreviewImage (unsigned int width = 0,
		   unsigned int height = 0,
		   const PreviewRgba pixels[] = 0);

    //-----------------------------------------------------
    // Copy constructor, destructor and assignment operator
    //-----------------------------------------------------

    IMF_EXPORT
     PreviewImage (const PreviewImage &other);
    IMF_EXPORT
    ~PreviewImage ();

    IMF_EXPORT
    PreviewImage &	operator = (const PreviewImage &other);


    //-----------------------------------------------
    // Access to width, height and to the pixel array
    //-----------------------------------------------

    inline
    unsigned int	width () const	{return _width;}
    inline
    unsigned int	height () const	{return _height;}

    inline
    PreviewRgba *	pixels ()	{return _pixels;}
    inline
    const PreviewRgba *	pixels () const	{return _pixels;}


    //----------------------------
    // Access to individual pixels
    //----------------------------

    inline
    PreviewRgba &	pixel (unsigned int x, unsigned int y)
    					{return _pixels[y * _width + x];}

    inline
    const PreviewRgba &	pixel (unsigned int x, unsigned int y) const
    					{return _pixels[y * _width + x];}

  private:

    unsigned int	_width;
    unsigned int	_height;
    PreviewRgba *	_pixels;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
