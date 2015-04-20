/*=========================================================================

  Program: librle, a minimal RLE library for DICOM

  Copyright (c) 2014 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "info.h"

#include <stdexcept>
#include <cassert>

namespace rle
{

// The byte of a multibyte number with the greatest importance: that is, the
// byte stored first on a big-endian system or last on a little-endian
// system.

pixel_info::pixel_info(
    unsigned char nc,
    unsigned char bpp
  ):
  number_components(nc),
  bits_per_pixel(bpp)
{
  if ( nc != 1 && nc != 3 )
    throw std::runtime_error( "invalid samples per pixel" );
  if( bpp != 8 && bpp != 16 && bpp != 32 )
    throw std::runtime_error( "invalid bits per pixel" );
}

static inline int compute_num_segments_impl( int nc, int bpp )
{
  // pre
  assert( bpp % 8 == 0 && (nc == 1 || nc == 3) );
  const int mult = bpp / 8;
  const int res = nc * mult;
  // post
  assert( res <= 12 && res > 0 );
  return res;
}

bool pixel_info::check_num_segments( const int num_segments )
{
  bool ok = false;
  // DICOM restrict number of possible values:
  switch( num_segments )
    {
    case 1:  // 1 -> Grayscale / 8bits
    case 2:  // 2 -> 16bits
    case 3:  // 3 -> RGB / 8bits
    case 4:  // 4 -> 32bits
    case 6:  // 6 -> RGB / 16bits
    case 12: // 12 -> RGB / 32 bits
      ok = true;
      break;
    }
  return ok;
}

static inline void compute_nc_bpp_from_num_segments( const int num_segments, int & nc, int & bpp )
{
  // pre condition:
  assert( pixel_info::check_num_segments( num_segments ) );

  if( num_segments % 3 == 0 )
    {
    nc = 3;
    bpp = ( num_segments / 3 ) * 8;
    }
  else
    {
    nc = 1;
    bpp = num_segments;
    }

  // post condition
  assert( compute_num_segments_impl( nc, bpp ) == num_segments );
}

pixel_info::pixel_info( int num_segments )
{
  int nc, bpp;
  compute_nc_bpp_from_num_segments( num_segments, nc, bpp );
  number_components = nc;
  bits_per_pixel = bpp;
}

int pixel_info::get_number_of_bits_per_pixel() const
{
  return bits_per_pixel;
}

int pixel_info::get_number_of_components() const
{
  return number_components;
}

// Segments are organised as follow, eg 16bits RGB:
/*
NumSeg: 6
Offset: 64      // Most Significant / Red
Offset: 12752   // Least Significant / Red
Offset: 62208   // Most Significant / Green
Offset: 74896   // Least Significant / Green
Offset: 124352  // Most Significant / Blue
Offset: 137040  // Least Significant / Blue
Offset: 0
Offset: 0
Offset: 0
Offset: 0
Offset: 0
Offset: 0
Offset: 0
Offset: 0
Offset: 0
*/

int pixel_info::compute_num_segments() const
{
  return compute_num_segments_impl(number_components, bits_per_pixel);
}

image_info::image_info(int w, int h, pixel_info const & pi, bool pc, bool le):
  width(w),
  height(h),
  pix(pi),
  planarconfiguration(pc),
  littleendian(le)
{
  if( width < 0 || height < 0 )
    throw std::runtime_error( "invalid dimensions" );
  if( pc && pix.get_number_of_components() != 3 )
    throw std::runtime_error( "invalid planar configuration" );
}



} // end namespace rle
