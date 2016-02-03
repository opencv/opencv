/*=========================================================================

  Program: librle, a minimal RLE library for DICOM

  Copyright (c) 2014 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "io.h"

#include "info.h"
#include <stdexcept>
#include <cassert>

namespace rle
{

// this function will read in len bytes so that out contains N pixel of
// pixel_type pt spread into chunks.
// Eg, for an RGB 8bits input, out will continas RRRRR ... GGGG .... BBBBB
int source::read_into_segments( char * out, int len, image_info const & ii )
{
  pixel_info pt = ii.get_pixel_info();
  const int nc = pt.get_number_of_components();
  const int bpp = pt.get_number_of_bits_per_pixel();
  const int numsegs = pt.compute_num_segments();
  const int npadded = bpp / 8; // aka composite pixel

  // fast path (should even be inlined on most compiler)
  if( numsegs == 1 )
    {
    const int nvalues = read(out, len);
    assert( nvalues == len );
    }
  else
    {
    assert( len % numsegs == 0 );
    // FIXME we should really try to use a buffer of 4096 bytes (avoid too many
    // virtual function calls)
    //const int buffer_size = numsegs;
    if( ii.get_planar_configuration() == 0 )
      {
      const int llen = len / numsegs;
      char *sbuf[12]; // max possible is 12
      for( int s = 0; s < numsegs; ++s )
        {
        sbuf[s] = out + s * llen;
        }
      char values[12];
      for(int l = 0; l < llen; ++l )
        {
        const int nvalues = read(values, numsegs);
        assert( nvalues == numsegs );
        for( int c = 0; c < nc; ++c )
          {
          for( int p = 0; p < npadded; ++p )
            {
            const int i = p + c * npadded;
            const int j = (npadded - 1 - p) + c * npadded; // little endian
            *sbuf[i]++ = values[j];
            }
          }
        }
      }
    else
      {
      assert( 0 ); // not implemented
      throw std::invalid_argument(" not implemented" );
      return -1;
      }
    }
  return len;
}

} // end namespace rle
