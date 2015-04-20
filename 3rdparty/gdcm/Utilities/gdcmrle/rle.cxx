/*=========================================================================

  Program: librle, a minimal RLE library for DICOM

  Copyright (c) 2014 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "rle.h"
#include "info.h"
#include "io.h"

#include <vector>
#include <cstring> // memcpy
#include <cassert>
#include <stdint.h> // uint32_t

namespace rle
{

static const int max_number_offset = 15;

struct header
{
  typedef uint32_t ul;
  ul num_segments;
  ul offset[15];
};

struct rle_encoder::internal
{
  image_info img;
  header rh;

  source * src;

  // when writing, need to keep updating offsets:
  header::ul comp_pos[16];

  // internal buffer mecanism:
  std::vector<char> invalues;
  std::vector<char> outvalues;
};

rle_encoder::rle_encoder(source & s, image_info const & ii):internals(NULL)
{
  internals = new internal;
  internals->img = ii;

  internals->src = s.clone();
  memset( (char*)&internals->rh, 0, sizeof(header) );
}

rle_encoder::~rle_encoder()
{
  delete internals->src;
  delete internals;
}


// check_header behavior is twofold:
//  - on first pass it will detect if user input was found to be invalid and
// update pixel_info accordingly.
// - on the second pass when the user update with the proper pixel_info, then
// the code will fails is the remaining of the header was found to be invalid
// as per DICOM spec.
static inline bool check_header( header const & rh, pixel_info & pt )
{
  // first operation is to update pixel_info from the header:
  const int ns = pt.compute_num_segments();
  const bool ok = ns == (int)rh.num_segments;
  if( !ok )
    {
    // in case num segments is valid, update pt from the derived info:
    if( pixel_info::check_num_segments( rh.num_segments ) )
      {
      pt = pixel_info( (int)rh.num_segments );
      }
    return false;
    }

  // at least one offset is required. By convention in DICOM, it should not be
  // padded (no extra blank space), thus value is offset 64:
  if( rh.offset[0] != 64 )
    return false;

  for( unsigned int i = 1; i < rh.num_segments; ++i )
    {
    // basic error checking:
    if( rh.offset[i - 1] >= rh.offset[i] )
      return false;
    }

  // DICOM mandates all unused segments to have there offset be 0:
  for( int i = rh.num_segments; i < max_number_offset; ++i )
    if( rh.offset[i] != 0 )
      return false;

  return true;
}

bool rle_encoder::write_header( dest & d )
{
  source * src = internals->src;
  const int w = internals->img.get_width();
  const int h = internals->img.get_height();
  pixel_info pt = internals->img.get_pixel_info();
  const int nsegs = pt.compute_num_segments();

  internals->invalues.resize( w * nsegs );
  char * buffer = &internals->invalues[0];
  size_t buflen = internals->invalues.size();

  header & rh = internals->rh;
  rh.num_segments = nsegs;

  source::streampos_t start = src->tell(); // remember start position

  header::ul comp_len[16] = {0}; // 15 is the max
  for( int y = 0; y < h; ++y )
    {
    src->read_into_segments( buffer, buflen, internals->img );
    for( int s = 0; s < nsegs; ++s )
      {
      const int ret = compute_compressed_length( buffer + s * w, w );
      assert( ret > 0 );
      comp_len[s] += ret;
      }
    }
  rh.offset[0] = 64; // required
  for( int s = 1; s < nsegs; ++s )
    {
    rh.offset[s] += rh.offset[s-1] + comp_len[s-1];
    }
  assert( check_header( rh, pt ) );
  d.write( (char*)&rh, sizeof(rh) );

  header::ul comp_pos[16] = {0};
  const header::ul *offsets = internals->rh.offset;
  for( int s = 0; s < nsegs; ++s )
    {
    comp_pos[s] = offsets[s];
    }
  memcpy( internals->comp_pos, comp_pos, sizeof( comp_pos ) );

  bool b = src->seek( start ); // go back to start position
  assert( b );

  return true;
}

/*
G.3 THE RLE ALGORITHM
The RLE algorithm described in this section is used to compress Byte Segments into RLE Segments.
There is a one-to-one correspondence between Byte Segments and RLE Segments. Each RLE segment
must be an even number of bytes or padded at its end with zero to make it even.

G.3.1 The RLE encoder
A sequence of identical bytes (Replicate Run) is encoded as a two-byte code:

  < -count + 1 > <byte value>, where

count = the number of bytes in the run, and 2 <= count <= 128

and a non-repetitive sequence of bytes (Literal Run) is encoded as:

  < count - 1 > <Literal sequence of bytes>, where

count = number of bytes in the sequence, and 1 <= count <= 128.

The value of -128 may not be used to prefix a byte value.
Note: It is common to encode a 2-byte repeat run as a Replicate Run except when preceded and followed by
a Literal Run, in which case it's best to merge the three runs into a Literal Run.
 - Three-byte repeats shall be encoded as Replicate Runs.
 - Each row of the image shall be encoded separately and not cross a row boundary.
*/
static inline int count_identical_bytes(const char *start, int len) // Replicate Run
{
  assert( len > 0 );
  const char ref = start[0];
  int count = 1; // start at one
  const int cmin = std::min(128,len);
  while( count < cmin && start[count] == ref )
    {
    ++count;
    }
  assert( 1 <= count && count <= 128 );
  return count;
}

static inline int count_nonrepetitive_bytes(const char *start, int len) // Literal Run
{
  assert( start && len > 0 );
  int count = 1;
  const int cmin = std::min(128,len); // CHAR_MAX is 127
  // This version properly encode: 0 1 1 0 as: 3 0 1 1 0 ...
  for( count = 1; count < cmin; ++count )
    {
    if( start[count] == start[count-1] )
      {
      if( count + 1 < cmin && start[count] != start[count+1] )
        {
        continue;
        }
      --count;
      break;
      }
    }
  assert( 1 <= count && count <= 128 );
  return count;
}

// After a long debate
// There is no single possible solution. I need to compress in two passes. One
// will computes the offset the second one will do the writing. Since the
// offset are precomputed this should limit the writing of data back-n-forth
int rle_encoder::compute_compressed_length( const byte * source, int sourcelen )
{
  int pout = 0;
  const char *pin = source;
  int length = sourcelen;
  while( pin != source + sourcelen )
    {
    assert( length <= sourcelen );
    assert( pin <= source + sourcelen );
    int count = count_identical_bytes(pin, length);
    if( count > 1 )
      {
      // repeat case:
      ++pout;
      ++pout;
      }
    else
      {
      // non repeat case:
      // ok need to compute non-repeat:
      count = count_nonrepetitive_bytes(pin, length);
      ++pout;
      pout += count;
      }
    // count byte where read, move pin to new position:
    pin += count;
    // compute remaining length:
    assert( count <= length );
    length -= count;
    }
  return pout;
}

int rle_encoder::encode_row( dest & d )
{
  source * src = internals->src;
  const int w = internals->img.get_width();
  const pixel_info & pt = internals->img.get_pixel_info();
  const int nc = pt.get_number_of_components();
  const int bpp = pt.get_number_of_bits_per_pixel();
  const int numsegs = internals->rh.num_segments;
  assert( numsegs == (bpp / 8) * nc );

  internals->invalues.resize( w * numsegs );
  internals->outvalues.resize( w * 2 ); // worse possible case ?

  src->read_into_segments( &internals->invalues[0], internals->invalues.size(), internals->img );

  header::ul *comp_pos = internals->comp_pos;
  int n = 0;
  for( int s = 0; s < numsegs; ++s )
    {
    const int ret = encode_row_internal(
      &internals->outvalues[0], internals->outvalues.size(),
      &internals->invalues[0] + s * w, w );
    if( ret < 0 ) return -1;
    n += ret;

    const bool b = d.seek( comp_pos[s] );
    assert(b);
    d.write( &internals->outvalues[0], ret );
    comp_pos[s] += ret;
    }

  return n;
}

int rle_encoder::encode_row_internal( byte * dest, int destlen, const byte * source, int sourcelen )
{
  char *pout = dest;
  const char *pin = source;
  int length = sourcelen;
  while( pin != source + sourcelen )
    {
    assert( length <= sourcelen );
    assert( pin <= source + sourcelen );
    int count = count_identical_bytes(pin, length);
    if( count > 1 )
      {
      // repeat case:
      //
      // Test first we are allowed to write two bytes:
      if( pout + 1 + 1 > dest + destlen ) return -1;
      *pout = (char)(-count + 1);
      assert( *pout <= -1 && *pout >= -127 );
      ++pout;
      *pout = *pin;
      ++pout;
      }
    else
      {
      // non repeat case:
      // ok need to compute non-repeat:
      count = count_nonrepetitive_bytes(pin, length);
      // first test we are allowed to write 1 + count bytes in the output buffer:
      if( pout + count + 1 > dest + destlen ) return -1;
      *pout = (char)(count - 1);
      assert( *pout != -128 && *pout+1 == count );
      assert( *pout >= 0 );
      ++pout;
      memcpy(pout, pin, count);
      pout += count;
      }
    // count byte were read, move pin to new position:
    pin += count;
    // compute remaining length:
    assert( count <= length );
    length -= count;
    }
  return pout - dest;
}

struct rle_decoder::internal
{
  image_info img;
  header rh;
  source ** sources;
  int nsources;

  // scanline buffering mecanism:
  std::vector<char> scanline;

  // row crossing handling. some RLE encoder are brain dead and do cross the
  // row boundary which makes it very difficult to handle in our case since we
  // have a strict requirement of only decoding on a per-row basis.
  // furthermore this memory storage should handle all possible segments (max: 15)
  char cross_row[16][128];
  int nstorage[16]; // number of stored bytes from previous run
};

rle_decoder::rle_decoder(source & s, image_info const & ii ):internals(NULL)
{
  internals = new internal;
  memset((char*)&internals->rh, 0, sizeof(header) );
  internals->img = ii;
  const int ns = ii.get_pixel_info().compute_num_segments();
  internals->sources = new source*[ ns ];
  internals->sources[0] = s.clone(); // only one for now (minimum for read_header)
  for(int i = 1; i < ns; ++i )
    internals->sources[i] = 0;
  internals->nsources = ns;

  for(int i = 0; i < 16; ++i )
    internals->nstorage[i] = 0;
}

rle_decoder::~rle_decoder()
{
  for(int i = 0; i < internals->nsources; ++i )
    delete internals->sources[i];
  delete[] internals->sources;
  delete internals;
}

static inline bool skip_row_internal( source & s, const int width )
{
  int numOutBytes = 0;
  rle_decoder::byte b;
  bool re = false; // read error
  while( numOutBytes < width && !re && !s.eof() )
    {
    const int check = s.read( &b, 1 );
    if( check != 1 ) re = true;
    if( b >= 0 /*&& b <= 127*/ ) /* 2nd is always true */
      {
      char buffer[128];
      const int nbytes = s.read( buffer, b + 1 );
      if( nbytes != b + 1 ) re = true;
      numOutBytes += nbytes;
      }
    else if( b <= -1 && b >= -127 )
      {
      rle_decoder::byte nextByte;
      const int nbytes = s.read( &nextByte, 1 );
      if( nbytes != 1 ) re = true;
      numOutBytes += -b + 1;
      }
    /* else b == -128 */
    }
  return numOutBytes == width && !re && !s.eof() ? true : false;
}

bool rle_decoder::skip_row()
{
  for( int i = 0; i < internals->nsources; ++i )
    {
    source * s = internals->sources[i];
    bool b = skip_row_internal( *s, internals->img.get_width() );
    if( !b ) return false;
    }
  return true;
}

static inline void memcpy_withstride( char * output, const char * input, size_t len, int stride_idx, int nstride )
{
  assert( nstride >= 0 );
  if( nstride == 0 )
    {
    assert( stride_idx == 0 );
    memcpy( output, input, len );
    }
  else
    {
    for( size_t i = 0; i < len; ++i )
      {
      output[ nstride * i + stride_idx ] = input[i];
      }
    }
}

static int decode_internal( char * output, source & s, const int maxlen, const int stride_idx, const int nstride, char * cross_row, int & nstorage )
{
  assert( output && cross_row && maxlen > 0 && nstorage >= 0 );
  int numOutBytes = 0;
  char * cur = output;
  // initialize from previous RLE run:
  if( nstorage )
    {
    memcpy_withstride( cur, cross_row, nstorage, stride_idx, nstride );
    cur += nstride * nstorage;
    numOutBytes += nstorage;
    }
  // real RLE:
  char buffer[128];
  rle_decoder::byte b;
  while( numOutBytes < maxlen && !s.eof() )
    {
    int check = s.read( &b, 1 );
    assert( check == 1 );
    if( b >= 0 /*&& b <= 127*/ ) /* 2nd is always true */
      {
      int nbytes = s.read( buffer, b + 1 );
      if( nbytes != b + 1 )
        {
        assert( s.eof() );
        break;
        }
      assert( (cur - output) % nstride == 0 );
      const int diff = ( cur - output ) / nstride + nbytes - maxlen;
      if( diff > 0 ) // handle row crossing artefacts
        {
        nbytes -= diff;
        memcpy( cross_row, buffer + nbytes, diff ); // actual memcpy
        nstorage = diff;
        assert( numOutBytes + nbytes == maxlen );
        }
      memcpy_withstride( cur, buffer, nbytes, stride_idx, nstride );
      cur += nbytes * nstride;
      numOutBytes += nbytes;
      }
    else if( b <= -1 && b >= -127 )
      {
      rle_decoder::byte nextByte;
      const int nbytes = s.read( &nextByte, 1 );
      assert( nbytes == 1 );
      int nrep = -b + 1; // number of repetitions
      memset(buffer, nextByte, nrep);
      assert( (cur - output) % nstride == 0 );
      const int diff = ( cur - output ) / nstride + nrep - maxlen;
      if( diff > 0 )
        {
        nrep -= diff;
        memcpy( cross_row, buffer + nrep, diff ); // actual memcpy
        nstorage = diff;
        assert( numOutBytes + nrep == maxlen );
        }
      memcpy_withstride( cur, buffer, nrep, stride_idx, nstride );
      cur += nrep * nstride;
      numOutBytes += nrep;
      }
    /* else b == -128 */
    }

  assert( cur - output == nstride * numOutBytes );
  assert( numOutBytes <= maxlen ); // ALOKA_SSD-8-MONO2-RLE-SQ.rle
  return numOutBytes;
}

int rle_decoder::decode_row( dest & d )
{
  // PaddedCompositePixelCode:
  const pixel_info & pt = internals->img.get_pixel_info();
  const int nc = pt.get_number_of_components();
  const int bpp = pt.get_number_of_bits_per_pixel();
  const int nsegs = pt.compute_num_segments();
  const int npadded = bpp / 8;
  assert( internals->nsources == nsegs );

  const size_t scanlen = internals->img.get_width() * nsegs;
  internals->scanline.resize( scanlen );
  char * scanbuf = &internals->scanline[0];

  int numOutBytesFull = 0;
  for( int c = 0; c < nc; ++c )
    {
    for( int p = 0; p < npadded; ++p )
      {
      const int i = p + c * npadded;
      source * s = internals->sources[i];
      const int j = (npadded - 1 - p) + c * npadded; // little endian
      const int numOutBytes = decode_internal( scanbuf, *s,
        internals->img.get_width(), j, internals->nsources,
        internals->cross_row[i], internals->nstorage[i] );
      assert( numOutBytes <= internals->img.get_width() );
      numOutBytesFull += numOutBytes;
      }
    }
  d.write( scanbuf, scanlen );
  return numOutBytesFull;
}

rle_decoder::streamsize_t rle_decoder::decode_frame( dest & d )
{
  for( int i = 0; i < internals->nsources; i++ )
    {
    source *s = internals->sources[i];
    assert( s->tell() == internals->rh.offset[i] );
    }
  int numOutBytesFull = 0;
  const int mult = internals->img.get_pixel_info().compute_num_segments();
  for(int h = 0; h < internals->img.get_height(); ++h)
    {
    const int numOutBytes = decode_row( d );
    assert( numOutBytes <= internals->img.get_width() * mult ); (void)mult;
    numOutBytesFull += numOutBytes;
    }
  return numOutBytesFull;
}

int rle_decoder::get_row_nbytes() const
{
  const int mult = internals->img.get_pixel_info().compute_num_segments();
  return internals->img.get_width() * mult;
}

bool rle_decoder::read_header(pixel_info & pi)
{
  header & rh = internals->rh;
  const int size = sizeof(rh);
  source * s = internals->sources[0];
  assert( s );
  const int nbytes = s->read( (char*)&rh, sizeof(rh) );
  // we are positioned exactly at offset 64, to read the first segment
  if( nbytes != size ) return false;

  // header has been read, fill value from user input:
  pi = internals->img.get_pixel_info();

  // check those values against what the decoder has been fed with:
  if( !check_header( rh, pi ) )
    return false;

  // now is a good time to initialize all sources:
  assert( internals->nsources == (int)internals->rh.num_segments );
  for( int i = 1; i < internals->nsources; i++ )
    {
    internals->sources[i] = s->clone();
    internals->sources[i]->seek( internals->rh.offset[i] );
    }

  return true;
}

} // end namespace rle
