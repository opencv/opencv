/*=========================================================================

  Program: librle, a minimal RLE library for DICOM

  Copyright (c) 2014 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#pragma once

#include "info.h"
#include "io.h"

namespace rle
{
// main encoder class
class rle_encoder
{
public:
  /// main encoder function. only a single row at a time
  /// return number of bytes written
  int encode_row( dest & d );

  /// compute and write header to dest `d`
  /// this call is actually the main design of the encoder. since we are
  /// encoding on a per line basis (well this apply for the general case where
  /// bpp != 8 actually). The underlying implementation will parse once the
  /// whole input to be able to compute offsets. Those offsets will then be used
  /// later on to actually store the data. The RLE encoder is thus a two passes
  /// process (repeats)
  bool write_header( dest & d );

  /// cstor
  rle_encoder(source & s, image_info const & ii);
  ~rle_encoder();
private:
  typedef char byte;
  int compute_compressed_length( const byte * source, int sourcelen );
  int encode_row_internal( byte * dest, int destlen, const byte * source, int sourcelen );

  struct internal;
  internal *internals;
};

// this is a limited implementation this decoder is only capable of generating
// Planar Configuration = 0 output.
class rle_decoder
{
public:
  typedef char byte;
  typedef unsigned int streamsize_t; // need an integer capable of storing 32bits (unsigned)

  /// decode all the scanlines of the image. The code simply call decode_row on
  /// all scanlines.
  /// return the size of bytes written ( = height * get_row_nbytes() )
  streamsize_t decode_frame( dest & d );

  /// Instead of decoding an entire row, simply skip it. May return an error
  /// that indicate the stream is not valid or not does not respect row
  /// boundaries crossing.
  bool skip_row();

  /// Only decompress a single row at a time. returns number of bytes written.
  /// some malformed RLE stream may cross the row-boundaries in which case it
  /// makes it hard to decode a single row at a time.
  /// an extra memory will be used to handle those cases.
  int decode_row( dest & d );

  /// Read the RLE header.
  /// return true on success / false on error. Upon success the value nc / bpp
  /// will contains the number of components and number of bits per pixel used to
  /// encode the stream
  bool read_header(pixel_info & pi);

  /// Compute the actual number of bytes a single row should hold
  /// return: width * sizeof( pixel )
  int get_row_nbytes() const;

  /// initializer. cstor may throw upon invalid input
  rle_decoder(source & s, image_info const & ii);
  ~rle_decoder();
private:
  struct internal;
  internal *internals;
};

} // end namespace rle
