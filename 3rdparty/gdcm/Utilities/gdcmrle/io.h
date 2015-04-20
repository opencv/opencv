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

namespace rle
{
// forward decl
class image_info;

// base class for a source object
// all functions are pure virtual and will need to be implemented in subclasses
// basically designed for a FILE* implementation, others should be trivials
class source
{
public:
  typedef unsigned int streampos_t; // 32bits unsigned

  int read_into_segments( char * out, int len, image_info const & ii);
  virtual int read( char * out, int len ) = 0;
  virtual streampos_t tell() = 0;
  virtual bool seek(streampos_t pos) = 0;
  virtual bool eof() = 0;
  virtual source * clone() = 0;
  virtual ~source() {}
};

// base class for a dest object.
// basically designed for a FILE* implementation, others should be trivials
class dest
{
public:
  typedef unsigned int streampos_t; // 32bits unsigned

  virtual int write( const char * in, int len ) = 0;
  virtual bool seek( streampos_t abs_pos ) = 0; // seek to absolute position
};

} // end namespace rle
