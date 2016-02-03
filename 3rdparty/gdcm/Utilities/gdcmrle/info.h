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

// class to handle the pixel type information

// it only cope with what RLE can digest: namely 1 or 3 components. and bpp being 8/16 or 32
// this is the minimal amount of information coded in the RLE header
class pixel_info
{
public:
  /// general + provide a default cstor
  pixel_info(
    unsigned char nc = 1,
    unsigned char bpp = 8
  );

  /// RLE specific, a pixel type can be deduced directly from the RLE number of segment header (advanced)
  explicit pixel_info( int num_segments );

  /// Return the number of segments (computed from the number of components and the bits per pixel)
  int compute_num_segments() const;

  /// return number of components:
  int get_number_of_components() const;

  /// return number of bits per pixel
  int get_number_of_bits_per_pixel() const;

  /// return wether or not the number of segments `num_segments` is valid
  static bool check_num_segments( const int num_segments );

private:
  unsigned char number_components;
  unsigned char bits_per_pixel;
};

// information about the image. Contains the pixel_info and some extra
// information needed to encode the series of pixels
class image_info
{
public:
  /// initializer, will throw an exception if input parameters are incorrects
  image_info(int width = 0, int height = 0, pixel_info const & pi = pixel_info(),
    bool planarconfiguration = 0, bool littleendian = true );

  /// return width
  int get_width() const { return width; }

  /// return height
  int get_height() const { return height; }

  /// return pixel_info
  pixel_info get_pixel_info() const { return pix; }

  /// return the planar configuration
  bool get_planar_configuration() const { return planarconfiguration; }
private:
  int width;
  int height;
  pixel_info pix;
  bool planarconfiguration;
  bool littleendian;
};

} // end namespace rle
