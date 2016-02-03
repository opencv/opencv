/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMPIXELFORMAT_H
#define GDCMPIXELFORMAT_H

#include "gdcmTypes.h"
#include <iostream>
#include <assert.h>

namespace gdcm
{

class TransferSyntax;

/**
 * \brief PixelFormat
 * \note
 * By default the Pixel Type will be instanciated with the following
 * parameters:
 * - SamplesPerPixel : 1
 * - BitsAllocated : 8
 * - BitsStored : 8
 * - HighBit : 7
 * - PixelRepresentation : 0
 *
 * Fundamentally PixelFormat is very close to what DICOM allows. It will be
 * very hard to extend this class for the upcoming DICOM standard where
 * Floating 32 and 64bits will be allowed.
 *
 * It is also very hard for this class to fully support 64bits integer type
 * (see GetMin / GetMax signature restricted to 64bits signed).
 */
class GDCM_EXPORT PixelFormat
{
  friend class Bitmap;
  friend std::ostream& operator<<(std::ostream &_os, const PixelFormat &pf);
public:
  // When adding a type please add its dual type (its unsigned conterpart)
  typedef enum {
    UINT8,
    INT8,
    UINT12,
    INT12,
    UINT16,
    INT16,
    UINT32,  // For some DICOM files (RT or SC)
    INT32,   //                        "   "
    UINT64,  // Needed when input is 32bits + intercept/slope (incomplete support)
    INT64,   //                        "   "
    FLOAT16, // sure why not...
    FLOAT32, // good ol' 'float'
    FLOAT64, // aka 'double'
    SINGLEBIT, // bool / monochrome
    UNKNOWN // aka BitsAllocated == 0 && PixelRepresentation == 0
  } ScalarType;

  // default cstor:
  explicit PixelFormat (
    unsigned short samplesperpixel = 1,
    unsigned short bitsallocated = 8,
    unsigned short bitsstored = 8,
    unsigned short highbit = 7,
    unsigned short pixelrepresentation = 0 ) :
  SamplesPerPixel(samplesperpixel),
  BitsAllocated(bitsallocated),
  BitsStored(bitsstored),
  HighBit(highbit),
  PixelRepresentation(pixelrepresentation) {}
  // helper, for the common case
  PixelFormat(ScalarType st);

  // For transparency of use
  operator ScalarType() const { return GetScalarType(); }

  /// Samples Per Pixel see (0028,0002) US Samples Per Pixel
  /// DICOM - only allows 1, 3 and 4 as valid value. Other value are undefined behavior.
  unsigned short GetSamplesPerPixel() const;
  void SetSamplesPerPixel(unsigned short spp)
    {
    gdcmAssertMacro( spp <= 4 );
    SamplesPerPixel = spp;
    assert( SamplesPerPixel == 1 || SamplesPerPixel == 3 || SamplesPerPixel == 4 );
    }

  /// BitsAllocated see Tag (0028,0100) US Bits Allocated
  unsigned short GetBitsAllocated() const
    {
    return BitsAllocated;
    }
  void SetBitsAllocated(unsigned short ba)
    {
    if( ba )
      {
      BitsAllocated = ba;
      BitsStored = ba;
      HighBit = (unsigned short)(ba - 1);
      }
    else // Make the PixelFormat as UNKNOWN
      {
      BitsAllocated = 0;
      PixelRepresentation = 0;
      }
    }

  /// BitsStored see Tag (0028,0101) US Bits Stored
  unsigned short GetBitsStored() const
    {
    assert( BitsStored <= BitsAllocated );
    return BitsStored;
    }
  void SetBitsStored(unsigned short bs)
    {
    if( bs <= BitsAllocated && bs )
      {
      BitsStored = bs;
      SetHighBit( (unsigned short) (bs - 1) );
      }
    }

  /// HighBit see Tag (0028,0102) US High Bit
  unsigned short GetHighBit() const
    {
    assert( HighBit < BitsStored );
    return HighBit;
    }
  void SetHighBit(unsigned short hb)
    {
    if( hb < BitsStored )
      HighBit = hb;
    }

  /// PixelRepresentation: 0 or 1, see Tag (0028,0103) US Pixel Representation
  unsigned short GetPixelRepresentation() const
    {
    return (unsigned short)(PixelRepresentation ? 1 : 0);
    }
  void SetPixelRepresentation(unsigned short pr)
    {
    PixelRepresentation = (unsigned short)(pr ? 1 : 0);
    }

  /// ScalarType does not take into account the sample per pixel
  ScalarType GetScalarType() const;

  /// Set PixelFormat based only on the ScalarType
  /// \warning: You need to call SetScalarType *before* SetSamplesPerPixel
  void SetScalarType(ScalarType st);
  const char *GetScalarTypeAsString() const;

  /// return the size of the pixel
  /// This is the number of words it would take to store one pixel
  /// \warning the return value takes into account the SamplesPerPixel
  /// \warning in the rare case when BitsAllocated == 12, the function
  /// assume word padding and value returned will be identical as if BitsAllocated == 16
  uint8_t GetPixelSize() const;

  /// Print
  void Print(std::ostream &os) const;

  /// return the min possible of the pixel
  int64_t GetMin() const;

  /// return the max possible of the pixel
  int64_t GetMax() const;

  /// return IsValid
  bool IsValid() const;

  bool operator==(ScalarType st) const
    {
    return GetScalarType() == st;
    }
  bool operator!=(ScalarType st) const
    {
    return GetScalarType() != st;
    }
  bool operator==(const PixelFormat &pf) const
    {
    return
      SamplesPerPixel     == pf.SamplesPerPixel &&
      BitsAllocated       == pf.BitsAllocated &&
      BitsStored          == pf.BitsStored &&
      HighBit             == pf.HighBit &&
      PixelRepresentation == pf.PixelRepresentation;
    }
  bool operator!=(const PixelFormat &pf) const
    {
    return
      SamplesPerPixel     != pf.SamplesPerPixel ||
      BitsAllocated       != pf.BitsAllocated ||
      BitsStored          != pf.BitsStored ||
      HighBit             != pf.HighBit ||
      PixelRepresentation != pf.PixelRepresentation;
    }

  bool IsCompatible(const TransferSyntax & ts ) const;
protected:
  /// When image with 24/24/23 was read, need to validate
  bool Validate();

private:
  // D 0028|0002 [US] [Samples per Pixel] [1]
  unsigned short SamplesPerPixel;
  // D 0028|0100 [US] [Bits Allocated] [8]
  unsigned short BitsAllocated;
  // D 0028|0101 [US] [Bits Stored] [8]
  unsigned short BitsStored;
  // D 0028|0102 [US] [High Bit] [7]
  unsigned short HighBit;
  // D 0028|0103 [US] [Pixel Representation] [0]
  unsigned short PixelRepresentation;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const PixelFormat &pf)
{
  pf.Print( os );
  return os;
}

} // end namespace gdcm

#endif //GDCMPIXELFORMAT_H
