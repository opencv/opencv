/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMLOOKUPTABLE_H
#define GDCMLOOKUPTABLE_H

#include "gdcmTypes.h"
#include "gdcmObject.h"
#include <stdlib.h>

namespace gdcm
{

class LookupTableInternal;
/**
 * \brief LookupTable class
 */
class GDCM_EXPORT LookupTable : public Object
{
public:
  typedef enum {
    RED = 0,  // Keep RED == 0
    GREEN,
    BLUE,
    GRAY,
    UNKNOWN
  } LookupTableType;

  LookupTable();
  ~LookupTable();
  void Print(std::ostream &) const {}

  /// Allocate the LUT
  void Allocate( unsigned short bitsample = 8 );
  /// Generic interface:
  //TODO: check to see if length should be unsigned short, unsigned int, or whatever
  void InitializeLUT(LookupTableType type, unsigned short length,
    unsigned short subscript, unsigned short bitsize);
  unsigned int GetLUTLength(LookupTableType type) const;
  virtual void SetLUT(LookupTableType type, const unsigned char *array,
    unsigned int length);
  void GetLUT(LookupTableType type, unsigned char *array, unsigned int &length) const;
  void GetLUTDescriptor(LookupTableType type, unsigned short &length,
    unsigned short &subscript, unsigned short &bitsize) const;

  /// RED / GREEN / BLUE specific:
  void InitializeRedLUT(unsigned short length, unsigned short subscript,
    unsigned short bitsize);
  void SetRedLUT(const unsigned char *red, unsigned int length);
  void InitializeGreenLUT(unsigned short length, unsigned short subscript,
    unsigned short bitsize);
  void SetGreenLUT(const unsigned char *green, unsigned int length);
  void InitializeBlueLUT(unsigned short length, unsigned short subscript,
    unsigned short bitsize);
  void SetBlueLUT(const unsigned char *blue, unsigned int length);

  /// Clear the LUT
  void Clear();

  /// Decode the LUT
  void Decode(std::istream &is, std::ostream &os) const;

  /// Decode the LUT
  /// outputbuffer will contains the RGB decoded PALETTE COLOR input image of size inlen
  /// the outputbuffer should be at least 3 times the size of inlen
  bool Decode(char *outputbuffer, size_t outlen, const char *inputbuffer, size_t inlen) const;

  LookupTable(LookupTable const &lut):Object(lut)
    {
    assert(0);
    }

  /// return the LUT as RGBA buffer
  bool GetBufferAsRGBA(unsigned char *rgba) const;

  /// return a raw pointer to the LUT
  const unsigned char *GetPointer() const;

  /// Write the LUT as RGBA
  bool WriteBufferAsRGBA(const unsigned char *rgba);

  /// return the bit sample
  unsigned short GetBitSample() const { return BitSample; }

  /// return whether the LUT has been initialized
  bool Initialized() const;

private:
  /// Unfinished work
  void Encode(std::istream &is, std::ostream &os);

protected:
  LookupTableInternal *Internal;
  unsigned short BitSample; // refer to the pixel type (not the bit size of LUT)
  bool IncompleteLUT:1;
};

} // end namespace gdcm

#endif //GDCMLOOKUPTABLE_H
