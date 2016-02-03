/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMBITMAP_H
#define GDCMBITMAP_H

#include "gdcmObject.h"
#include "gdcmCurve.h"
#include "gdcmDataElement.h"
//#include "gdcmIconImage.h"
#include "gdcmLookupTable.h"
#include "gdcmOverlay.h"
#include "gdcmPhotometricInterpretation.h"
#include "gdcmPixelFormat.h"
#include "gdcmSmartPointer.h"
#include "gdcmTransferSyntax.h"

#include <vector>

namespace gdcm
{

/**
 * \brief Bitmap class
 * A bitmap based image. Used as parent for both IconImage and the main Pixel Data Image
 * It does not contains any World Space information (IPP, IOP)
 */
class GDCM_EXPORT Bitmap : public Object
{
public:
  Bitmap();
  ~Bitmap();
  void Print(std::ostream &) const;

  virtual bool AreOverlaysInPixelData() const { return false; }

  /// Return the number of dimension of the pixel data bytes; for example 2 for a 2D matrices of values
  unsigned int GetNumberOfDimensions() const;
  void SetNumberOfDimensions(unsigned int dim);

  /// return the planar configuration
  unsigned int GetPlanarConfiguration() const;
  /// \warning you need to call SetPixelFormat first (before SetPlanarConfiguration) for consistency checking
  void SetPlanarConfiguration(unsigned int pc);

  bool GetNeedByteSwap() const
    {
    return NeedByteSwap;
    }
  void SetNeedByteSwap(bool b)
    {
    NeedByteSwap = b;
    }


  /// Transfer syntax
  void SetTransferSyntax(TransferSyntax const &ts) {
    TS = ts;
  }
  const TransferSyntax &GetTransferSyntax() const {
    return TS;
  }
  bool IsTransferSyntaxCompatible( TransferSyntax const & ts ) const;
  void SetDataElement(DataElement const &de) {
    PixelData = de;
  }
  const DataElement& GetDataElement() const { return PixelData; }
  DataElement& GetDataElement() { return PixelData; }

  /// Set/Get LUT
  void SetLUT(LookupTable const &lut)
    {
    LUT = SmartPointer<LookupTable>( const_cast<LookupTable*>(&lut) );
    }
  const LookupTable &GetLUT() const
    {
    return *LUT;
    }
  LookupTable &GetLUT()
    {
    return *LUT;
    }

  /// Return the dimension of the pixel data, first dimension (x), then 2nd (y), then 3rd (z)...
  const unsigned int *GetDimensions() const;
  unsigned int GetDimension(unsigned int idx) const;

  void SetColumns(unsigned int col) { SetDimension(0,col); }
  unsigned int GetColumns() const { return GetDimension(0); }
  void SetRows(unsigned int rows) { SetDimension(1,rows); }
  unsigned int GetRows() const { return GetDimension(1); }
  void SetDimensions(const unsigned int dims[3]);
  void SetDimension(unsigned int idx, unsigned int dim);
  /// Get/Set PixelFormat
  const PixelFormat &GetPixelFormat() const
    {
    return PF;
    }
  PixelFormat &GetPixelFormat()
    {
    return PF;
    }
  void SetPixelFormat(PixelFormat const &pf)
    {
    PF = pf;
    PF.Validate();
    }

  /// return the photometric interpretation
  const PhotometricInterpretation &GetPhotometricInterpretation() const;
  void SetPhotometricInterpretation(PhotometricInterpretation const &pi);

  bool IsEmpty() const { return Dimensions.size() == 0; }
  void Clear();

  /// Return the length of the image after decompression
  /// WARNING for palette color: It will NOT take into account the Palette Color
  /// thus you need to multiply this length by 3 if computing the size of equivalent RGB image
  unsigned long GetBufferLength() const;

  /// Acces the raw data
  bool GetBuffer(char *buffer) const;

  /// Return whether or not the image was compressed using a lossy compressor or not
  bool IsLossy() const;

  /// Specifically set that the image was compressed using a lossy compression mechanism
  void SetLossyFlag(bool f) { LossyFlag = f; }

protected:
  bool TryRAWCodec(char *buffer, bool &lossyflag) const;
  bool TryJPEGCodec(char *buffer, bool &lossyflag) const;
  bool TryPVRGCodec(char *buffer, bool &lossyflag) const;
  bool TryKAKADUCodec(char *buffer, bool &lossyflag) const;
  bool TryJPEGLSCodec(char *buffer, bool &lossyflag) const;
#if defined(GDCM_USE_OPENJPEG)
  bool TryJPEG2000Codec(char *buffer, bool &lossyflag) const;
#endif
  bool TryRLECodec(char *buffer, bool &lossyflag) const;

  bool TryJPEGCodec2(std::ostream &os) const;
#if defined(GDCM_USE_OPENJPEG)
  bool TryJPEG2000Codec2(std::ostream &os) const;
#endif
  bool GetBuffer2(std::ostream &os) const;

  friend class PixmapReader;
  friend class ImageChangeTransferSyntax;
  // Function to compute the lossy flag based only on the image buffer.
  // Watch out that image can be lossy but in implicit little endian format...
  bool ComputeLossyFlag();

//private:
protected:
  unsigned int PlanarConfiguration;
  unsigned int NumberOfDimensions;
  TransferSyntax TS;
  PixelFormat PF; // SamplesPerPixel, BitsAllocated, BitsStored, HighBit, PixelRepresentation
  PhotometricInterpretation PI;
  // Mind dump: unsigned int is required here, since we are reading (0028,0008) Number Of Frames
  // which is VR::IS, so I cannot simply assumed that unsigned short is enough... :(
  std::vector<unsigned int> Dimensions; // Col/Row
  DataElement PixelData; // copied from 7fe0,0010

  typedef SmartPointer<LookupTable> LUTPtr;
  LUTPtr LUT;
  // I believe the following 3 ivars can be derived from TS ...
  bool NeedByteSwap;
  bool LossyFlag;

private:
  bool GetBufferInternal(char *buffer, bool &lossyflag) const;
};

} // end namespace gdcm

#endif //GDCMBITMAP_H
