/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMICONIMAGE_H
#define GDCMICONIMAGE_H

#if 0
#include "gdcmObject.h"
#include "gdcmDataElement.h"
#include "gdcmPhotometricInterpretation.h"
#include "gdcmPixelFormat.h"
#include "gdcmTransferSyntax.h"

#include <vector>

namespace gdcm
{

/**
 * \brief IconImage class
 */
class GDCM_EXPORT IconImage : public Object
{
public:
  IconImage();
  ~IconImage();
  void Print(std::ostream &) const {}

  /// Transfer syntax
  void SetTransferSyntax(TransferSyntax const &ts) {
    TS = ts;
  }
  const TransferSyntax &GetTransferSyntax() const {
    return TS;
  }
  void SetDataElement(DataElement const &de) {
    PixelData = de;
  }
  const DataElement& GetDataElement() const { return PixelData; }

  void SetColumns(unsigned int col) { SetDimension(0,col); }
  void SetRows(unsigned int rows) { SetDimension(1,rows); }
  void SetDimension(unsigned int idx, unsigned int dim);
  int GetColumns() const { return Dimensions[0]; }
  int GetRows() const { return Dimensions[1]; }
  // Get/Set PixelFormat
  const PixelFormat &GetPixelFormat() const
    {
    return PF;
    }
  void SetPixelFormat(PixelFormat const &pf)
    {
    PF = pf;
    }

  const PhotometricInterpretation &GetPhotometricInterpretation() const;
  void SetPhotometricInterpretation(PhotometricInterpretation const &pi);

  bool IsEmpty() const { return Dimensions.size() == 0; }
  void Clear();

  bool GetBuffer(char *buffer) const;

private:
  TransferSyntax TS;
  PixelFormat PF; // SamplesPerPixel, BitsAllocated, BitsStored, HighBit, PixelRepresentation
  PhotometricInterpretation PI;
  std::vector<unsigned int> Dimensions; // Col/Row
  std::vector<double> Spacing; // PixelAspectRatio ?
  DataElement PixelData; // copied from 7fe0,0010
  static const unsigned int NumberOfDimensions = 2;
};

} // end namespace gdcm
#endif
#include "gdcmBitmap.h"

namespace gdcm
{
  //class GDCM_EXPORT IconImage : public Pixmap {};
  typedef Bitmap IconImage;
}

#endif //GDCMICONIMAGE_H
