/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPNMCODEC_H
#define GDCMPNMCODEC_H

#include "gdcmImageCodec.h"

namespace gdcm
{

/**
 * \brief Class to do PNM
 * PNM is the Portable anymap file format. The main web page can be found at:
 *   http://netpbm.sourceforge.net/
 * \note
 * Only support P5 & P6 PNM file (binary grayscale and binary rgb)
 */
class GDCM_EXPORT PNMCodec : public ImageCodec
{
public:
  PNMCodec();
  ~PNMCodec();
  bool CanDecode(TransferSyntax const &ts) const;
  bool CanCode(TransferSyntax const &ts) const;

  unsigned long GetBufferLength() const { return BufferLength; }
  void SetBufferLength(unsigned long l) { BufferLength = l; }

  bool GetHeaderInfo(std::istream &is, TransferSyntax &ts);
  virtual ImageCodec * Clone() const;

  bool Read(const char *filename, DataElement &out) const;
  bool Write(const char *filename, const DataElement &out) const;
  //bool Write(const char *filename);
private:
  unsigned long BufferLength;
};

} // end namespace gdcm

#endif //GDCMPNMCODEC_H
