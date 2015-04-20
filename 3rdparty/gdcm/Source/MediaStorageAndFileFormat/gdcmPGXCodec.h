/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPGXCODEC_H
#define GDCMPGXCODEC_H

#include "gdcmImageCodec.h"

namespace gdcm
{

/**
 * \brief Class to do PGX
 * See PGX as used in JPEG 2000 implementation and reference images
 */
class GDCM_EXPORT PGXCodec : public ImageCodec
{
public:
  PGXCodec();
  ~PGXCodec();
  bool CanDecode(TransferSyntax const &ts) const;
  bool CanCode(TransferSyntax const &ts) const;

  bool GetHeaderInfo(std::istream &is, TransferSyntax &ts);
  virtual ImageCodec * Clone() const;

  bool Read(const char *filename, DataElement &out) const;
  bool Write(const char *filename, const DataElement &out) const;
private:
};

} // end namespace gdcm

#endif //GDCMPGXCODEC_H
