/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPDFCODEC_H
#define GDCMPDFCODEC_H

#include "gdcmCodec.h"

namespace gdcm
{

/**
 * \brief PDFCodec class
 */
class GDCM_EXPORT PDFCodec : public Codec
{
public:
  PDFCodec();
  ~PDFCodec();
  bool CanCode(TransferSyntax const &) const { return false; }
  bool CanDecode(TransferSyntax const &) const { return false; }
  bool Decode(DataElement const &is, DataElement &os);
};

} // end namespace gdcm

#endif //GDCMPDFCODEC_H
