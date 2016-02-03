/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDELTAENCODINGCODEC_H
#define GDCMDELTAENCODINGCODEC_H

#include "gdcmImageCodec.h"
#error do not use

namespace gdcm
{

/**
 * \brief DeltaEncodingCodec compression used by some private
 * vendor
 */
class DeltaEncodingCodec : public ImageCodec
{
public:
  DeltaEncodingCodec();
  ~DeltaEncodingCodec();
  bool CanDecode(TransferSyntax const &ts);
  bool Decode(DataElement const &is, DataElement &os);
protected:
  bool Decode(std::istream &is, std::ostream &os);
};

} // end namespace gdcm

#endif //GDCMDELTAENCODINGCODEC_H
