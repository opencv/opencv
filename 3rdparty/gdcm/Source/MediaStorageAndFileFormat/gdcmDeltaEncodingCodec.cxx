/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDeltaEncodingCodec.h"
#include "gdcmTransferSyntax.h"
#include "gdcmByteSwap.txx"
#include "gdcmDataElement.h"
#include "gdcmSequenceOfFragments.h"

#include <sstream>

namespace gdcm
{

DeltaEncodingCodec::DeltaEncodingCodec()
{
}

DeltaEncodingCodec::~DeltaEncodingCodec()
{
}

bool DeltaEncodingCodec::CanDecode(TransferSyntax const &ts)
{
  return true; // FIXME
}

bool DeltaEncodingCodec::Decode(DataElement const &in, DataElement &out)
{
  out = in;
  return true;
}

bool DeltaEncodingCodec::Decode(std::istream &is, std::ostream &os)
{
  abort();
  return true;
}

} // end namespace gdcm
