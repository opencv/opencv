/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCodec.h"

namespace gdcm
{
class DummyCodec : public Codec
{
public:
  bool CanDecode(TransferSyntax const &) const { return false; }
  bool CanCode(TransferSyntax const &) const { return false; }
};
}

int TestCodec(int , char *[])
{
  gdcm::DummyCodec c;
  (void)c;

  return 0;
}
