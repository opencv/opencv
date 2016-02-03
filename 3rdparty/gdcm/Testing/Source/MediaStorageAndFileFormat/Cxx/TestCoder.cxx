/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCoder.h"

namespace gdcm
{
class DummyCoder : public Coder
{
public:
  bool CanCode(TransferSyntax const &) const { return false; }
};
}

int TestCoder(int, char *[])
{
  gdcm::DummyCoder c;
  return 0;
}
