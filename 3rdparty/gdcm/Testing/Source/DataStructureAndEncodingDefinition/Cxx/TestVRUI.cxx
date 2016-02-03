/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmVR.h"

#include <sstream>
#include <iostream>
#include <iomanip>

using gdcm::UIComp;

int TestVRUI(int, char *[])
{
  UIComp ui = "1.2.3.4";
  std::cout << ui << "/" << ui.size() << std::endl;
  if( ui.size() % 2 )
    {
    return 1;
    }
  if( ui[ ui.size() - 1] != 0 )
    {
    return 1;
    }

  return 0;
}
