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

using gdcm::LTComp;

int TestVRLT(int, char *[])
{
  LTComp lt = "hello";
  std::cout << lt << std::endl;
  if( lt.size() % 2 )
    {
    return 1;
    }
  if( lt[ lt.size() - 1] != ' ' )
    {
    return 1;
    }

  return 0;
}
