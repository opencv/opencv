/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <limits>
#include <iostream>
#include <sstream>
#include <fstream>

int TestFloatingPointDouble(int, char *[])
{
  // Not applicable
  const char strnan[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x7F};
  const char strinf[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x7F};
  double inf = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::cout << inf << std::endl;
  std::cout << nan << std::endl;


  return 0;
}
