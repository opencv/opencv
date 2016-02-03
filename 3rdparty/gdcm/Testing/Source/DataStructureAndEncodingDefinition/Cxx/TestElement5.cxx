/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmElement.h"
#include "gdcmVR.h"

#include <sstream>

int TestLO()
{
  gdcm::Element<gdcm::VR::LO, gdcm::VM::VM1_n> el;
  const char str[] = "WINDOW1\\WINDOW2\\WINDOW3";
  const size_t lenstr = strlen(str);
  std::stringstream ss;
  ss.str( str );
  unsigned int count = gdcm::VM::GetNumberOfElementsFromArray(str, lenstr);
  gdcm::VR vr = gdcm::VR::LO;
  //gdcm::VM vm = gdcm::VM::VM2;
  //gdcm::VR vr = gdcm::VR::DS;
  //if( len != vr.GetSizeof() * vm.GetLength() )
  //  {
  //  return 1;
  //  }
  el.SetLength( count * vr.GetSizeof() );
  el.Read( ss );
  std::cout << el.GetLength() << std::endl;
  std::cout << el.GetValue(0) << std::endl;
  std::cout << el.GetValue(1) << std::endl;
  return 1;
}

int TestElement5(int , char *[])
{
  gdcm::Element<gdcm::VR::DS, gdcm::VM::VM1_n> spacing;
  const char strspacing[] = "1.2345\\6.7890";
  std::stringstream ss;
  ss.str( strspacing );
  unsigned int len = 2 * sizeof(double);
  gdcm::VM vm = gdcm::VM::VM2;
  gdcm::VR vr = gdcm::VR::DS;
  if( len != vr.GetSizeof() * vm.GetLength() )
    {
    return 1;
    }
  spacing.SetLength( len );
  spacing.Read( ss );
  std::cout << spacing.GetValue() << std::endl;
  std::cout << spacing.GetValue(1) << std::endl;

  //int res =
  TestLO();

  return 0;
}
