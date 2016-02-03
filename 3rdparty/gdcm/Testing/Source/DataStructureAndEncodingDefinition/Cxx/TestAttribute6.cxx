/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAttribute.h"

int main()
{
  gdcm::Attribute<0x0018,0x1182, gdcm::VR::IS, gdcm::VM::VM1> fd = {0};
  fd.Print( std::cout );

/*
  bool b = gdcm::VR::OB & gdcm::VR::UL;
  std::cout << b << std::endl;

  gdcm::VR vr = fd.GetVR();
  gdcm::VR dictvr = fd.GetDictVR();
  b = vr & dictvr;
  std::cout << vr << " " << dictvr << std::endl;
  std::cout << b << std::endl;

  gdcm::VM vm = fd.GetVM();
  gdcm::VM dictvm = fd.GetDictVM();
  b = vm & dictvm;
  std::cout << vm << " " << dictvm << std::endl;
  std::cout << b << std::endl;

  // Let's if we can construct a private element:
  gdcm::Attribute<0x1233,0x5678, gdcm::VR::IS, gdcm::VM::VM1> fd2 = {0};
*/
  return 0;
}
