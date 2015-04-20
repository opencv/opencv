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
#include "gdcmElement.h"
#include "gdcmVR.h"

#include <sstream>

int TestElement6(int , char *[])
{
  gdcm::Attribute<0x18,0x1310> at = { 0, 256, 256, 0 };
  gdcm::DataElement de = at.GetAsDataElement();

  const gdcm::Tag & t = at.GetTag();
  const gdcm::VM vm = gdcm::VM::VM4;

  // mimic string filter behavior:
  const char input[] = "0\\256\\256\\0";
  const char * value = input;
  const size_t len = strlen( input );

  std::istringstream is;
  std::ostringstream os;

  gdcm::VR vr = gdcm::VR::US;
  std::string s(value,value+len);
  is.str( s );

  gdcm::VL vl = 8;
  gdcm::Element<gdcm::VR::US,gdcm::VM::VM1_n> el;
  el.SetLength( vl );
  for(unsigned int i = 0; i < vm.GetLength(); ++i)
    {
    is >> el.GetValue(i);
    is.get();
    std::cout << el.GetValue(i) << std::endl;
    }
  el.Write(os);

  std::string s2 = os.str();

  std::cout << s2.size() << std::endl;

  return 0;
}
