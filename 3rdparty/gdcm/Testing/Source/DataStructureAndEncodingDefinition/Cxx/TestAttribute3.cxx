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

int main(int argc, char *argv[])
{
  gdcm::Attribute<0x0008,0x0000> a = { 38 };
  a.Print( std::cout << std::endl );

  gdcm::Attribute<0x0018,0x106c> b = { 123, 456 };
  b.Print( std::cout << std::endl );

  gdcm::Attribute<0x0018,0x1624> c = { 123, 456, 789 };
  c.Print( std::cout << std::endl );

  gdcm::Attribute<0x0072,0x0108> d = { 1.2, 3.4, 5.6, 7.8 };
  d.Print( std::cout << std::endl );

  gdcm::Attribute<0x3002,0x0010> e = { 1.2, 3.4, 5.6, 7.8, 9.0, 10. };
  e.Print( std::cout << std::endl );

  gdcm::Attribute<0x0018,0x1149, gdcm::VR::IS, gdcm::VM::VM2> f = { 12 , 13 };
  f.Print( std::cout << std::endl );

  gdcm::Attribute<0x0018,0x1149, gdcm::VR::IS, gdcm::VM::VM1> g = { 12 };
  g.Print( std::cout << std::endl );

  // grrrr.... too dangerous for users
  gdcm::Attribute<0x0018,0x1149, gdcm::VR::IS, gdcm::VM::VM1 > h = { 12 };
  h.Print( std::cout << std::endl );

  typedef gdcm::Attribute<0x3002,0x0010>::ArrayType type;
  const type &val = e.GetValue(2);
  std::cout << std::endl << "val=" << val;
  e.SetValue( 123.456, 2 );
  std::cout << std::endl << "val=" << val;

  // gdcm::Attribute<0x3002,0x0010>::VMType == 6, let's check that:
  const type my[ gdcm::Attribute<0x3002,0x0010>::VMType ] = { 1.2 };
  e.SetValues( my );
  e.Print( std::cout << std::endl );


  //  TODO:
//  gdcm::Attribute<0x0002,0x0001> i = { '0', '1' };
//  i.Print( std::cout << std::endl );

  gdcm::Attribute<0x0002, 0x0002> m1 = { "1.2.840.10008.5.1.4.1.1.2" };
  m1.Print( std::cout << std::endl );
  gdcm::Attribute<0x0008, 0x0016> m2 = { "1.2.840.10008.5.1.4.1.1.3" };
  m2.Print( std::cout << std::endl );
  m1.SetValues( m2.GetValues() ); // copy all the 64+1 char
  m1.Print( std::cout << std::endl );

  std::cout << std::endl;

  return 0;
}
