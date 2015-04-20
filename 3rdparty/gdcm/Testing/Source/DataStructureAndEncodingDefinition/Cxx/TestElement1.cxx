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
#include "gdcmDataSet.h"

#define TPI 3.1415926535897931

namespace gdcm
{

int TestFL()
{
  Element<VR::FL, VM::VM1> a = {{ (float)TPI }};
  a.Print( std::cout );
  std::cout << std::endl;

  Element<VR::FL, VM::VM8> b =
    {{ 0,1,2,3,4,5,6,7 }};
  b.Print( std::cout );
  std::cout << std::endl;

  float f[10] = {};
  Element<VR::FL, VM::VM1_n> c;
  c.SetArray( f, sizeof(f), false);
  c.Print( std::cout );
  std::cout << std::endl;

  // Make sure this is possible to output as DataElement
  // an Element, in case one cannot use gdcm::Attribute
  // Eg. Sup 145 are not available -yet-
{
  DataSet ds;
  Element<VR::FL,VM::VM1> el;
  el.SetValue(1.2f);

  DataElement de = el.GetAsDataElement();
  de.SetTag( Tag(0x0048,0x0201) );
  ds.Insert( de );
}


  return 0;
}

int TestFD()
{
  Element<VR::FD, VM::VM1> a = {{ TPI }};
  std::ostringstream os;
  a.Print( os );
  const std::string st = os.str(); // important
  const char *s = st.c_str();
  std::cout << s << std::endl;
  //double t = *reinterpret_cast<const double*>(*s);
  //std::cout << t << std::endl;

  Element<VR::FD, VM::VM8> b;
  double array[] = { 1,2,3,4,5,6,7,9 };
  b = reinterpret_cast<Element<VR::FD, VM::VM8>& >( array );
  b.Print( std::cout );
  std::cout << std::endl;

  return 0;
}

int TestAS()
{
  Element<VR::AS, VM::VM5> a = { "019Y" };
  a.Print( std::cout );
  std::cout << std::endl;

  // TODO this should not compile:
  Element<VR::AS, VM::VM6> b = {{ "019Yb" }};
  b = b;//to avoid the warning of b not being useful

  return 0;
}

int TestUL()
{
  const char array[4] = {-78, 1, 0, 0}; // 434
  {
  Element<VR::UL, VM::VM1> a;
  // reinterpret_cast< const Element<VR::UL, VM::VM1>& > ( array );
  memcpy((void*)&a, array, 4);
  a.Print( std::cout );
  }
  std::cout << std::endl;

  return 0;
}

int TestAT()
{
  // = (0020,5000) : (0010,0010)\(0010,0020)\(0020,0013)
  Element<VR::AT, VM::VM3> a;
  Tag list[3];
  list[0] = Tag(0x0010,0x0010);
  list[1] = Tag(0x0010,0x0020);
  list[2] = Tag(0x0020,0x0013);
  memcpy(&a, list, sizeof(list));
  a.Print( std::cout );
  std::cout << std::endl;

  Element<VR::AT, VM::VM1_n> b;
  b.SetArray( list, sizeof(list), false);
  b.Print( std::cout );
  std::cout << std::endl;

  return 0;
}

int TestOB()
{
  const unsigned char array[] =
    { 0x00,0x00,0x00,0x01,0x42,0x12,0xf9,0x22,0x00,0x31,0x00,0x00,0x00,0xc0,0x00,0x00,0x00,0x00,0x03,0xfe,0x02,0x71 };
  // Bad no such thing as 1-n for OB/OW:
  Element<VR::OB, VM::VM1_n> a;
  a.SetArray( array, sizeof(array), false);
  // reinterpret_cast< const Element<VR::UL, VM::VM1>& > ( array );
  //memcpy((void*)&a, array, sizeof(array));
  a.Print( std::cout );
  std::cout << std::endl;

  Element<VR::OB, VM::VM1> b;
  b.SetArray( array, sizeof(array), false);
  // reinterpret_cast< const Element<VR::UL, VM::VM1>& > ( array );
  //memcpy((void*)&a, array, sizeof(array));
  b.Print( std::cout );
  std::cout << std::endl;

  return 0;
}

int TestUSVM3()
{
  Element<VR::US, VM::VM3> a = {{ 0x0001, 0x0002, 0x0003 }};
  a.Print( std::cout );
  std::cout << std::endl;
  unsigned short tmp = a.GetValue(0);
  if( tmp != 0x0001 )
    {
    return 1;
    }
  tmp = a.GetValue(1);
  if( tmp != 0x0002 )
    {
    return 1;
    }
  tmp = a.GetValue(2);
  if( tmp != 0x0003 )
    {
    return 1;
    }
  std::stringstream ss;
  a.Write( ss );

  Element<VR::US, VM::VM3> b;
  b.Read( ss );
  b.Print( std::cout );
  tmp = b.GetValue(0);
  if( tmp != 0x0001 )
    {
    return 1;
    }
  tmp = b.GetValue(1);
  if( tmp != 0x0002 )
    {
    return 1;
    }
  tmp = b.GetValue(2);
  if( tmp != 0x0003 )
    {
    return 1;
    }
  std::cout << std::endl;

  return 0;
}
}

int TestElement1(int , char *[])
{
  int r = 0;
  r += gdcm::TestFL();
  r += gdcm::TestFD();
  r += gdcm::TestAS();
  r += gdcm::TestUSVM3();
  r += gdcm::TestUL();
  r += gdcm::TestOB();
  r += gdcm::TestAT();

  return r;
}
