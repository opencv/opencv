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

/*
  (0008,0000) UL 38
  (0008,0001) UL 262302
  (0008,0010) LO [ACRNEMA_LIBIDO_1.1]
  (0028,0000) UL 100
  (0028,0005) US 2
  (0028,0010) US 512
  (0028,0011) US 512
  (0028,0015) ?? 00\00
  (0028,0016) ?? 00\00
  (0028,0100) US 8
  (0028,0101) US 8
  (0028,0102) US 7
  (0028,0103) US 0
  (0028,0199) ?? 70\00
  (7fe0,0000) UL 262152
  (7fe0,0010) OW ea00\eaea\e9e9\e9e9\e9e9\e
*/


int main(int argc, char *argv[])
{
  const char *filename;
  if( argc < 2 )
    {
    filename = "/tmp/dummy.dcm";
    }
  else
    {
    filename = argv[1];
    }
  //std::cout << "Reading: " << filename << std::endl;
  std::ifstream is(filename, std::ios::binary );

  gdcm::Attribute<0x0008,0x0000, gdcm::VR::UL, gdcm::VM::VM1> a;
  a.Read(is);
  a.Print( std::cout << std::endl );
  gdcm::Attribute<0x0008,0x0001, gdcm::VR::UL, gdcm::VM::VM1> b;
  b.Read(is);
  b.Print( std::cout << std::endl );
  gdcm::Attribute<0x0008,0x0010, gdcm::VR::LO, gdcm::VM::VM24> c = {};
  c.Read(is);
  c.Print( std::cout << std::endl );
  gdcm::Attribute<0x0028,0x0000, gdcm::VR::UL, gdcm::VM::VM1> d;
  d.Read(is);
  d.Print( std::cout << std::endl );
  gdcm::Attribute<0x0028,0x0005, gdcm::VR::US, gdcm::VM::VM1> e;
  e.Read(is);
  e.Print( std::cout << std::endl );
  gdcm::Attribute<0x0028,0x0010, gdcm::VR::US, gdcm::VM::VM1> f;
  f.Read(is);
  f.Print( std::cout << std::endl );
  gdcm::Attribute<0x0028,0x0011, gdcm::VR::US, gdcm::VM::VM1> g;
  g.Read(is);
  g.Print( std::cout << std::endl );

// 0028 0015 US 1 UsedNbX ACR Special (RET)
// 0028 0016 US 1 UsedNbY ACR Special (RET)

  gdcm::Attribute<0x0028,0x0015, gdcm::VR::US, gdcm::VM::VM1> h;
  h.Read(is);
  h.Print( std::cout << std::endl );
  gdcm::Attribute<0x0028,0x0016, gdcm::VR::US, gdcm::VM::VM1> i;
  i.Read(is);
  i.Print( std::cout << std::endl );
  gdcm::Attribute<0x0028,0x0100, gdcm::VR::US, gdcm::VM::VM1> j;
  j.Read(is);
  j.Print( std::cout << std::endl );
  gdcm::Attribute<0x0028,0x0101, gdcm::VR::US, gdcm::VM::VM1> k;
  k.Read(is);
  k.Print( std::cout << std::endl );
  gdcm::Attribute<0x0028,0x0102, gdcm::VR::US, gdcm::VM::VM1> l;
  l.Read(is);
  l.Print( std::cout << std::endl );
  gdcm::Attribute<0x0028,0x0103, gdcm::VR::US, gdcm::VM::VM1> m;
  m.Read(is);
  m.Print( std::cout << std::endl );
// 0028 0199 US 1 Special Code (RET)
  gdcm::Attribute<0x0028,0x0199, gdcm::VR::US, gdcm::VM::VM1> n;
  n.Read(is);
  n.Print( std::cout << std::endl );
  gdcm::Attribute<0x7fe0,0x0000, gdcm::VR::UL, gdcm::VM::VM1> o;
  o.Read(is);
  o.Print( std::cout << std::endl );
  gdcm::Attribute<0x7fe0,0x0010, gdcm::VR::OW, gdcm::VM::VM1> p;
  //
  char bytes[512*512];
  p.SetBytes(bytes, 512*512);
  p.Read(is);
  p.Print( std::cout << std::endl );

  std::streampos pos = is.tellg();
  std::cout << "Pos=" << (int)pos << std::endl;
  is.close();

  return 0;
}
