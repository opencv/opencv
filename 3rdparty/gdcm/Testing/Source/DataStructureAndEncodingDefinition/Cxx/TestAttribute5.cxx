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


struct dummy {
  int  u;
  char v[5];
};

int TestAttribute(int argc, char *argv[])
{

  dummy du = { 2, "date" };

  const char *filename;
  if( argc < 2 )
    {
    filename = "/tmp/dummy.dcm";
    }
  else
    {
    filename = argv[1];
    }
  std::ofstream os(filename, std::ios::binary);

  //gdcm::Attribute<0x0008,0x0000, gdcm::VR::UL, gdcm::VM::VM1> a = { 38 };
  gdcm::Attribute<0x0008,0x0000> a = { 38 };
  a.Print( std::cout << std::endl );
  a.Write(os);
  gdcm::Attribute<0x0008,0x0001, gdcm::VR::UL, gdcm::VM::VM1> b = { 262302 };
  b.Print( std::cout << std::endl );
  b.Write(os);
  gdcm::Attribute<0x0008,0x0010, gdcm::VR::LO, gdcm::VM::VM1> c = { "ACRNEMA_LIBIDO_1.1" };
  c.Print( std::cout << std::endl );
  c.Write(os);

// 0008 0082 SQ 1 Institution Code Sequence
  gdcm::Attribute<0x0008,0x0082, gdcm::VR::SQ, gdcm::VM::VM1,
    gdcm::Attribute<0x0008,0x0080, gdcm::VR::LO>
    > sq = {
  "Institution Name"
    };
  sq.Print( std::cout << std::endl );
  sq.Write(os);

  gdcm::Attribute<0x0028,0x0000, gdcm::VR::UL, gdcm::VM::VM1> d = { 100 };
  d.Print( std::cout << std::endl );
  d.Write(os);
  gdcm::Attribute<0x0028,0x0005, gdcm::VR::US, gdcm::VM::VM1> e = { 2 };
  e.Print( std::cout << std::endl );
  e.Write(os);
  gdcm::Attribute<0x0028,0x0010, gdcm::VR::US, gdcm::VM::VM1> f = { 512 };
  f.Print( std::cout << std::endl );
  f.Write(os);
  gdcm::Attribute<0x0028,0x0011, gdcm::VR::US, gdcm::VM::VM1> g = { 512 };
  g.Print( std::cout << std::endl );
  g.Write(os);

// 0028 0015 US 1 UsedNbX ACR Special (RET)
// 0028 0016 US 1 UsedNbY ACR Special (RET)

  gdcm::Attribute<0x0028,0x0015, gdcm::VR::US, gdcm::VM::VM1> h = { 0 };
  h.Print( std::cout << std::endl );
  h.Write(os);
  gdcm::Attribute<0x0028,0x0016, gdcm::VR::US, gdcm::VM::VM1> i = { 0 };
  i.Print( std::cout << std::endl );
  i.Write(os);
  gdcm::Attribute<0x0028,0x0100, gdcm::VR::US, gdcm::VM::VM1> j = { 8 };
  j.Print( std::cout << std::endl );
  j.Write(os);
  gdcm::Attribute<0x0028,0x0101, gdcm::VR::US, gdcm::VM::VM1> k = { 8 };
  k.Print( std::cout << std::endl );
  k.Write(os);
  gdcm::Attribute<0x0028,0x0102, gdcm::VR::US, gdcm::VM::VM1> l = { 7 };
  l.Print( std::cout << std::endl );
  l.Write(os);
  gdcm::Attribute<0x0028,0x0103, gdcm::VR::US, gdcm::VM::VM1> m = { 0 };
  m.Print( std::cout << std::endl );
  m.Write(os);
// 0028 0199 US 1 Special Code (RET)
  gdcm::Attribute<0x0028,0x0199, gdcm::VR::US, gdcm::VM::VM1> n = { 112 };
  n.Print( std::cout << std::endl );
  n.Write(os);
  gdcm::Attribute<0x7fe0,0x0000, gdcm::VR::UL, gdcm::VM::VM1> o = { 262152 };
  o.Print( std::cout << std::endl );
  o.Write(os);
  gdcm::Attribute<0x7fe0,0x0010, gdcm::VR::OW, gdcm::VM::VM1> p;
  //
  char bytes[512*512] = {};
  p.SetBytes(bytes, 512*512);
  p.Print( std::cout << std::endl );
  p.Write(os);

  os.close();

  return 0;
}
