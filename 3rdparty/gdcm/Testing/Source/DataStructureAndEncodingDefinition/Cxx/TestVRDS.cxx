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
#include "gdcmAttribute.h"
#include "gdcmByteValue.h"

#include <sstream>
#include <iostream>
#include <iomanip>


template <typename T, unsigned int MAXBYTES>
std::string TestVRDSFunc(const char *str)
{
  std::istringstream is( str );
  T d;
  is >> d;
  std::ostringstream os;
  os << std::setprecision(MAXBYTES);
  os << d;
  std::cout << std::setprecision(MAXBYTES);
  std::cout << d << std::endl;
  std::string copy = os.str();
  return copy;
}

/*
 * Test to make sure that double precision ieee 'double' is ok for DICOM VR = 'DS'
 */
int TestVRDS(int, char *[])
{
  const unsigned int dsmaxbytes = 16;
  const char str0[dsmaxbytes+1] = "0.123456789123";
  std::string copy;

  // Let's demonstrate the float can easily fails;
  if( (copy = TestVRDSFunc<float,dsmaxbytes>(str0)) == str0 )
    {
    std::cerr << "Float works:" << copy << " vs " << str0 << std::endl;
    return 1;
    }

  // Repeat with double, it works this time
  if( (copy = TestVRDSFunc<double,dsmaxbytes>(str0)) != str0 )
    {
    std::cerr << "Double does not work:" << copy << " vs " << str0 << std::endl;
    return 1;
    }

  const double d1 = -118.242525316066;
  const double d2 = 0.00149700609543456;
  const double d3 = 0.059303515816892;

  gdcm::Attribute<0x20,0x32> at;
  at.SetValue( d1, 0);
  at.SetValue( d2, 1);
  at.SetValue( d3, 2);

  gdcm::DataElement de = at.GetAsDataElement();
  std::cout << de << std::endl;

  const gdcm::ByteValue* bv = de.GetByteValue();
{
  std::string str = bv->GetPointer();
  std::string::size_type pos1 = str.find("\\");
  std::string::size_type pos2 = str.find("\\", pos1 + 1);

  if( pos1 > dsmaxbytes )
    {
    std::string s = str.substr(0, pos1);
    std::cout << "Problem with: " << s << " " << s.size() << std::endl;
    return 1;
    }
  if( (pos2 - pos1) > dsmaxbytes )
    {
    std::string s  = str.substr(pos1 + 1, pos2 - pos1 - 1);
    std::cout << "Problem with: " << s << " " << s.size() << std::endl;
    return 1;
    }
  if( (str.size() - pos2) > dsmaxbytes )
    {
    std::string s = str.substr(pos2 + 1);
    std::cout << "Problem with: " << s << " " << s.size() << std::endl;
    return 1;
    }
}

  return 0;
}
