/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImplicitDataElement.h"
#include "gdcmStringStream.h"
#include "gdcmSwapper.h"

int TestImplicitDataElement1(const uint16_t group,
                             const uint16_t element,
                             const uint32_t vl)
{
  const char *str;
  std::stringstream ss;
  str = reinterpret_cast<const char*>(&group);
  ss.write(str, sizeof(group));
  str = reinterpret_cast<const char*>(&element);
  ss.write(str, sizeof(element));
  str = reinterpret_cast<const char*>(&vl);
  ss.write(str, sizeof(vl));

  gdcm::ImplicitDataElement de;
  if( !de.Read<gdcm::SwapperNoOp>(ss) )
    {
    std::cerr << de << std::endl;
    return 1;
    }
  if( de.GetTag().GetGroup()   != group ||
      de.GetTag().GetElement() != element ||
      de.GetVL()               != vl )
    {
    std::cerr << de << std::endl;
    return 1;
    }
  std::cout << de << std::endl;
  return 0;
}

int TestImplicitDataElement2(const uint16_t group,
                             const uint16_t element,
                             const char *value)
{
  const char *str;
  const uint32_t vl = strlen(value);
  std::stringstream ss;
  str = reinterpret_cast<const char*>(&group);
  ss.write(str, sizeof(group));
  str = reinterpret_cast<const char*>(&element);
  ss.write(str, sizeof(element));
  str = reinterpret_cast<const char*>(&vl);
  ss.write(str, sizeof(vl));
  str = value;
  ss.write(str, vl);

  gdcm::ImplicitDataElement de;
  if( !de.Read<gdcm::SwapperNoOp>(ss) )
    {
    std::cerr << de << std::endl;
    return 1;
    }
  if( de.GetTag().GetGroup()   != group ||
      de.GetTag().GetElement() != element ||
      de.GetVL()               != vl )
    {
    std::cerr << de << std::endl;
    return 1;
    }
  const gdcm::ByteValue *bv =
    dynamic_cast<const gdcm::ByteValue*>(&de.GetValue());
  if( !bv )
    {
    return 1;
    }
  if( !(gdcm::ByteValue(value, vl) == *bv ) )
    {
    return 1;
    }
  std::cout << de << std::endl;
  return 0;
}

inline void WriteRead(gdcm::DataElement const &w, gdcm::DataElement &r)
{
  // w will be written
  // r will be read back
  std::stringstream ss;
  w.Write<gdcm::SwapperNoOp>(ss);
  r.Read<gdcm::SwapperNoOp>(ss);
}

int TestImplicitDataElement(int, char *[])
{
  const uint16_t group   = 0x0010;
  const uint16_t element = 0x0012;
  const uint32_t vl      = 0x0;
  const char value[]     = "GDCM";
  int r = 0;
  r += TestImplicitDataElement1(group, element, vl);
  r += TestImplicitDataElement2(group, element, value);

  gdcm::ImplicitDataElement de1(gdcm::Tag(0x1234, 0x5678), 0x4321);
  gdcm::ImplicitDataElement de2(gdcm::Tag(0x1234, 0x6789), 0x9876);
  WriteRead(de1, de2);
  if( !(de1 == de2) )
    {
    std::cerr << de1 << std::endl;
    std::cerr << de2 << std::endl;
    return 1;
    }

  return r;
}
