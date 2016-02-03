/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "gdcmMeshPrimitive.h"
#include "gdcmCodeString.h"

#include <cstring>

namespace gdcm
{

static const char * MPStrings[] = {
  "VERTEX",
  "EDGE",
  "TRIANGLE",
  "TRIANGLESTRIP",
  "TRIANGLEFAN",
  "LINE",
  "FACET",
  0
};

const char * MeshPrimitive::GetMPTypeString(const MPType type)
{
  assert( type <= MPType_END );
  return MPStrings[(int)type];
}

MeshPrimitive::MPType MeshPrimitive::GetMPType(const char * type)
{
  if(!type) return MPType_END;

  // Delete possible space as last character
  String<>  str( type );
  str.Trim();
  const char * typeClear = str.Trim().c_str();

  for(unsigned int i = 0; MPStrings[i] != 0; ++i)
  {
    if( strcmp(typeClear, MPStrings[i]) == 0 )
    {
      return (MPType)i;
    }
  }
  // Ouch ! We did not find anything, that's pretty bad, let's hope that
  // the toolkit which wrote the image is buggy and tolerate space padded binary
  // string
  CodeString  codestring  = typeClear;
  std::string cs          = codestring.GetAsString();
  for(unsigned int i = 0; MPStrings[i] != 0; ++i)
  {
    if( strcmp(cs.c_str(), MPStrings[i]) == 0 )
    {
      return (MPType)i;
    }
  }

  return MPType_END;
}

MeshPrimitive::MeshPrimitive():
  PrimitiveType(MPType_END),
  PrimitiveData(1, DataElement())
{
}

MeshPrimitive::~MeshPrimitive()
{
}

MeshPrimitive::MPType MeshPrimitive::GetPrimitiveType() const
{
  return PrimitiveType;
}

void MeshPrimitive::SetPrimitiveType(const MPType type)
{
    assert( type <= MPType_END );
    PrimitiveType = type;
}

const DataElement & MeshPrimitive::GetPrimitiveData() const
{
  return PrimitiveData.front();
}

DataElement & MeshPrimitive::GetPrimitiveData()
{
  return PrimitiveData.front();
}

void MeshPrimitive::SetPrimitiveData(DataElement const & de)
{
  PrimitiveData.insert(PrimitiveData.begin(), de);
}

const MeshPrimitive::PrimitivesData & MeshPrimitive::GetPrimitivesData() const
{
  return PrimitiveData;
}

MeshPrimitive::PrimitivesData & MeshPrimitive::GetPrimitivesData()
{
  return PrimitiveData;
}

void MeshPrimitive::SetPrimitivesData(PrimitivesData const & DEs)
{
  PrimitiveData = DEs;
}

void MeshPrimitive::SetPrimitiveData(const unsigned int idx, DataElement const & de)
{
  PrimitiveData.insert(PrimitiveData.begin() + idx, de);
}

void MeshPrimitive::AddPrimitiveData(DataElement const & de)
{
  PrimitiveData.push_back(de);
}

const DataElement & MeshPrimitive::GetPrimitiveData(const unsigned int idx) const
{
    assert( idx < this->GetNumberOfPrimitivesData() );
    return PrimitiveData[idx];
}
DataElement & MeshPrimitive::GetPrimitiveData(const unsigned int idx)
{
    assert( idx < this->GetNumberOfPrimitivesData() );
    return PrimitiveData[idx];
}

unsigned int MeshPrimitive::GetNumberOfPrimitivesData() const
{
  return (unsigned int)PrimitiveData.size();
}

}
