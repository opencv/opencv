/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSegment.h"
#include "gdcmCodeString.h"

#include <cstring>

namespace gdcm
{

static const char * ALGOTypeStrings[] = {
  "MANUAL",
  "AUTOMATIC",

  0
};

const char * Segment::GetALGOTypeString(ALGOType type)
{
  assert( type <= ALGOType_END );
  return ALGOTypeStrings[(int)type];
}

Segment::ALGOType Segment::GetALGOType(const char * type)
{
  if(!type) return ALGOType_END;

  // Delete possible space as last character
  String<>  str( type );
  str.Trim();

  const char * strClear = str.Trim().c_str();

  for(unsigned int i = 0; ALGOTypeStrings[i] != 0; ++i)
    {
    if( strcmp(strClear, ALGOTypeStrings[i]) == 0 )
      {
      return (ALGOType)i;
      }
    }
  // Ouch ! We did not find anything, that's pretty bad, let's hope that
  // the toolkit which wrote the image is buggy and tolerate space padded binary
  // string
  CodeString codestring = strClear;
  std::string cs = codestring.GetAsString();
  for(unsigned int i = 0; ALGOTypeStrings[i] != 0; ++i)
    {
    if( strcmp(cs.c_str(), ALGOTypeStrings[i]) == 0 )
      {
      return (ALGOType)i;
      }
    }

  return ALGOType_END;
}

Segment::Segment():
  SegmentNumber(0),
  SegmentLabel(""),
  SegmentDescription(""),
  AnatomicRegion(),
  PropertyCategory(),
  PropertyType(),
  SegmentAlgorithmType(ALGOType_END),
  SegmentAlgorithmName(""),
  SurfaceCount(0),
  Surfaces()
{
}

Segment::~Segment()
{
}

unsigned short Segment::GetSegmentNumber() const
{
  return SegmentNumber;
}

void Segment::SetSegmentNumber(const unsigned short num)
{
  SegmentNumber = num;
}

const char * Segment::GetSegmentLabel() const
{
  return SegmentLabel.c_str();
}

void Segment::SetSegmentLabel(const char * label)
{
  SegmentLabel = label;
}

const char * Segment::GetSegmentDescription() const
{
  return SegmentDescription.c_str();
}

void Segment::SetSegmentDescription(const char * description)
{
  SegmentDescription = description;
}

SegmentHelper::BasicCodedEntry const & Segment::GetAnatomicRegion() const
{
  return AnatomicRegion;
}

SegmentHelper::BasicCodedEntry & Segment::GetAnatomicRegion()
{
  return AnatomicRegion;
}

void Segment::SetAnatomicRegion(SegmentHelper::BasicCodedEntry const & BSE)
{
  AnatomicRegion.CV   = BSE.CV;
  AnatomicRegion.CSD  = BSE.CSD;
  AnatomicRegion.CM   = BSE.CM;
}

SegmentHelper::BasicCodedEntry const & Segment::GetPropertyCategory() const
{
  return PropertyCategory;
}

SegmentHelper::BasicCodedEntry & Segment::GetPropertyCategory()
{
  return PropertyCategory;
}

void Segment::SetPropertyCategory(SegmentHelper::BasicCodedEntry const & BSE)
{
  PropertyCategory.CV   = BSE.CV;
  PropertyCategory.CSD  = BSE.CSD;
  PropertyCategory.CM   = BSE.CM;
}

SegmentHelper::BasicCodedEntry const & Segment::GetPropertyType() const
{
  return PropertyType;
}

SegmentHelper::BasicCodedEntry & Segment::GetPropertyType()
{
  return PropertyType;
}

void Segment::SetPropertyType(SegmentHelper::BasicCodedEntry const & BSE)
{
  PropertyType.CV   = BSE.CV;
  PropertyType.CSD  = BSE.CSD;
  PropertyType.CM   = BSE.CM;
}

Segment::ALGOType Segment::GetSegmentAlgorithmType() const
{
  return SegmentAlgorithmType;
}

void Segment::SetSegmentAlgorithmType(Segment::ALGOType type)
{
  assert(type <= ALGOType_END);
  SegmentAlgorithmType = type;
}

void Segment::SetSegmentAlgorithmType(const char * typeStr)
{
  SetSegmentAlgorithmType( GetALGOType(typeStr) );
}

const char * Segment::GetSegmentAlgorithmName() const
{
  return SegmentAlgorithmName.c_str();
}

void Segment::SetSegmentAlgorithmName(const char * name)
{
  SegmentAlgorithmName = name;
}

void Segment::ComputeSurfaceCount()
{
  SurfaceCount = Surfaces.size();
}

unsigned long Segment::GetSurfaceCount()
{
  if (SurfaceCount == 0)
  {
    ComputeSurfaceCount();
  }

  return SurfaceCount;
}

void Segment::SetSurfaceCount(const unsigned long nb)
{
  SurfaceCount = nb;
}

Segment::SurfaceVector const & Segment::GetSurfaces() const
{
  return Surfaces;
}

Segment::SurfaceVector & Segment::GetSurfaces()
{
  return Surfaces;
}

SmartPointer< Surface > Segment::GetSurface(const unsigned int idx /*= 0*/) const
{
  assert( idx < SurfaceCount );
  return Surfaces[idx];
}

void Segment::AddSurface(SmartPointer< Surface > surface)
{
  Surfaces.push_back(surface);
}

}
