/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmGroupDict.h"

namespace gdcm
{

std::string const &GroupDict::GetAbbreviation(uint16_t num) const
{
  assert(num < Abbreviations.size());
  return Abbreviations[num];
}

std::string const &GroupDict::GetName(uint16_t num) const
{
  assert(num < Names.size());
  return Names[num];
}

void GroupDict::Add(std::string const &abbreviation, std::string const &name)
{
  Abbreviations.push_back(abbreviation);
  Names.push_back(name);
}

void GroupDict::Insert(uint16_t num, std::string const &abbreviation,
  std::string const &name)
{
  Abbreviations.resize(num+1);
  Names.resize(num+1);
  Abbreviations.insert(Abbreviations.begin()+num, abbreviation);
  Names.insert(Names.begin()+num, name);
}


} // end namespace gdcm
