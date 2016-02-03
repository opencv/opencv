/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmQueryBase.h"
#include "gdcmBaseRootQuery.h"

namespace gdcm
{

std::vector<Tag> QueryBase::GetAllTags(const ERootType& inRootType) const
{
  std::vector<Tag> theReturn = GetRequiredTags(inRootType);
  std::vector<Tag> theNext = GetUniqueTags(inRootType);
  theReturn.insert(theReturn.end(), theNext.begin(), theNext.end());
  theNext = GetOptionalTags(inRootType);
  theReturn.insert(theReturn.end(), theNext.begin(), theNext.end());
  return theReturn;
}

std::vector<Tag> QueryBase::GetAllRequiredTags(const ERootType& inRootType) const
{
  std::vector<Tag> theReturn = GetRequiredTags(inRootType);
  std::vector<Tag> theNext = GetUniqueTags(inRootType);
  theReturn.insert(theReturn.end(), theNext.begin(), theNext.end());
  return theReturn;
}

} // end namespace gdcm
