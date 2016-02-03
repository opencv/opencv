/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSegmentHelper.h"

namespace gdcm
{

namespace SegmentHelper
{

bool BasicCodedEntry::IsEmpty(const bool checkOptionalAttributes/* = false*/) const
{
  bool res = true;

  if (!CV.empty() && !CSD.empty() && !CM.empty())
  {
    if (checkOptionalAttributes)
    {
      if (!CSV.empty())
      {
        res = false;
      }
    }
    else
    {
      res = false;
    }
  }

  return res;
}

} // end of SegmentHelper namespace

} // end of gdcm namespace
