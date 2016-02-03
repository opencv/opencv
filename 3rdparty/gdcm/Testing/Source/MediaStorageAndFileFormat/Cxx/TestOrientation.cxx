/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmOrientation.h"

#include <iostream>

int TestOrientation(int, char *[])
{
  gdcm::Orientation o;
{
  double dircos[] = { 1, 0, 0, 0, 1, 0 };
  gdcm::Orientation::OrientationType type = gdcm::Orientation::GetType(dircos);

  if( type != gdcm::Orientation::AXIAL )
    {
    std::cerr << "Should be AXIAL: " << gdcm::Orientation::GetLabel( type ) << std::endl;
    return 1;
    }
}
{
  double dircos[] = { 1, 0, 0, 0, 0, 1 };
  gdcm::Orientation::OrientationType type = gdcm::Orientation::GetType(dircos);

  if( type != gdcm::Orientation::CORONAL )
    {
    return 1;
    }
}
{
  double dircos[] = { 0, 1, 0, 0, 0, -1 };
  gdcm::Orientation::OrientationType type = gdcm::Orientation::GetType(dircos);

  if( type != gdcm::Orientation::SAGITTAL )
    {
    return 1;
    }
}


  if( o.GetObliquityThresholdCosineValue() != 0.8 )
    {
    return 1;
    }
  gdcm::Orientation::SetObliquityThresholdCosineValue( 0.75 );
  if( o.GetObliquityThresholdCosineValue() != 0.75 )
    {
    return 1;
    }

  return 0;
}
