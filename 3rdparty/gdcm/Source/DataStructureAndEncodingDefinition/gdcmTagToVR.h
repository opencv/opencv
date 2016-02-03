/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTAGTOVR_H
#define GDCMTAGTOVR_H

#include "gdcmVR.h"

namespace gdcm 
{
  class Tag;
  VR::VRType GetVRFromTag( Tag const & tag );
}

#endif // GDCMTAGTOVR_H
