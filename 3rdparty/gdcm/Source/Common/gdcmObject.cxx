/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmObject.h"

namespace gdcm
{
  // Don't ask why, but this is EXTREMELY important on Win32
  // Apparently the compiler is doing something special the first time it compiles
  // this instanciation unit
  // If this fake file is not present I get an unresolved symbol for each function
  // of the gdcm::Object class

}
