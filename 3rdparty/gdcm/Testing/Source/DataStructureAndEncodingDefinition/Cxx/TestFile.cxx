/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFile.h"
#include "gdcmSmartPointer.h"

int TestFile(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  gdcm::File f;

  gdcm::SmartPointer<gdcm::File> pf = new gdcm::File;
  {
  gdcm::SmartPointer<gdcm::File> other = pf;
  }


  return 0;
}
