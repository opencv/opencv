/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDICOMDIR.h"
#include "gdcmFileSet.h"

int TestDICOMDIR(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  gdcm::DICOMDIR dd;

  gdcm::FileSet fs;
  gdcm::File f1;
  gdcm::File f2;
  fs.AddFile( f1 );
  fs.AddFile( f2 );

  gdcm::DICOMDIR dd2(fs);

  return 0;
}
