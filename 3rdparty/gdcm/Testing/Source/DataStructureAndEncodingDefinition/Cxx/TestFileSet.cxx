/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileSet.h"
#include "gdcmFile.h"

int TestFileSet(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  gdcm::FileSet fs;
  std::string f1;
  std::string f2;
  fs.AddFile( f1.c_str() );
  fs.AddFile( f2.c_str() );

  return 0;
}
