/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFilenameGenerator.h"

#include <iostream>

int TestFilenameGenerator(int , char *[])
{
  gdcm::FilenameGenerator fg;
  const char pattern[] = "/tmp/bla%01d";
  const unsigned int nfiles = 11;
  fg.SetPattern( pattern );
  fg.SetNumberOfFilenames( nfiles );
  if( !fg.Generate() )
    {
    std::cerr << "Could not generate" << std::endl;
    return 1;
    }
  for( unsigned int i = 0; i < nfiles; ++i )
    {
    std::cout << fg.GetFilename( i ) << std::endl;
    }

  return 0;
}
