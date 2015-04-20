/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTesting.h"
#include "gdcmSystem.h"

int TestTesting(int , char *[])
{
  gdcm::Testing testing;
  testing.Print( std::cout );

  const char *f = gdcm::Testing::GetFileName( 100000 );
  if( f ) return 1;

  std::cout << "Num:" << gdcm::Testing::GetNumberOfFileNames() << std::endl;

  const char * const * md5 = gdcm::Testing::GetMD5DataImage( 100000 );
  if( !md5 ) return 1;
  if( md5[0] || md5[1] ) return 1;

  const char * const *null = gdcm::Testing::GetMD5DataImage(1000000000u);
  if( null[0] != NULL || null[1] != NULL )
    {
    return 1;
    }


  const char *tmp = gdcm::Testing::GetTempDirectory();
  if( !gdcm::System::FileExists(tmp) )
    {
    return 1;
    }
  if( !gdcm::System::FileIsDirectory(tmp) )
    {
    return 1;
    }

  return 0;
}
