/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmReader.h"

#include "gdcmTesting.h"

// Reproduce issue #3538586
int TestReader4(int , char *[])
{
  const char subdir[] = "TestReader4";
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    }
  std::string outfilename = tmpdir;
  outfilename += "/";
  outfilename += "fake.jpg";

  const unsigned char jpeg[] = { 0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46 };
  std::ofstream out( outfilename.c_str() );
  out.write( (char*)jpeg, sizeof( jpeg ) );
  out.close();

  const char *filename = outfilename.c_str();
  gdcm::Reader reader;
  reader.SetFileName( filename );
  if ( reader.Read() )
    {
    std::cerr << "Success to read zip file !: " << filename << std::endl;
    return 1;
    }
  if ( reader.Read() )
    {
    std::cerr << "Success to read zip file !: " << filename << std::endl;
    return 1;
    }

  return 0;
}
