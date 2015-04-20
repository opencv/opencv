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
#include "gdcmFileMetaInformation.h"

#include "gdcmDataImages.h"

int TestTortureRead(const char* filename)
{
  // TODO
  unsigned long copy = rand()*original_size;
  std::copy(new_file, filename);

  gdcm::Reader reader;
  reader.SetFileName( new_file );
  if ( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  std::cerr << "Success to read: " << filename << std::endl;

  const gdcm::FileMetaInformation &h = reader.GetHeader();
  std::cout << h << std::endl;

  const gdcm::DataSet &ds = reader.GetDataSet();
  std::cout << ds << std::endl;

  return 0;
}

int TestTorture(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestRead(filename);
    }

  // else
  int r = 0, i = 0;
  const char *filename;
  while( (filename = gdcmDataImages[i]) )
    {
    r += TestRead( filename );
    ++i;
    }

  return r;
}
