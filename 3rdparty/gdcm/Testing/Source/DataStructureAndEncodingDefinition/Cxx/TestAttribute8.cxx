/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAttribute.h"
#include "gdcmReader.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmFile.h"
#include "gdcmTesting.h"
#include "gdcmMediaStorage.h"

/*
 * Contributed by Michele Bosi on gdcm-dev ML
 */
int TestAttribute8Func(const char *filename, bool verbose = false)
{
  if( verbose )
  std::cout << "TestRead: " << filename << std::endl;

  gdcm::Reader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    std::cerr << "TestReadError: Failed to read: " << filename << std::endl;
    return 1;
    }

  //const gdcm::FileMetaInformation &h = reader.GetFile().GetHeader();
  //h is unused
  //std::cout << h << std::endl;

  const gdcm::DataSet &ds = reader.GetFile().GetDataSet();

//  gdcm::DataSet& ds = file.GetDataSet();

  gdcm::Attribute<0x28,0x1050> win_center;
  const gdcm::DataElement& de = ds.GetDataElement( win_center.GetTag() );
  if(!de.IsEmpty())
    {
    win_center.SetFromDataElement( de );
    std::cout << win_center.GetNumberOfValues() << ": ";
    for( unsigned int i = 0; i < win_center.GetNumberOfValues(); ++i)
      std::cout << win_center[i] << ",";
    }
  std::cout << std::endl;

  return 0;
}

int TestAttribute8(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestAttribute8Func(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestAttribute8Func( filename );
    ++i;
    }

  return r;
}
