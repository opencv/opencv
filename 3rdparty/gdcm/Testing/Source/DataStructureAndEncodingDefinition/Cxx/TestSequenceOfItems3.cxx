/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSequenceOfItems.h"
#include "gdcmReader.h"
#include "gdcmTesting.h"

int TestSequenceOfItems3Func(const char * filename, bool verbose = false)
{
  gdcm::Reader r;
  r.SetFileName( filename );
  if( !r.Read() )
    {
    return 1;
    }
  gdcm::DataSet const& ds = r.GetFile().GetDataSet();

  gdcm::DataSet::ConstIterator it = ds.Begin();
  for( ; it != ds.End(); ++it)
    {
    const gdcm::DataElement &de = *it;
    gdcm::SmartPointer<gdcm::SequenceOfItems> sqi = de.GetValueAsSQ();
    if( sqi )
      {
      std::stringstream ss;
      sqi->Print( ss ); // debug invalid read/write
      }
    }

  return 0;
}

int TestSequenceOfItems3(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestSequenceOfItems3Func(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestSequenceOfItems3Func( filename );
    ++i;
    }

  return r;
}
