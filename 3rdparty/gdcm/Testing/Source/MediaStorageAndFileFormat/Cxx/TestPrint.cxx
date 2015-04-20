/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmStringFilter.h"
#include "gdcmReader.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmTrace.h"
#include "gdcmTesting.h"

int TestSimplePrint(const char *filename, bool verbose = false)
{
  gdcm::Reader r;
  r.SetFileName( filename );
  if( !r.Read() )
    {
    return 1;
    }
  gdcm::DataSet const& ds = r.GetFile().GetDataSet();

  int ret = 0;
  gdcm::DataSet::ConstIterator it = ds.Begin();
  std::ostringstream os;
  for( ; it != ds.End(); ++it)
    {
    const gdcm::DataElement &ref = *it;
    os << ref << std::endl;
    }
  if( verbose ) std::cout << os.str();

  return ret;
}

int TestPrint(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestSimplePrint(filename, true);
    }

  // else
  // First of get rid of warning/debug message
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestSimplePrint( filename );
    ++i;
    }

  return r;
}
