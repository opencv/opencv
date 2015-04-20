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
#include "gdcmDumper.h"
#include "gdcmTesting.h"
#include "gdcmCSAHeader.h"

int TestDump(const char *filename)
{
  gdcm::Reader r;
  r.SetFileName( filename );
  if( !r.Read() )
    {
    return 1;
    }

  std::ostringstream out;
{
  gdcm::Dumper p;
  p.SetFile( r.GetFile() );
  //p.Print( std::cout );
  p.Print( out );
}

  // Test CSA Header here too
  gdcm::CSAHeader csa;
  const gdcm::DataSet& ds = r.GetFile().GetDataSet();
  const gdcm::PrivateTag &t1 = csa.GetCSAImageHeaderInfoTag();
  const gdcm::PrivateTag &t2 = csa.GetCSASeriesHeaderInfoTag();

  if( ds.FindDataElement( t1 ) )
    {
    csa.LoadFromDataElement( ds.GetDataElement( t1 ) );
    csa.Print( std::cout );
    }
  if( ds.FindDataElement( t2 ) )
    {
    csa.LoadFromDataElement( ds.GetDataElement( t2 ) );
    csa.Print( std::cout );
    }
  if( csa.GetFormat() == gdcm::CSAHeader::DATASET_FORMAT )
    {
    gdcm::Dumper p;
    gdcm::File f;
    f.SetDataSet( csa.GetDataSet() );
    p.SetFile( f );
    //p.Print( std::cout );
    p.Print( out );
    }

  return 0;
}


int TestDumper(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestDump(filename);
    }

  // else
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestDump( filename );
    ++i;
    }

  return r;
}
