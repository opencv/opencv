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
#include "gdcmDataSet.h"
#include "gdcmWriter.h"
#include "gdcmTesting.h"

int TestCopyDataSet(int, char *[])
{
  std::string dataroot = gdcm::Testing::GetDataRoot();
  std::string filename = dataroot + "/test.acr";
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if ( !reader.Read() )
    {
    return 1;
    }

  const gdcm::DataSet &ds = reader.GetFile().GetDataSet();

  gdcm::DataSet ds_copy = ds;

  gdcm::DataElement n( gdcm::Tag(0x0028,0x0005) );
  n.SetByteValue( "3", 1 );
  std::cout << n << std::endl;

  ds_copy.Replace( n );

  std::cout << ds_copy << std::endl;
  // roup="0018" element="1020" vr="LO" vm="1-n" na
  gdcm::DataElement n2( gdcm::Tag(0x0018,0x1020) );
  //const char versions[] = "1234567890\\1234567890\\1234567890\\1234567890\\1234567890\\1234567890";
  const char versions[] = "12345678901234567890123456789012345678901234567890123\\45678901234567890";
  n2.SetByteValue( versions, (uint32_t)strlen(versions) );
  ds_copy.Replace( n2 );

  std::string outfilename = gdcm::Testing::GetTempFilename( "TestCopyDataSet.dcm" );
  gdcm::Writer writer;
  writer.SetFile( reader.GetFile() );
  writer.GetFile().GetDataSet().Replace( n2 );
  writer.SetFileName( outfilename.c_str() );
  writer.SetCheckFileMetaInformation( false );
  writer.Write();

  return 0;
}
