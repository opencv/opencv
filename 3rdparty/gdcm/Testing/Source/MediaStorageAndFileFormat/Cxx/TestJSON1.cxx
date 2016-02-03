/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmJSON.h"
#include "gdcmFilename.h"
#include "gdcmTesting.h"
#include "gdcmDataSet.h"
#include "gdcmWriter.h"
#include "gdcmFile.h"

int TestJSON1(int, char *[])
{
  //std::string sup166 = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/QIDO-RS_examplesup166.json" );
  std::string sup166 = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/QIDO-RS_examplesup166_2.json" );
  if( !gdcm::System::FileExists( sup166.c_str() ) )
    {
    std::cerr << sup166 << std::endl;
    return 1;
    }

  std::ifstream is( sup166.c_str() );
  gdcm::JSON json;
  json.PrettyPrintOn();

#if 1
  gdcm::DataSet ds;
  json.Decode(is, ds );
  //std::cout << ds << std::endl;
#else
  gdcm::Writer w;
  gdcm::File & ff = w.GetFile();
  gdcm::DataSet &ds = ff.GetDataSet();
  ff.GetHeader().SetDataSetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );
  json.Decode(is, ff.GetDataSet() );
  w.SetFileName( "/tmp/debug2.dcm" );
  if( !w.Write() ) return 1;
#endif

  std::stringstream ss;
  json.Code(ds, ss );

  std::cout << ss.str() << std::endl;

  return 0;
}
