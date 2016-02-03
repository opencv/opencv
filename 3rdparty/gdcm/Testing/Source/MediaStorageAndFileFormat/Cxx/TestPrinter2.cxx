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
#include "gdcmPrinter.h"
#include "gdcmFilename.h"
#include "gdcmTesting.h"
#include "gdcmGlobal.h"
#include "gdcmDicts.h"

/*
 This test exercise a very obscure section of gdcm::Printer
 when it handles a SQ when it is hiden as VR::UN + undefined
 length which means this is a Implicit VR Little Endian
 encoded SQ in an Explicit VR Little Endian DataSet
*/
int TestPrinter2(int , char *[])
{
  const char *directory = gdcm::Testing::GetDataRoot();
  std::string filename = std::string(directory) + "/undefined_length_un_vr.dcm";

  gdcm::Reader r;
  r.SetFileName( filename.c_str() );
  if( !r.Read() )
    {
    return 1;
    }

  gdcm::Printer print;
  print.SetFile( r.GetFile() );
  std::ostringstream out;
  print.Print( out );

  gdcm::Global &g = gdcm::Global::GetInstance();
  gdcm::Dicts &dicts = g.GetDicts();
  gdcm::PrivateDict &priv_dict = dicts.GetPrivateDict();

  gdcm::PrivateTag pt(0x2001,0x005f,"Philips Imaging DD 001");
  if( !priv_dict.RemoveDictEntry( pt ) )
    {
    return 1;
    }

  gdcm::Reader r2;
  r2.SetFileName( filename.c_str() );
  if( !r2.Read() )
    {
    return 1;
    }

  gdcm::Printer print2;
  print2.SetFile( r.GetFile() );
  std::ostringstream out2;
  print2.Print( out2 );

//  if( out2.str() != out.str() )
//    {
//    return 1;
//    }

  return 0;
}
