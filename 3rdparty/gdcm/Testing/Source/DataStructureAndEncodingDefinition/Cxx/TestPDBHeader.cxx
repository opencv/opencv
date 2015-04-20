/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPDBHeader.h"
#include "gdcmTesting.h"
#include "gdcmReader.h"

int TestPDBHeader(int , char *[])
{
  const char *dataroot = gdcm::Testing::GetDataRoot();
  // gdcmData/GE_MR_0025xx1bProtocolDataBlock.dcm
  std::string filename = dataroot;
  filename += "/GE_MR_0025xx1bProtocolDataBlock.dcm";
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  gdcm::PDBHeader pdb;
  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();

  const gdcm::PrivateTag &t1 = pdb.GetPDBInfoTag();

  bool found = false;
  int ret = 0;
  if( ds.FindDataElement( t1 ) )
    {
    pdb.LoadFromDataElement( ds.GetDataElement( t1 ) );
    pdb.Print( std::cout );
    found = true;
    }
  if( !found )
    {
    std::cerr << "no pdb tag found" << std::endl;
    ret = 1;
    }

  const gdcm::PDBElement &pe = pdb.GetPDBElementByName( "SEDESC" );
  std::cout << pe << std::endl;
  if( pe.GetValue() != std::string("AX FSE T2") )
    {
    std::cerr << "Value found: " << pe.GetValue() << std::endl;
    ret = 1;
    }

  return ret;
}
