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

#include "gdcmAttribute.h"
#include "gdcmElement.h"
#include "gdcmTesting.h"

// D_CLUNIE_CT1_J2KI.dcm
//
int TestReader2(int , char *[])
{
  //const char *filename = argv[1];
  std::string dataroot = gdcm::Testing::GetDataRoot();
  std::string filename = dataroot + "/012345.002.050.dcm";

  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if ( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  const gdcm::DataSet &ds = reader.GetFile().GetDataSet();
  //gdcm::Attribute<0x0008,0x0008> a1;
  //a1.Set( ds[ gdcm::Tag(0x0008,0x0008) ].GetValue() );
  //a1.Print( std::cout );

  gdcm::Attribute<0x0020,0x0037> a2;
  a2.SetFromDataElement( ds[ gdcm::Tag(0x0020,0x0037) ] );
  a2.Print( std::cout );
  std::cout << std::endl;

/*
  // (0043,1013) SS 107\21\4\2\20                            #  10, 5 ReconKernelParameters
  gdcm::Element<gdcm::VR::SS,gdcm::VM::VM5> el1;
  el1.Set( ds[ gdcm::Tag(0x0043,0x1013) ].GetValue() );
  el1.Print( std::cout );
  std::cout << std::endl;

  // (0043,1031) DS [-11.200000\9.700000]                    #  20, 2 RACoordOfTargetReconCentre
  gdcm::Element<gdcm::VR::DS,gdcm::VM::VM2> el2;
  el2.Set( ds[ gdcm::Tag(0x0043,0x1031) ].GetValue() );
  el2.Print( std::cout );
  std::cout << std::endl;
*/

  return 0;
}
