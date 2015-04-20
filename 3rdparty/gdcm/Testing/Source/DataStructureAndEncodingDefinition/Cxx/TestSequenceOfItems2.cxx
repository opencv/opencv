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
#include "gdcmPrivateTag.h"

int TestSequenceOfItems2(int, char *[])
{
  const char *dataroot = gdcm::Testing::GetDataRoot();
  // gdcmData/AMIInvalidPrivateDefinedLengthSQasUN.dcm
  std::string filename = dataroot;
  filename += "/AMIInvalidPrivateDefinedLengthSQasUN.dcm";
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  gdcm::PrivateTag pt(0x0009,0x10,"GEIIS");
  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();

  if( !ds.FindDataElement( pt ) )
    {
    return 1;
    }

  const gdcm::DataElement &de = ds.GetDataElement( pt );
  if( de.IsEmpty() ) return 1;

  std::cout << de << std::endl;

  gdcm::SmartPointer<gdcm::SequenceOfItems> sqi = de.GetValueAsSQ();
  if( !sqi ) return 1;

  //std::cout << *sqi << std::endl;
  //sqi->Print( std::cout );
  gdcm::SequenceOfItems::SizeType n = sqi->GetNumberOfItems();
  if( n != 1 ) return 1;

  const gdcm::Item & item = sqi->GetItem( 1 );
  const gdcm::DataSet &subds = item.GetNestedDataSet();

  // std::cout << subds << std::endl;
  gdcm::Tag ticonpixeldata(0x7fe0,0x0010);
  //const gdcm::DataElement &iconpixeldata = subds.GetDataElement( ticonpixeldata );
//  const gdcm::ByteValue *bv = iconpixeldata.GetByteValue();//unused unless that lower #def is set to true

  // I could test that gdcm::JPEGCodec::GetHeaderInfo return JPEG file:
  // JPEG image data, JFIF standard 1.01

  gdcm::PrivateTag tgeiiscompressiontype(0x7fd1,0x10,"GEIIS");
  const gdcm::DataElement &geiiscompressiontype = subds.GetDataElement( tgeiiscompressiontype );
  gdcm::Element<gdcm::VR::UL,gdcm::VM::VM1> el;
  el.SetFromDataElement( geiiscompressiontype );

  unsigned int v = el.GetValue();
  if( v != 26 ) return 1;

#if 0
  std::ofstream of( "/tmp/o.jpg", std::ios::binary );
  of.write( bv->GetPointer(), bv->GetLength() );
  of.close();
#endif

  return 0;
}
