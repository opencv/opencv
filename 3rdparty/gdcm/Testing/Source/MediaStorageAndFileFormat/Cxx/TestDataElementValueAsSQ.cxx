/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTesting.h"
#include "gdcmReader.h"
#include "gdcmDataElement.h"
#include "gdcmSequenceOfItems.h"

int TestDataElementValueAsSQ(int , char *[])
{
  int ret = 0;
  const char *filenames[] = {
    "D_CLUNIE_CT1_J2KI.dcm",
    "PET-cardio-Multiframe-Papyrus.dcm"
  };
  const gdcm::Tag tags[] = {
    gdcm::Tag(0x8,0x2112),
    gdcm::Tag(0x41,0x1010)
  };
  const unsigned int nfiles = sizeof(filenames)/sizeof(*filenames);
  const char *root = gdcm::Testing::GetDataRoot();
  if( !root || !*root )
    {
    std::cerr << "root is not defiend" << std::endl;
    return 1;
    }
  std::string sroot = root;
  //sroot += "/DISCIMG/IMAGES/";
  sroot += "/";
  for(unsigned int i = 0; i < nfiles; ++i)
    {
    std::string filename = sroot + filenames[i];
    //std::cout << filename << std::endl;
    gdcm::Reader r;
    r.SetFileName( filename.c_str() );
    if( !r.Read() )
      {
      ret++;
      std::cerr << "could not read: " << filename << std::endl;
      }
    const gdcm::Tag &tag = tags[i];
    gdcm::DataSet &ds = r.GetFile().GetDataSet();
    const gdcm::DataElement &roicsq = ds.GetDataElement( tag );
    gdcm::SmartPointer<gdcm::SequenceOfItems> sqi = roicsq.GetValueAsSQ();
    if(!sqi)
      {
      ++ret;
      std::cerr << "could not get SQ " << tag << " from: " << filename << std::endl;
      }
    }

  return ret;
}
