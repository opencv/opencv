/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImage.h"
#include "gdcmImageReader.h"

int TestImage(int, char *[])
{
  gdcm::Image img;

  gdcm::ImageReader reader;
  const gdcm::Image &img2 = reader.GetImage();
  img2.Print(std::cout);//just to avoid the warning of img2 being unused

#if 0
{
  gdcm::Image img3 = reader.GetImage();
}
  gdcm::SmartPointer<gdcm::Image> img4 = const_cast<gdcm::Image*>(&img2);

  gdcm::ImageReader r;
  //r.SetFileName( "/home/mathieu/Creatis/gdcmData/test.acr" );
  r.SetFileName( "/home/mathieu/Creatis/gdcmData/DermaColorLossLess.dcm" );
  r.Read();

  //std::vector< gdcm::SmartPointer<gdcm::Image> > images;
  std::vector< gdcm::Image > images;

{
  const gdcm::Image &ref = r.GetImage();
  images.push_back( ref );

  ref.Print(std::cout);
  gdcm::Image copy1 = ref;

  copy1.Print(std::cout);

  gdcm::SmartPointer<gdcm::Image> copy2;
  copy2 = r.GetImage();
  copy2->Print(std::cout);

  gdcm::ImageReader r2;
  r2.SetFileName( "/home/mathieu/Creatis/gdcmData/012345.002.050.dcm" );
  r2.Read();

std::cout << " ------------------ " << std::endl;

  images.push_back( r2.GetImage() );

  ref.Print(std::cout);
  copy1.Print(std::cout);
  copy2->Print(std::cout);
}

  images[0].Print( std::cout );
  images[1].Print( std::cout );
#endif

  return 0;
}
