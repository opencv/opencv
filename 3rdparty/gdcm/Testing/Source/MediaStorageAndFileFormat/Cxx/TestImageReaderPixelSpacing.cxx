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
#include "gdcmImageReader.h"
#include "gdcmImage.h"

int TestImageReaderPixelSpacing(int argc, char *argv[])
{
  int ret = 0;
  const char *filenames[] = { "CRIMAGE", "DXIMAGE", "MGIMAGE" };
  const unsigned int nfiles = sizeof(filenames)/sizeof(*filenames);
  const char *root = gdcm::Testing::GetPixelSpacingDataRoot();
  if( !root || !*root )
    {
    std::cerr << "root is not defiend" << std::endl;
    return 1;
    }
  std::string sroot = root;
  sroot += "/DISCIMG/IMAGES/";
  const double spacing_ref[] = {0.5, 0.5};
  for(unsigned int i = 0; i < nfiles; ++i)
    {
    std::string filename = sroot + filenames[i];
    //std::cout << filename << std::endl;
    gdcm::ImageReader r;
    r.SetFileName( filename.c_str() );
    if( !r.Read() )
      {
      ret++;
      std::cerr << "could not read: " << filename << std::endl;
      }
    const gdcm::Image &image = r.GetImage();
    const double *spacing = image.GetSpacing();
    std::cout << spacing[0] << ","
      << spacing[1] << ","
      << spacing[2] << std::endl;
    if( spacing[0] != spacing_ref[0]
      || spacing[1] != spacing_ref[1] )
      {
      std::cerr << "Wrong spacing for: " << filename << std::endl;
      ++ret;
      }
    }

  return ret;
}
