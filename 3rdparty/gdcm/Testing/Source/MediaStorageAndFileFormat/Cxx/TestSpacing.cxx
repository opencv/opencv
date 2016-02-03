/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSpacing.h"

int TestSpacing(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  gdcm::Spacing s;

  // gdcmData/gdcm-MR-PHILIPS-16-NonSquarePixels.dcm
  // (0028,0030) DS [ 0.487416\0.194966]                       # 18,2 Pixel Spacing
  // 0.487416 / 0.194966 = 2.5000051290994327

  // Simple case ratio 1:1
  gdcm::Attribute<0x28,0x30> pixelspacing = {{0.5, 0.5}};
  gdcm::Attribute<0x28,0x34> par = gdcm::Spacing::ComputePixelAspectRatioFromPixelSpacing(pixelspacing);
  if( par[0] != 1 || par[1] != 1 )
    {
    std::cerr << "par[0] = " << par[0] << " par[1]=" << par[1] << std::endl;
    return 1;
    }

  // More complex
  pixelspacing[0] = 0.487416;
  pixelspacing[1] = 0.194966;
  par = gdcm::Spacing::ComputePixelAspectRatioFromPixelSpacing(pixelspacing);
  if( par[0] != 5 || par[1] != 2 )
    {
    std::cerr << "par[0] = " << par[0] << " par[1]=" << par[1] << std::endl;
    return 1;
    }

  return 0;
}
