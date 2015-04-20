/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmOverlay.h"
#include "gdcmPixmapReader.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"

int TestOverlay2(int, char *[])
{
  const char *extradataroot = gdcm::Testing::GetDataExtraRoot();
  if( !extradataroot )
    {
    return 1;
    }
  if( !gdcm::System::FileIsDirectory(extradataroot) )
    {
    std::cerr << "No such directory: " << extradataroot <<  std::endl;
    return 1;
    }

  std::string filename = extradataroot;
  filename += "/gdcmSampleData/images_of_interest/XA_GE_JPEG_02_with_Overlays.dcm";
  if( !gdcm::System::FileExists(filename.c_str()) )
    {
    return 1;
    }

  gdcm::PixmapReader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "could not read: " << filename << std::endl;
    return 1;
    }
  gdcm::Pixmap &pixmap = reader.GetPixmap();

  if( pixmap.GetNumberOfOverlays() != 8 )
    {
    return 1;
    }
  size_t numoverlays = pixmap.GetNumberOfOverlays();
  for( size_t ovidx = 0; ovidx < numoverlays; ++ovidx )
    {
    const gdcm::Overlay& ov = pixmap.GetOverlay(ovidx);
    if( ov.GetTypeAsEnum() != gdcm::Overlay::Graphics )
      {
      std::cerr << "Wrong Type for overlay #" << ovidx << std::endl;
      return 1;
      }
    }

  return 0;
}
