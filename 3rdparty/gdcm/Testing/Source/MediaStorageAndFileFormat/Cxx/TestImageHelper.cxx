/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageHelper.h"
#include "gdcmMediaStorage.h"
#include "gdcmDataSet.h"
#include "gdcmAttribute.h"
#include "gdcmDirectionCosines.h"

int TestImageHelper(int, char *[])
{
//  gdcm::ImageHelper sh;
  const double pos[] = { 0,0,0,
                         1,1,1};
  //const double answer[3] = {1,1,1};

  std::vector<double> impos(pos,pos+6);
  std::vector<double> spacing;
  spacing.resize(3);
  if( !gdcm::ImageHelper::ComputeSpacingFromImagePositionPatient(impos, spacing) )
    {
    return 1;
    }
  std::cout << spacing[0] << std::endl;
  std::cout << spacing[1] << std::endl;
  std::cout << spacing[2] << std::endl;


  std::vector<double> dircos;
  // make it an invalid call
  dircos.resize(6);
  dircos[0] = 1;
  dircos[1] = 0;
  dircos[2] = 0;
  dircos[3] = 1;
  dircos[4] = 0;
  dircos[5] = 0;
  // Try SC
{
  gdcm::MediaStorage ms( gdcm::MediaStorage::SecondaryCaptureImageStorage );
  gdcm::DataSet ds;

    gdcm::DataElement de( gdcm::Tag(0x0008, 0x0016) );
    const char* msstr = gdcm::MediaStorage::GetMSString(ms);
    de.SetByteValue( msstr, (uint32_t)strlen(msstr) );
    de.SetVR( gdcm::Attribute<0x0008, 0x0016>::GetVR() );
    ds.Insert( de );

  // Since SC this is a no-op
  gdcm::ImageHelper::SetDirectionCosinesValue( ds, dircos );

  // previous call removed the attribute
  gdcm::Tag iop(0x0020,0x0037);
  if(  ds.FindDataElement( iop ) ) return 1;
}

  // MR now
{
  gdcm::MediaStorage ms( gdcm::MediaStorage::MRImageStorage );
  gdcm::DataSet ds;

    gdcm::DataElement de( gdcm::Tag(0x0008, 0x0016) );
    const char* msstr = gdcm::MediaStorage::GetMSString(ms);
    de.SetByteValue( msstr, (uint32_t)strlen(msstr) );
    de.SetVR( gdcm::Attribute<0x0008, 0x0016>::GetVR() );
    ds.Insert( de );

  // Since SC this is a no-op
  gdcm::ImageHelper::SetDirectionCosinesValue( ds, dircos );

  // previous call should not have replaced tag with default value
  gdcm::Tag iop(0x0020,0x0037);
  if(  !ds.FindDataElement( iop ) ) return 1;

  const gdcm::DataElement &iopde = ds.GetDataElement( iop );
  gdcm::Attribute<0x0020,0x0037> at;
  at.SetFromDataElement( iopde );

  dircos.clear();
  dircos.push_back( at.GetValue(0) );
  dircos.push_back( at.GetValue(1) );
  dircos.push_back( at.GetValue(2) );
  dircos.push_back( at.GetValue(3) );
  dircos.push_back( at.GetValue(4) );
  dircos.push_back( at.GetValue(5) );
  gdcm::DirectionCosines dc( &dircos[0] );
  if (!dc.IsValid()) return 1;


  if ( (at.GetValue(0) != 1) || (at.GetValue(1) != 0) || (at.GetValue(2) != 0) ||
       (at.GetValue(3) != 0) || (at.GetValue(4) != 1) || (at.GetValue(5) != 0) ) return 1;

}

  // SC family
  //ms = gdcm::MediaStorage::SecondaryCaptureImageStorage;
  gdcm::PixelFormat pf;
  gdcm::PhotometricInterpretation pi = gdcm::PhotometricInterpretation::MONOCHROME2;
  gdcm::MediaStorage ms;
  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality( "OT", 2 );
  if( ms != gdcm::MediaStorage::SecondaryCaptureImageStorage )
    {
    std::cerr << "SecondaryCaptureImageStorage 1" << std::endl;
    return 1;
    }
  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality( "OT", 2, pf );
  if( ms != gdcm::MediaStorage::SecondaryCaptureImageStorage )
    {
    std::cerr << "SecondaryCaptureImageStorage 2" << std::endl;
    return 1;
    }
  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality( "OT", 2, pf, pi );
  if( ms != gdcm::MediaStorage::SecondaryCaptureImageStorage )
    {
    std::cerr << "SecondaryCaptureImageStorage 3" << std::endl;
    return 1;
    }
  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality( "OT", 3, pf, pi );
  if( ms != gdcm::MediaStorage::MultiframeGrayscaleByteSecondaryCaptureImageStorage )
    {
    std::cerr << "MultiframeGrayscaleByteSecondaryCaptureImageStorage" << std::endl;
    return 1;
    }
  pf.SetBitsAllocated( 1 );
  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality( "OT", 3, pf, pi );
  if( ms != gdcm::MediaStorage::MultiframeSingleBitSecondaryCaptureImageStorage )
    {
    std::cerr << "MultiframeSingleBitSecondaryCaptureImageStorage" << std::endl;
    return 1;
    }
  pf.SetBitsAllocated( 16 );
  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality( "OT", 3, pf, pi );
  if( ms != gdcm::MediaStorage::MultiframeGrayscaleWordSecondaryCaptureImageStorage )
    {
    std::cerr << "MultiframeGrayscaleWordSecondaryCaptureImageStorage" << std::endl;
    return 1;
    }
  pf.SetBitsAllocated( 8 );
  pf.SetSamplesPerPixel( 3 );
  pi = gdcm::PhotometricInterpretation::RGB;
  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality( "OT", 3, pf, pi );
  if( ms != gdcm::MediaStorage::MultiframeTrueColorSecondaryCaptureImageStorage )
    {
    std::cerr << "MultiframeTrueColorSecondaryCaptureImageStorage" << std::endl;
    return 1;
    }
  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality( "OT", 3, pf, pi, -1024, 1 );
  if( ms != gdcm::MediaStorage::MS_END )
    {
    std::cerr << "MultiframeTrueColorSecondaryCaptureImageStorage 2" << std::endl;
    return 1;
    }
  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality( "OT", 3, pf, pi, 0, 2 );
  if( ms != gdcm::MediaStorage::MS_END )
    {
    std::cerr << "MultiframeTrueColorSecondaryCaptureImageStorage 3" << std::endl;
    return 1;
    }

  return 0;
}
