/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIconImageFilter.h"
#include "gdcmTesting.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmImageReader.h"
#include "gdcmImageWriter.h"
#include "gdcmImage.h"

static const char * const iconimagearray[][2] = {
  { "b818c90fc4135423dfc118c3305d23ef" , "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm" },
  { "57e43cf467d9bc4c4a43e0a97329075d" , "SIEMENS_CSA2.dcm" },
  { "fc5db4e2e7fca8445342b83799ff16d8" , "simpleImageWithIcon.dcm" },
  { "93c1b9e4c97cf5ff3501f3d8114c3b89" , "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm" },
  { "07950df5d3662740874e93d2c41dec18" , "MR-SIEMENS-DICOM-WithOverlays.dcm" },
  { "e1305d5341e8ced04caff40c706c23b0" , "AMIInvalidPrivateDefinedLengthSQasUN.dcm" },
  { "5f76af83e7b99cab45a70248824c2145" , "PICKER-16-MONO2-Nested_icon.dcm" },
  { "59d8479e0025d8bbb3244551d6535890" , "MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm" },
  { "9ad43e72601a228c4a3d021f08a09b69" , "CT-SIEMENS-Icone-With-PaletteColor.dcm" },
  { "a9c3c78082a46e2226a4b1ff499ccd74" , "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" },
  { "50387cdd945b22ccbb5d1824e955deeb" , "05148044-mr-siemens-avanto-syngo.dcm" },
  { "e42ea2852be5d20f95083991728d8623" , "GE_LOGIQBook-8-RGB-HugePreview.dcm" },
  { "c38df20a8514714f5d5af1699a841c60" , "GE_CT_With_Private_compressed-icon.dcm" },
  { "61b2bf04c18a0f67b7e720e07804dcdd" , "KODAK_CompressedIcon.dcm" },
  { "620f0b67a91f7f74151bc5be745b7110" , "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" },
  { "938f6ea0bea13ff5c45c7934e603caac" , "US-GE-4AICL142.dcm" },

  // VEPRO VIF
  { "ea4673b2aa72f477188bac340e115f4c" , "PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm" },
  { "660417e04b9af62832a43bf82369e4fa" , "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm" },

  // gdcmDataExtra
  { "a144b851c9262c97dde567d4d3781733" , "2929J888_8b_YBR_RLE_PlanConf0_breaker.dcm" },

  // sentinel
  { 0, 0 }
};

int TestIconImageFilterFunc(const char *filename, bool verbose = false)
{
  gdcm::ImageReader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    return 0;
    }

  gdcm::IconImageFilter iif;
  iif.SetFile( reader.GetFile() );
  bool b = iif.Extract();

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();

  unsigned int i = 0;
  const char *p = iconimagearray[i][1];
  while( p != 0 )
    {
    if( strcmp( name, p ) == 0 )
      {
      break;
      }
    ++i;
    p = iconimagearray[i][1];
    }
  const char *refmd5 = iconimagearray[i][0];

  if( b )
    {
    if( iif.GetNumberOfIconImages() != 1 ) return 1;

    const gdcm::IconImage &icon = iif.GetIconImage(0);
    if( verbose ) icon.Print( std::cout );
    unsigned long len = icon.GetBufferLength();
    std::vector< char > vbuffer;
    vbuffer.resize( len );
    char *buffer = &vbuffer[0];
    bool res2 = icon.GetBuffer(buffer);
    if( !res2 )
      {
      std::cerr << "res2 failure:" << filename << std::endl;
      return 1;
      }
    char digest[33];
    gdcm::Testing::ComputeMD5(buffer, len, digest);
    if( verbose )
      {
      std::cout << "ref=" << refmd5 << std::endl;
      std::cout << "md5=" << digest << std::endl;
      }
    if( !refmd5 )
      {
      std::cerr << "Problem with : " << name << " missing md5= " << digest << std::endl;
      return 1;
      }
    if( strcmp( refmd5, digest) )
      {
      std::cerr << "Problem with : " << name << " " << refmd5 << " vs " << digest << std::endl;
      return 1;
      }
    }
  else
    {
    assert( refmd5 == 0 );
    }

  return 0;
}

int TestIconImageFilter(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestIconImageFilterFunc(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestIconImageFilterFunc( filename );
    ++i;
    }

  return r;
}
