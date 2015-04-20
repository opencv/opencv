/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTransferSyntax.h"

static const int losslylosslessarray[][3] = {
    { 0, 1, 1 }, //    ImplicitVRLittleEndian = 0,
    { 0, 1, 1 }, //    ImplicitVRBigEndianPrivateGE,
    { 0, 1, 1 }, //    ExplicitVRLittleEndian,
    { 0, 1, 1 }, //    DeflatedExplicitVRLittleEndian,
    { 0, 1, 1 }, //    ExplicitVRBigEndian,
    { 1, 0, 1 }, //    JPEGBaselineProcess1,
    { 1, 0, 1 }, //    JPEGExtendedProcess2_4,
    { 1, 0, 1 }, //    JPEGExtendedProcess3_5,
    { 1, 0, 1 }, //    JPEGSpectralSelectionProcess6_8,
    { 1, 0, 1 }, //    JPEGFullProgressionProcess10_12,
    { 0, 1, 0 }, //    JPEGLosslessProcess14,
    { 0, 1, 0 }, //    JPEGLosslessProcess14_1,
    { 0, 1, 0 }, //    JPEGLSLossless,
    { 1, 1, 1 }, //    JPEGLSNearLossless,
    { 0, 1, 0 }, //    JPEG2000Lossless,
    { 1, 1, 1 }, //    JPEG2000,
    { 0, 1, 0 }, //    JPEG2000Part2Lossless,
    { 1, 1, 1 }, //    JPEG2000Part2,
    { 0, 1, 0 }, //    RLELossless,
    { 1, 0, 1 }, //    MPEG2MainProfile,
    { 0, 1, 1 }, //    ImplicitVRBigEndianACRNEMA,
    { 0, 1, 1 }, //    WeirdPapryus,
    { 0, 1, 1 }, //    CT_private_ELE,
    { 1, 1, 1 }, //    JPIPReferenced
};

static int TestTransferSyntaxAll()
{
  for(int i = 0; i < gdcm::TransferSyntax::TS_END; ++i )
    {
    gdcm::TransferSyntax ts = (gdcm::TransferSyntax::TSType)i;
    const int *ll = losslylosslessarray[i];
    if( ll[0] )
      {
      if( !ts.IsLossy() )
        {
        std::cerr << "Lossy Problem with: " << gdcm::TransferSyntax::GetTSString( ts ) << std::endl;
        return 1;
        }
      }
    if( ll[1] )
      {
      if( !ts.IsLossless() )
        {
        std::cerr << "Lossless Problem with: " << gdcm::TransferSyntax::GetTSString( ts ) << std::endl;
        return 1;
        }
      }
    if( ll[2] )
      {
      if( !ts.CanStoreLossy() )
        {
        std::cerr << "CanLossy Problem with: " << gdcm::TransferSyntax::GetTSString( ts ) << std::endl;
        return 1;
        }
      }
    }
  return 0;
}

int TestTransferSyntax(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  if( TestTransferSyntaxAll() )
    {
    return 1;
    }
  gdcm::TransferSyntax ts;

  ts = gdcm::TransferSyntax::JPEG2000;
  if( !ts.IsLossless() )
    {
    return 1;
    }
  if( !ts.IsLossy() )
    {
    return 1;
    }
  ts = gdcm::TransferSyntax::JPEGLosslessProcess14_1;
  if( !ts.IsLossless() )
    {
    return 1;
    }
  if( ts.IsLossy() )
    {
    return 1;
    }
  ts = gdcm::TransferSyntax::DeflatedExplicitVRLittleEndian;
  if( !ts.IsLossless() )
    {
    return 1;
    }
  if( ts.IsLossy() )
    {
    return 1;
    }

  return 0;
}
