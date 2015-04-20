/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageChangePlanarConfiguration.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmFragment.h"

namespace gdcm
{
/*
 * C.7.6.3.1.3 Planar Configuration
 * Note: Planar Configuration (0028,0006) is not meaningful when a compression transfer syntax is
 * used that involves reorganization of sample components in the compressed bit stream. In such
 * cases, since the Attribute is required to be sent, then an appropriate value to use may be
 * specified in the description of the Transfer Syntax in PS 3.5, though in all likelihood the value of
 * the Attribute will be ignored by the receiving implementation.
 */

bool ImageChangePlanarConfiguration::Change()
{
  if( PlanarConfiguration != 0 && PlanarConfiguration != 1 ) return false; // seriously
  Output = Input;
  if( Input->GetPixelFormat().GetSamplesPerPixel() != 3 )
    {
    return true;
    }
  assert( Input->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL
    || Input->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL_422
    || Input->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_RCT
    || Input->GetPhotometricInterpretation() == PhotometricInterpretation::RGB );
  if( Input->GetPlanarConfiguration() == PlanarConfiguration )
    {
    return true;
    }

  const Bitmap &image = *Input;

  const unsigned int *dims = image.GetDimensions();
  unsigned long len = image.GetBufferLength();
  char *p = new char[len];
  image.GetBuffer( p );

  assert( len % 3 == 0 );
  const size_t ps = Input->GetPixelFormat().GetPixelSize();
  const size_t framesize = dims[0] * dims[1] * ps;
  assert( framesize * dims[2] == len );

  char *copy = new char[len];
  size_t size = framesize / 3;
  if( PlanarConfiguration == 0 )
    {
    for(unsigned int z = 0; z < dims[2]; ++z)
      {
      const char *frame = p + z * framesize;
      const char *r = frame + 0;
      const char *g = frame + size;
      const char *b = frame + size + size;

      char *framecopy = copy + z * framesize;
      ImageChangePlanarConfiguration::RGBPlanesToRGBPixels(framecopy, r, g, b, size);
      }
    }
  else // User requested to do PlanarConfiguration == 1
    {
    assert( PlanarConfiguration == 1 );
    for(unsigned int z = 0; z < dims[2]; ++z)
      {
      const char *frame = p + z * framesize;
      char *framecopy = copy + z * framesize;
      char *r = framecopy + 0;
      char *g = framecopy + size;
      char *b = framecopy + size + size;

      ImageChangePlanarConfiguration::RGBPixelsToRGBPlanes(r, g, b, frame, size);
      }
    }
  delete[] p;

  DataElement &de = Output->GetDataElement();
  de.SetByteValue( copy, (uint32_t)len );
  delete[] copy;

  Output->SetPlanarConfiguration( PlanarConfiguration );
  if( Input->GetTransferSyntax().IsImplicit() )
    {
    assert( Output->GetTransferSyntax().IsImplicit() );
    }
  else if( Input->GetTransferSyntax() == TransferSyntax::ExplicitVRBigEndian )
    {
    Output->SetTransferSyntax( TransferSyntax::ExplicitVRBigEndian );
    }
  else
    {
    Output->SetTransferSyntax( TransferSyntax::ExplicitVRLittleEndian );
    }
  //assert( Output->GetTransferSyntax().IsRaw() );
  assert( Output->GetPhotometricInterpretation() == Input->GetPhotometricInterpretation() );

  return true;
}


} // end namespace gdcm
