/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageApplyLookupTable.h"

#include <limits>

namespace gdcm
{

bool ImageApplyLookupTable::Apply()
{
  Output = Input;
  const Bitmap &image = *Input;

  PhotometricInterpretation pi = image.GetPhotometricInterpretation();
  if( pi != PhotometricInterpretation::PALETTE_COLOR )
    {
    gdcmDebugMacro( "Image is not palettized" );
    return false;
    }
  const LookupTable &lut = image.GetLUT();
  int bitsample = lut.GetBitSample();
  if( !bitsample ) return false;

  const unsigned long len = image.GetBufferLength();
  std::vector<char> v;
  v.resize( len );
  char *p = &v[0];
  image.GetBuffer( p );
  std::stringstream is;
  if( !is.write( p, len ) )
    {
    gdcmErrorMacro( "Could not write to stringstream" );
    return false;
    }

  DataElement &de = Output->GetDataElement();
#if 0
  std::ostringstream os;
  lut.Decode(is, os);
  const std::string str = os.str();
  VL::Type strSize = (VL::Type)str.size();
  de.SetByteValue( str.c_str(), strSize);
#else
  std::vector<char> v2;
  v2.resize( len * 3 );
  lut.Decode(&v2[0], v2.size(), &v[0], v.size());
  assert( v2.size() < (size_t)std::numeric_limits<uint32_t>::max );
  de.SetByteValue( &v2[0], (uint32_t)v2.size());
#endif
  Output->GetLUT().Clear();
  Output->SetPhotometricInterpretation( PhotometricInterpretation::RGB );
  Output->GetPixelFormat().SetSamplesPerPixel( 3 );
  Output->SetPlanarConfiguration( 0 ); // FIXME OT-PAL-8-face.dcm has a PlanarConfiguration while being PALETTE COLOR...
  const TransferSyntax &ts = image.GetTransferSyntax();
  //assert( ts == TransferSyntax::RLELossless );
  if( ts.IsExplicit() )
    {
    Output->SetTransferSyntax( TransferSyntax::ExplicitVRLittleEndian );
    }
  else
    {
    assert( ts.IsImplicit() );
    Output->SetTransferSyntax( TransferSyntax::ImplicitVRLittleEndian );
    }


  bool success = true;
  return success;
}


} // end namespace gdcm
