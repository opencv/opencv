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
#include "gdcmTrace.h"

#include <assert.h>
#include <string.h>

#include <string>
#include <iostream>

namespace gdcm
{

//#include "gdcmUIDs.cxx"

static const char *TSStrings[] = {
    // Implicit VR Little Endian
  "1.2.840.10008.1.2",
  // Implicit VR Big Endian DLX (G.E Private)
  "1.2.840.113619.5.2",
  // Explicit VR Little Endian
  "1.2.840.10008.1.2.1",
  // Deflated Explicit VR Little Endian
  "1.2.840.10008.1.2.1.99",
  // Explicit VR Big Endian
  "1.2.840.10008.1.2.2",
  // JPEG Baseline (Process 1)
  "1.2.840.10008.1.2.4.50",
  // JPEG Extended (Process 2 & 4)
  "1.2.840.10008.1.2.4.51",
  // JPEG Extended (Process 3 & 5)
  "1.2.840.10008.1.2.4.52",
  // JPEG Spectral Selection, Non-Hierarchical (Process 6 & 8)
  "1.2.840.10008.1.2.4.53",
  // JPEG Full Progression, Non-Hierarchical (Process 10 & 12)
  "1.2.840.10008.1.2.4.55",
  // JPEG Lossless, Non-Hierarchical (Process 14)
  "1.2.840.10008.1.2.4.57",
  // JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14,
  //                                                       [Selection Value 1])
  "1.2.840.10008.1.2.4.70",
  // JPEG-LS Lossless Image Compression
  "1.2.840.10008.1.2.4.80",
  // JPEG-LS Lossy (Near-Lossless) Image Compression
  "1.2.840.10008.1.2.4.81",
  // JPEG 2000 Lossless
  "1.2.840.10008.1.2.4.90",
  // JPEG 2000
  "1.2.840.10008.1.2.4.91",
  // JPEG 2000 Part 2 Lossless
  "1.2.840.10008.1.2.4.92",
  // JPEG 2000 Part 2
  "1.2.840.10008.1.2.4.93",
  // RLE Lossless
  "1.2.840.10008.1.2.5",
  // MPEG2 Main Profile @ Main Level
  "1.2.840.10008.1.2.4.100",
  // Old ACR NEMA, fake a TS
  "ImplicitVRBigEndianACRNEMA",
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
  // Weird Papyrus
  "1.2.840.10008.1.20",
#endif
  "1.3.46.670589.33.1.4.1",
  // JPIP Referenced
  "1.2.840.10008.1.2.4.94",
  // Unknown
  "Unknown Transfer Syntax", // Pretty sure we never use this case...
  0 // Compilers have no obligation to finish by NULL, do it ourself
};

TransferSyntax::TSType TransferSyntax::GetTSType(const char *cstr)
{
  // trim trailing whitespace
  std::string str = cstr;
  std::string::size_type notspace = str.find_last_not_of(" ") + 1;
  if( notspace != str.size() )
    {
    gdcmDebugMacro( "BUGGY HEADER: TS contains " <<
      str.size()-notspace << " whitespace character(s)" );
    str.erase(notspace);
    }

  int i = 0;
  while(TSStrings[i] != 0)
  //while(TransferSyntaxStrings[i] != 0)
    {
    if( str == TSStrings[i] )
    //if( str == TransferSyntaxStrings[i] )
      return (TSType)i;
    ++i;
    }
  return TS_END;
}

const char* TransferSyntax::GetTSString(TSType ts)
{
  assert( ts <= TS_END );
  return TSStrings[(int)ts];
  //return TransferSyntaxStrings[(int)ts];
}

bool TransferSyntax::IsImplicit(TSType ts) const
{
  assert( ts != TS_END );
  return ts == ImplicitVRLittleEndian
    || ts == ImplicitVRBigEndianACRNEMA
    || ts == ImplicitVRBigEndianPrivateGE
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    || ts == WeirdPapryus
#endif
    ;
}

bool TransferSyntax::IsImplicit() const
{
  if ( TSField == TS_END ) return false;
  return TSField == ImplicitVRLittleEndian
    || TSField == ImplicitVRBigEndianACRNEMA
    || TSField == ImplicitVRBigEndianPrivateGE
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    || TSField == WeirdPapryus
#endif
    ;
}

bool TransferSyntax::IsExplicit() const
{
  if ( TSField == TS_END ) return false; // important !
  return !IsImplicit();
}

bool TransferSyntax::IsLossy() const
{
  if (
    TSField == JPEGBaselineProcess1 ||
    TSField == JPEGExtendedProcess2_4 ||
    TSField == JPEGExtendedProcess3_5 ||
    TSField == JPEGSpectralSelectionProcess6_8 ||
    TSField == JPEGFullProgressionProcess10_12 ||
    TSField == JPEGLSNearLossless ||
    TSField == JPEG2000 ||
    TSField == JPEG2000Part2 ||
    TSField == JPIPReferenced ||
    TSField == MPEG2MainProfile
  )
    {
    return true;
    }
  return false;

}

// This function really test the kind of compression algorithm and the matching
// transfer syntax.  If you use the JPEG compression algorithm (ITU-T T.81,
// ISO/IEC IS 10918-1), You will not be able to declare a lossy compress pixel
// data using JPEGLosslessProcess14_1 For the same reason using J2K (ITU-T
// T.800, ISO/IEC IS 15444-1), you shoult not be allowed to stored an
// irreversible wavelet compressed pixel data in a file declared with transfer
// syntax JPEG2000Lossless.
// Same goes for JPEG-LS (ITU-T T.87, ISO/IEC IS 14495-1), and to some extent
// RLE which does not even allow lossy compression...
bool TransferSyntax::CanStoreLossy() const
{
  if (
    TSField == JPEGLosslessProcess14 ||
    TSField == JPEGLosslessProcess14_1 ||
    TSField == JPEGLSLossless ||
    TSField == JPEG2000Lossless ||
    TSField == JPEG2000Part2Lossless ||
    TSField == RLELossless
  )
    {
    return false;
    }
  return true;
}

bool TransferSyntax::IsLossless() const
{
  if (
    TSField == JPEGBaselineProcess1 ||
    TSField == JPEGExtendedProcess2_4 ||
    TSField == JPEGExtendedProcess3_5 ||
    TSField == JPEGSpectralSelectionProcess6_8 ||
    TSField == JPEGFullProgressionProcess10_12 ||
    // TSField == JPEGLSNearLossless || -> can be lossy & lossless
    // TSField == JPEG2000 || -> can be lossy & lossless
    // TSField == JPEG2000Part2 || -> can be lossy & lossless
    // TSField == JPIPReferenced || -> can be lossy & lossless
    TSField == MPEG2MainProfile
  )
    {
    return false;
    }
  return true;
}

// By implementation those two functions form a partition
bool TransferSyntax::IsExplicit(TSType ts) const
{
  assert( ts != TS_END );
  return !IsImplicit(ts);
}

TransferSyntax::NegociatedType TransferSyntax::GetNegociatedType() const
{
  if( TSField == TS_END )
    {
    return TransferSyntax::Unknown;
    }
  else if( IsImplicit(TSField) )
    {
    return TransferSyntax::Implicit;
    }
  return TransferSyntax::Explicit;
}

bool TransferSyntax::IsLittleEndian(TSType ts) const
{
  assert( ts != TS_END );
  return !IsBigEndian(ts);
}

bool TransferSyntax::IsBigEndian(TSType ts) const
{
  assert( ts != TS_END );
  return ts == ExplicitVRBigEndian
//    || ts == ImplicitVRBigEndianPrivateGE // Indeed this is LittleEndian
    || ts == ImplicitVRBigEndianACRNEMA;
}

SwapCode TransferSyntax::GetSwapCode() const
{
  assert( TSField != TS_END );
  if( IsBigEndian( TSField ) )
    {
    return SwapCode::BigEndian;
    }
  assert( IsLittleEndian( TSField ) );
  return SwapCode::LittleEndian;
}

bool TransferSyntax::IsEncoded() const
{
  return TSField == DeflatedExplicitVRLittleEndian;
}

bool TransferSyntax::IsEncapsulated() const
{
  bool ret = false;
  switch( TSField )
    {
  //case ImplicitVRLittleEndian:
  //case ImplicitVRBigEndianPrivateGE:
  //case ExplicitVRLittleEndian:
  //case DeflatedExplicitVRLittleEndian:
  //case ExplicitVRBigEndian:
  case JPEGBaselineProcess1:
  case JPEGExtendedProcess2_4:
  case JPEGExtendedProcess3_5:
  case JPEGSpectralSelectionProcess6_8:
  case JPEGFullProgressionProcess10_12:
  case JPEGLosslessProcess14:
  case JPEGLosslessProcess14_1:
  case JPEGLSLossless:
  case JPEGLSNearLossless:
  case JPEG2000Lossless:
  case JPEG2000:
  case JPEG2000Part2Lossless:
  case JPEG2000Part2:
  case JPIPReferenced:
  case RLELossless:
  case MPEG2MainProfile:
  //case ImplicitVRBigEndianACRNEMA:
  //case WeirdPapryus:
    ret = true;
    break;
  default:
    ;
    }
  return ret;
}

} // end namespace gdcm
