/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageFragmentSplitter.h"
#include "gdcmSequenceOfFragments.h"

namespace gdcm
{

bool ImageFragmentSplitter::Split()
{
  Output = Input;
  const Bitmap &image = *Input;

  const unsigned int *dims = image.GetDimensions();
  if( dims[2] != 1 )
    {
    gdcmDebugMacro( "Cannot split a 3D image" );
    return false;
    }
  const DataElement& pixeldata = image.GetDataElement();

  const SequenceOfFragments *sqf = pixeldata.GetSequenceOfFragments();
  if( !sqf )
    {
    gdcmDebugMacro( "Cannot split a non-encapsulated syntax" );
    return false;
    }

  if ( sqf->GetNumberOfFragments() != 1 )
    {
    gdcmDebugMacro( "Case not handled (for now)" );
    return false;
    }

  //assert( sqf->GetNumberOfFragments() == 1 );

  // WARNING do not keep the same Basic Offset Table...
  const Fragment& frag = sqf->GetFragment(0);
  const ByteValue *bv = frag.GetByteValue();
  const char *p = bv->GetPointer();
  unsigned long len = bv->GetLength();
  if( FragmentSizeMax > len && !Force )
    {
    // I think it is ok
    return true;
    }
  // prevent zero division
  if( FragmentSizeMax == 0 )
    {
    gdcmDebugMacro( "Need to set a real value for fragment size" );
    return false; // seriously...
    }
  unsigned long nfrags = len / FragmentSizeMax;
  unsigned long lastfrag = len % FragmentSizeMax;

  SmartPointer<SequenceOfFragments> sq = new SequenceOfFragments;
  // Let's do all complete frag:
  for(unsigned long i = 0; i < nfrags; ++i)
    {
    Fragment splitfrag;
    splitfrag.SetByteValue( p + i * FragmentSizeMax, FragmentSizeMax);
    sq->AddFragment( splitfrag );
    }
  // Last (incomplete one):
  if( lastfrag )
    {
    Fragment splitfrag;
    splitfrag.SetByteValue( p + nfrags * FragmentSizeMax, (uint32_t)lastfrag );
    assert( nfrags * FragmentSizeMax + lastfrag == len );
    sq->AddFragment( splitfrag );
    }
  Output->GetDataElement().SetValue( *sq );

  bool success = true;
  return success;
}

void ImageFragmentSplitter::SetFragmentSizeMax(unsigned int fragsize)
{
/*
 * A.4 TRANSFER SYNTAXES FOR ENCAPSULATION OF ENCODED PIXEL DATA
 *
 * All items containing an encoded fragment shall be made of an even number of bytes
 * greater or equal to two. The last fragment of a frame may be padded, if necessary,
 * to meet the sequence item format requirements of the DICOM Standard.
 */
  FragmentSizeMax = fragsize;
  if( fragsize % 2 )
    {
    // what if FragmentSizeMax == 0 ...
    FragmentSizeMax--;
    }
  // How do I handle this one...
  if( fragsize < 2 )
    {
    FragmentSizeMax = 2;
    }
  // \postcondition:
  assert( FragmentSizeMax >= 2 && (FragmentSizeMax % 2) == 0 );
}

} // end namespace gdcm
