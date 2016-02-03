/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMUNEXPLICITDATAELEMENT_TXX
#define GDCMUNEXPLICITDATAELEMENT_TXX

#include "gdcmSequenceOfItems.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmVL.h"
#include "gdcmParseException.h"

#include "gdcmValueIO.h"
#include "gdcmSwapper.h"

namespace gdcm
{

//-----------------------------------------------------------------------------
template <typename TSwap>
std::istream &UNExplicitDataElement::Read(std::istream &is)
{
  ReadPreValue<TSwap>(is);
  return ReadValue<TSwap>(is);
}

template <typename TSwap>
std::istream &UNExplicitDataElement::ReadPreValue(std::istream &is)
{
  TagField.Read<TSwap>(is);
  // See PS 3.5, Data Element Structure With UNExplicit VR
  // Read Tag
  if( !is )
    {
    if( !is.eof() ) // FIXME This should not be needed
      {
      assert(0 && "Should not happen" );
      }
    return is;
    }
  if( TagField == Tag(0xfffe,0xe0dd) )
    {
    ParseException pe;
    pe.SetLastElement( *this );
    throw pe;
    }
  assert( TagField != Tag(0xfffe,0xe0dd) );

  const Tag itemDelItem(0xfffe,0xe00d);
  if( TagField == itemDelItem )
    {
    if( !ValueLengthField.Read<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
    if( ValueLengthField != 0 )
      {
      gdcmDebugMacro(
        "Item Delimitation Item has a length different from 0" );
      }
    // Set pointer to NULL to avoid user error
    ValueField = 0;
    VRField = VR::INVALID;
    return is;
    }

  // Read VR
  try
    {
    if( !VRField.Read(is) )
      {
      assert(0 && "Should not happen" );
      return is;
      }
    }
  catch( Exception &ex )
    {
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    // gdcm-MR-PHILIPS-16-Multi-Seq.dcm
    // assert( TagField == Tag(0xfffe, 0xe000) );
    // -> For some reason VR is written as {44,0} well I guess this is a VR...
    // Technically there is a second bug, dcmtk assume other things when reading this tag,
    // so I need to change this tag too, if I ever want dcmtk to read this file. oh well
    // 0019004_Baseline_IMG1.dcm
    // -> VR is garbage also...
    // assert( TagField == Tag(8348,0339) || TagField == Tag(b5e8,0338))
    //gdcmWarningMacro( "Assuming 16 bits VR for Tag=" <<
    //  TagField << " in order to read a buggy DICOM file." );
    //VRField = VR::INVALID;
    (void)ex;
    ParseException pe;
    pe.SetLastElement( *this );
    throw pe;
#else
  throw ex;
#endif /* GDCM_SUPPORT_BROKEN_IMPLEMENTATION */
    }

  if( VRField == VR::UN )
    {
    // backtrack ...
    is.seekg(-2, std::ios::cur);
    }
  // Read Value Length
  if( VR::GetLength(VRField) == 4 && VRField != VR::UN )
    {
    if( !ValueLengthField.Read<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
    }
  else
    {
    // 16bits only
    if( !ValueLengthField.template Read16<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
    }
  return is;
}

template <typename TSwap>
std::istream &UNExplicitDataElement::ReadValue(std::istream &is, bool readvalues)
{
  if( is.eof() ) return is;
  if( ValueLengthField == 0 )
    {
    // Simple fast path
    ValueField = 0;
    return is;
    }

  //std::cerr << "exp cur tag=" << TagField << " VR=" << VRField << " VL=" << ValueLengthField << std::endl;
  // Read the Value
  //assert( ValueField == 0 );
  if( VRField == VR::SQ )
    {
    // Check wether or not this is an undefined length sequence
    assert( TagField != Tag(0x7fe0,0x0010) );
    ValueField = new SequenceOfItems;
    }
  else if( ValueLengthField.IsUndefined() )
    {
    if( VRField == VR::UN )
      {
      // Support cp246 conforming file:
      // Enhanced_MR_Image_Storage_PixelSpacingNotIn_0028_0030.dcm (illegal)
      // vs
      // undefined_length_un_vr.dcm
      assert( TagField != Tag(0x7fe0,0x0010) );
      ValueField = new SequenceOfItems;
      ValueField->SetLength(ValueLengthField); // perform realloc
      try
        {
        //if( !ValueIO<UNExplicitDataElement,TSwap>::Read(is,*ValueField) ) // non cp246
        if( !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,readvalues) ) // cp246 compliant
          {
          assert(0);
          }
        }
      catch( std::exception &)
        {
        // Must be one of those non-cp246 file...
        // but for some reason seekg back to previous offset + Read
        // as UNExplicit does not work...
        throw Exception( "CP 246" );
        }
      return is;
      }
    else
      {
      // Ok this is Pixel Data fragmented...
      assert( TagField == Tag(0x7fe0,0x0010) );
      assert( VRField & VR::OB_OW );
      ValueField = new SequenceOfFragments;
      }
    }
  else
    {
    //assert( TagField != Tag(0x7fe0,0x0010) );
    ValueField = new ByteValue;
    }
  // We have the length we should be able to read the value
  ValueField->SetLength(ValueLengthField); // perform realloc


  if( TagField == Tag(0x2001,0xe05f)
    || TagField == Tag(0x2001,0xe100)
    || TagField == Tag(0x2005,0xe080)
    || TagField == Tag(0x2005,0xe083)
    || TagField == Tag(0x2005,0xe084)
    || TagField == Tag(0x2005,0xe402)
    //TagField.IsPrivate() && VRField == VR::SQ
    //-> Does not work for 0029
    //we really need to read item marker
  )
    {
    assert(0); // Could we possibly be so unlucky to have this mixture of bugs...
    }

  if( !ValueIO<UNExplicitDataElement,TSwap>::Read(is,*ValueField,readvalues) )
    {
    ParseException pe;
    pe.SetLastElement( *this );
    throw pe;
    return is;
    }

  return is;
}

template <typename TSwap>
std::istream &UNExplicitDataElement::ReadWithLength(std::istream &is, VL & length)
{
  return Read<TSwap>(is); (void)length;
}


} // end namespace gdcm

#endif // GDCMUNEXPLICITDATAELEMENT_TXX
