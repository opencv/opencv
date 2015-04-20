/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMVR16EXPLICITDATAELEMENT_TXX
#define GDCMVR16EXPLICITDATAELEMENT_TXX

#include "gdcmSequenceOfItems.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmVL.h"
#include "gdcmParseException.h"
#include "gdcmImplicitDataElement.h"

#include "gdcmValueIO.h"
#include "gdcmSwapper.h"

namespace gdcm
{
//-----------------------------------------------------------------------------
template <typename TSwap>
std::istream &VR16ExplicitDataElement::Read(std::istream &is)
{
  ReadPreValue<TSwap>(is);
  return ReadValue<TSwap>(is);
}

template <typename TSwap>
std::istream &VR16ExplicitDataElement::ReadPreValue(std::istream &is)
{
  TagField.Read<TSwap>(is);
  // See PS 3.5, Data Element Structure With Explicit VR
  // Read Tag
  if( !is )
    {
    if( !is.eof() ) // FIXME This should not be needed
      {
      assert(0 && "Should not happen" );
      }
    return is;
    }
  assert( TagField != Tag(0xfffe,0xe0dd) );
  //assert( TagField != Tag(0xfeff,0xdde0) );
  const Tag itemDelItem(0xfffe,0xe00d);
  const Tag itemStartItem(0xfffe,0x0000);
  assert( TagField != itemStartItem );
  if( TagField == itemDelItem )
    {
    if( !ValueLengthField.Read<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
    if( ValueLengthField )
      {
      gdcmWarningMacro(
        "Item Delimitation Item has a length different from 0 and is: " << ValueLengthField );
      }
    // Set pointer to NULL to avoid user error
    ValueField = 0;
    return is;
    }

#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
  if( TagField == Tag(0x00ff, 0x4aa5) )
    {
    assert(0 && "Should not happen" );
    //  char c;
    //  is.read(&c, 1);
    //  std::cerr << "Debug: " << c << std::endl;
    }
#endif
  // Read VR
  // FIXME
  // Special hack for KONICA_VROX.dcm where in fact the VR=OX, in Pixel Data element
  // in which case we need to assume a 32bits VR...for now this is a big phat hack !
  bool OX_hack = false;
  try
    {
    if( !VRField.Read(is) )
      {
      assert(0 && "Should not happen" );
      return is;
      }
    }
  catch( Exception & )
    {
    VRField = VR::INVALID;
    // gdcm-MR-PHILIPS-16-Multi-Seq.dcm
    if( TagField == Tag(0xfffe, 0xe000) )
      {
      gdcmWarningMacro( "Found item delimitor in item" );
      ParseException pe;
      pe.SetLastElement( *this );
      throw pe;
      }
    // -> For some reason VR is written as {44,0} well I guess this is a VR...
    // Technically there is a second bug, dcmtk assume other things when reading this tag,
    // so I need to change this tag too, if I ever want dcmtk to read this file. oh well
    // 0019004_Baseline_IMG1.dcm
    // -> VR is garbage also...
    // assert( TagField == Tag(8348,0339) || TagField == Tag(b5e8,0338))
    if( TagField == Tag(0x7fe0,0x0010) )
      {
      OX_hack = true;
      VRField = VR::UN; // make it a fake 32bits for now...
      char dummy[2];
      is.read(dummy,2);
      assert( dummy[0] == 0 && dummy[1] == 0 );
      gdcmWarningMacro( "Assuming 32 bits VR for Tag=" <<
        TagField << " in order to read a buggy DICOM file." );
      }
    else
      {
      gdcmWarningMacro( "Assuming 16 bits VR for Tag=" <<
        TagField << " in order to read a buggy DICOM file." );
      }
    }
  // Read Value Length
  if( VR::GetLength(VRField) == 4 )
    {
    if( !ValueLengthField.Read<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
    if( OX_hack )
      {
      VRField = VR::INVALID; // revert to a pseudo unknown VR...
      }
    }
  else
    {
    assert( OX_hack == false );
    // 16bits only
    if( !ValueLengthField.template Read16<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    // HACK for SIEMENS Leonardo
    if( ValueLengthField == 0x0006
     && VRField == VR::UL
     && TagField.GetGroup() == 0x0009 )
      {
      gdcmWarningMacro( "Replacing VL=0x0006 with VL=0x0004, for Tag=" <<
        TagField << " in order to read a buggy DICOM file." );
      ValueLengthField = 0x0004;
      }
#endif
    }
  //std::cerr << "exp cur tag=" << TagField << " VR=" << VRField << " VL=" << ValueLengthField << std::endl;
  //
  // I don't like the following 3 lines, what if 0000,0000 was indeed -wrongly- sent, we should be able to continue
  // chances is that 99% of times there is now way we can reach here, so safely throw an exception
  if( TagField == Tag(0x0000,0x0000) && ValueLengthField == 0 && VRField == VR::INVALID )
    {
    // This handles DMCPACS_ExplicitImplicit_BogusIOP.dcm
    ParseException pe;
    pe.SetLastElement( *this );
    throw pe;
    }
  return is;
}

template <typename TSwap>
std::istream &VR16ExplicitDataElement::ReadValue(std::istream &is, bool readvalues )
{
  if( is.eof() ) return is;

  if( ValueLengthField == 0 )
    {
    // Simple fast path
    ValueField = 0;
    return is;
    }

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
        //if( !ValueIO<VR16ExplicitDataElement,TSwap>::Read(is,*ValueField) ) // non cp246
        if( !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,readvalues) ) // cp246 compliant
          {
          assert(0);
          }
        }
      catch( std::exception &)
        {
        // Must be one of those non-cp246 file...
        // but for some reason seekg back to previous offset + Read
        // as Explicit does not work...
        ParseException pe;
        pe.SetLastElement(*this);
        throw pe;
        }
      return is;
      }
    else
      {
      if( TagField != Tag(0x7fe0,0x0010) )
        {
        // gdcmSampleData/ForSeriesTesting/Perfusion/DICOMDIR
        ParseException pe;
        pe.SetLastElement(*this);
        throw pe;
        }
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
#if defined(GDCM_SUPPORT_BROKEN_IMPLEMENTATION) && 0
  // PHILIPS_Intera-16-MONO2-Uncompress.dcm
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
    gdcmWarningMacro( "ByteSwaping Private SQ: " << TagField );
    assert( VRField == VR::SQ );
    assert( TagField.IsPrivate() );
    try
      {
      if( !ValueIO<VR16ExplicitDataElement,SwapperDoOp>::Read(is,*ValueField,readvalues) )
        {
        assert(0 && "Should not happen");
        }
      Value* v = &*ValueField;
      SequenceOfItems *sq = dynamic_cast<SequenceOfItems*>(v);
      assert( sq );
      SequenceOfItems::Iterator it = sq->Begin();
      for( ; it != sq->End(); ++it)
        {
        Item &item = *it;
        DataSet &ds = item.GetNestedDataSet();
        ByteSwapFilter bsf(ds);
        bsf.ByteSwap();
        }
      }
    catch( std::exception &ex )
      {
      ValueLengthField = ValueField->GetLength();
      }
    return is;
    }
#endif

  if( !ValueIO<VR16ExplicitDataElement,TSwap>::Read(is,*ValueField,readvalues) )
    {
    // Might be the famous UN 16bits
    ParseException pe;
    pe.SetLastElement( *this );
    throw pe;
    return is;
    }

  return is;
}

template <typename TSwap>
std::istream &VR16ExplicitDataElement::ReadWithLength(std::istream &is, VL & length)
{
  return Read<TSwap>(is); (void)length;
}


} // end namespace gdcm

#endif // GDCMVR16EXPLICITDATAELEMENT_TXX
