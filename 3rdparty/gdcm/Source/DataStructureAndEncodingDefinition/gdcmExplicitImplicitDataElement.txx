/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMEXPLICITIMPLICITDATAELEMENT_TXX
#define GDCMEXPLICITIMPLICITDATAELEMENT_TXX

#include "gdcmExplicitImplicitDataElement.h"

#include "gdcmSequenceOfItems.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmVL.h"
#include "gdcmExplicitDataElement.h"
#include "gdcmImplicitDataElement.h"
#include "gdcmValueIO.h"
#include "gdcmSwapper.h"

namespace gdcm
{
//-----------------------------------------------------------------------------
template <typename TSwap>
std::istream &ExplicitImplicitDataElement::Read(std::istream &is)
{
  ReadPreValue<TSwap>(is);
  return ReadValue<TSwap>(is);
}

template <typename TSwap>
std::istream &ExplicitImplicitDataElement::ReadPreValue(std::istream &is)
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
  if( TagField == Tag(0xfffe,0xe0dd) )
    {
    ParseException pe;
    pe.SetLastElement( *this );
    throw pe;
    }
  //assert( TagField != Tag(0xfeff,0xdde0) );
  const Tag itemDelItem(0xfffe,0xe00d);
  if( TagField == itemDelItem )
    {
    //ParseException pe;
    //pe.SetLastElement( *this );
    //throw pe;
    if( !ValueLengthField.Read<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
    if( ValueLengthField )
      {
      gdcmDebugMacro(
        "Item Delimitation Item has a length different from 0 and is: " << ValueLengthField );
      }
    // Set pointer to NULL to avoid user error
    ValueField = 0;
    VRField = VR::INVALID;
    return is;
    }

#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
  if( TagField == Tag(0x00ff, 0x4aa5) )
    {
    //assert(0 && "Should not happen" );
    // gdcmDataExtra/gdcmBreakers/DigitexAlpha_no_7FE0.dcm
    is.seekg( -4, std::ios::cur );
    TagField = Tag(0x7fe0,0x0010);
    VRField = VR::OW;
    ValueField = new ByteValue;
    std::streampos s = is.tellg();
    is.seekg( 0, std::ios::end);
    std::streampos e = is.tellg();
    is.seekg( s, std::ios::beg );
    ValueField->SetLength( (int32_t)(e - s) );
    ValueLengthField = ValueField->GetLength();
    bool failed = !ValueIO<ExplicitDataElement,TSwap,uint16_t>::Read(is,*ValueField,true);
    gdcmAssertAlwaysMacro( !failed );
    return is;
    //throw Exception( "Unhandled" );
    }
#endif
  // Read VR
  try
    {
    if( !VRField.Read(is) )
      {
      assert(0 && "Should not happen" );
      return is;
      }
  // Read Value Length
  if( VR::GetLength(VRField) == 4 )
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
    }
  catch( Exception &ex )
    {
    (void)ex;
    VRField = VR::INVALID;
    is.seekg( -2, std::ios::cur );

  const Tag itemStartItem(0xfffe,0xe000);
  if( TagField == itemStartItem ) return is;

  //assert( TagField != Tag(0xfffe,0xe0dd) );
  // Read Value Length
  if( !ValueLengthField.Read<TSwap>(is) )
    {
    //assert(0 && "Should not happen");
    throw Exception("Impossible");
    return is;
    }
  //std::cerr << "imp cur tag=" << TagField <<  " VL=" << ValueLengthField << std::endl;
  if( ValueLengthField == 0 )
    {
    // Simple fast path
    ValueField = 0;
    return is;
    }
  else if( ValueLengthField.IsUndefined() )
    {
    //assert( de.GetVR() == VR::SQ );
    // FIXME what if I am reading the pixel data...
    //assert( TagField != Tag(0x7fe0,0x0010) );
    if( TagField != Tag(0x7fe0,0x0010) )
      {
      ValueField = new SequenceOfItems;
      }
    else
      {
      gdcmErrorMacro( "Undefined value length is impossible in non-encapsulated Transfer Syntax" );
      ValueField = new SequenceOfFragments;
      }
    //VRField = VR::SQ;
    }
  else
    {
    if( true /*ValueLengthField < 8 */ )
      {
      ValueField = new ByteValue;
      }
    else
      {
      // In the following we read 4 more bytes in the Value field
      // to find out if this is a SQ or not
      // there is still work to do to handle the PMS featured SQ
      // where item Start is in fact 0xfeff, 0x00e0 ... sigh
      const Tag itemStart(0xfffe, 0xe000);
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
      const Tag itemPMSStart(0xfeff, 0x00e0);
      const Tag itemPMSStart2(0x3f3f, 0x3f00);
#endif
      Tag item;
      // TODO FIXME
      // This is pretty dumb to actually read to later on seekg back, why not `peek` directly ?
      item.Read<TSwap>(is);
      // Maybe this code can later be rewritten as I believe that seek back
      // is very slow...
      is.seekg(-4, std::ios::cur );
      if( item == itemStart )
        {
        assert( TagField != Tag(0x7fe0,0x0010) );
        ValueField = new SequenceOfItems;
        }
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
      else if ( item == itemPMSStart )
        {
        // MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm
        gdcmWarningMacro( "Illegal: Explicit SQ found in a file with "
          "TransferSyntax=Implicit for tag: " << TagField );
        // TODO: We READ Explicit ok...but we store Implicit !
        // Indeed when copying the VR will be saved... pretty cool eh ?
        ValueField = new SequenceOfItems;
        ValueField->SetLength(ValueLengthField); // perform realloc
        try
          {
          if( !ValueIO<ExplicitDataElement,SwapperDoOp>::Read(is,*ValueField,true) )
            {
            assert(0 && "Should not happen");
            }
          }
        catch( std::exception &ex2 )
          {
          (void)ex2;
          ValueLengthField = ValueField->GetLength();
          }
        return is;
        }
      else if ( item == itemPMSStart2 && false )
        {
        gdcmWarningMacro( "Illegal: SQ start with " << itemPMSStart2
          << " instead of " << itemStart << " for tag: " << TagField );
        ValueField = new SequenceOfItems;
        ValueField->SetLength(ValueLengthField); // perform realloc
        if( !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,true) )
          {
          assert(0 && "Should not happen");
          }
        return is;
        }
#endif
      else
        {
        ValueField = new ByteValue;
        }
      }
    }
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
  // THE WORST BUG EVER. From GE Workstation
  if( ValueLengthField == 13 )
    {
    // Historically gdcm did not enforce proper length
    // thus Theralys started writing illegal DICOM images:
    const Tag theralys1(0x0008,0x0070);
    const Tag theralys2(0x0008,0x0080);
    if( TagField != theralys1
     && TagField != theralys2 )
      {
      gdcmWarningMacro( "GE,13: Replacing VL=0x000d with VL=0x000a, for Tag="
        << TagField << " in order to read a buggy DICOM file." );
      ValueLengthField = 10;
      }
    }
#endif
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
  if( ValueLengthField == 0x31f031c && TagField == Tag(0x031e,0x0324) )
    {
    // TestImages/elbow.pap
    gdcmWarningMacro( "Replacing a VL. To be able to read a supposively"
      "broken Payrus file." );
    ValueLengthField = 202; // 0xca
    }
#endif
  // We have the length we should be able to read the value
  ValueField->SetLength(ValueLengthField); // perform realloc
  if( !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,true) )
    {
    // Special handling for PixelData tag:
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    if( TagField == Tag(0x7fe0,0x0010) )
      {
      gdcmWarningMacro( "Incomplete Pixel Data found, use file at own risk" );
      is.clear();
      }
    else
#endif /* GDCM_SUPPORT_BROKEN_IMPLEMENTATION */
      {
      throw Exception("Should not happen (imp)");
      }
    return is;
    }

#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
  // dcmtk 3.5.4 is resilient to broken explicit SQ length and will properly recompute it
  // as long as each of the Item lengths are correct
  VL dummy = ValueField->GetLength();
  if( ValueLengthField != dummy )
    {
    gdcmWarningMacro( "ValueLengthField was bogus" ); assert(0);
    ValueLengthField = dummy;
    }
#else
  assert( ValueLengthField == ValueField->GetLength() );
  assert( VRField == VR::INVALID );
#endif

  return is;

    }
  //std::cerr << "exp cur tag=" << TagField << " VR=" << VRField << " VL=" << ValueLengthField << std::endl;
  //
  // I don't like the following 3 lines, what if 0000,0000 was indeed -wrongly- sent, we should be able to continue
  // chances is that 99% of times there is now way we can reach here, so safely throw an exception
  if( TagField == Tag(0x0000,0x0000) && ValueLengthField == 0 && VRField == VR::INVALID )
    {
    ParseException pe;
    pe.SetLastElement( *this );
    throw pe;
    }

#ifdef ELSCINT1_01F7_1070
  if( TagField == Tag(0x01f7,0x1070) )
    {
    ValueLengthField = ValueLengthField - 7;
    }
#endif

  return is;
}

template <typename TSwap>
std::istream &ExplicitImplicitDataElement::ReadValue(std::istream &is, bool readvalues)
{
  if( is.eof() ) return is;
  /* thechnically the following is bad
     it assumes that in the case of explicit/implicit dataset
     we are not handle the prevalue call properly for buggy implicit attribute
   */
  if( VRField == VR::INVALID ) return is;

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
        //if( !ValueIO<ExplicitDataElement,TSwap>::Read(is,*ValueField) ) // non cp246
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
  this->SetValueFieldLength( ValueLengthField, readvalues );
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
      if( !ValueIO<ExplicitDataElement,SwapperDoOp>::Read(is,*ValueField,readvalues) )
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

  bool failed;
  //assert( VRField != VR::UN );
  if( VRField & VR::VRASCII )
    {
    //assert( VRField.GetSize() == 1 );
    failed = !ValueIO<ExplicitDataElement,TSwap>::Read(is,*ValueField,readvalues);
    }
  else
    {
    assert( VRField & VR::VRBINARY );
    unsigned int vrsize = VRField.GetSize();
    assert( vrsize == 1 || vrsize == 2 || vrsize == 4 || vrsize == 8 );
    if(VRField==VR::AT) vrsize = 2;
    switch(vrsize)
      {
    case 1:
      failed = !ValueIO<ExplicitImplicitDataElement,TSwap,uint8_t>::Read(is,*ValueField,readvalues);
      break;
    case 2:
      failed = !ValueIO<ExplicitImplicitDataElement,TSwap,uint16_t>::Read(is,*ValueField,readvalues);
      break;
    case 4:
      failed = !ValueIO<ExplicitImplicitDataElement,TSwap,uint32_t>::Read(is,*ValueField,readvalues);
      break;
    case 8:
      failed = !ValueIO<ExplicitImplicitDataElement,TSwap,uint64_t>::Read(is,*ValueField,readvalues);
      break;
    default:
    failed = true;
      assert(0);
      }
    }
  if( failed )
    {
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    if( TagField == Tag(0x7fe0,0x0010) )
      {
      // BUG this should be moved to the ImageReader class, only this class knows
      // what 7fe0 actually is, and should tolerate partial Pixel Data element...
      // PMS-IncompletePixelData.dcm
      gdcmWarningMacro( "Incomplete Pixel Data found, use file at own risk" );
      is.clear();
      }
    else
#endif /* GDCM_SUPPORT_BROKEN_IMPLEMENTATION */
      {
      // Might be the famous UN 16bits
      ParseException pe;
      pe.SetLastElement( *this );
      throw pe;
      }
    return is;
    }
  return is;

}


} // end namespace gdcm

#endif // GDCMEXPLICITIMPLICITDATAELEMENT_TXX
