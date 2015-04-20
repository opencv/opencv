/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMIMPLICITDATAELEMENT_TXX
#define GDCMIMPLICITDATAELEMENT_TXX

#include "gdcmSequenceOfItems.h"
#include "gdcmValueIO.h"
#include "gdcmSwapper.h"
#ifdef GDCM_WORDS_BIGENDIAN
#include "gdcmTagToVR.h"
#endif

namespace gdcm
{

//-----------------------------------------------------------------------------
template <typename TSwap>
std::istream &ImplicitDataElement::Read(std::istream &is)
{
  ReadPreValue<TSwap>(is);
  return ReadValue<TSwap>(is);
}

template <typename TSwap>
std::istream &ImplicitDataElement::ReadPreValue(std::istream& is)
{
  TagField.Read<TSwap>(is);
  // See PS 3.5, 7.1.3 Data Element Structure With Implicit VR
  // Read Tag
  if( !is )
    {
    if( !is.eof() ) // FIXME This should not be needed
      assert(0 && "Should not happen");
    return is;
    }
  const Tag itemStartItem(0xfffe,0xe000);
  if( TagField == itemStartItem ) return is;

  //assert( TagField != Tag(0xfffe,0xe0dd) );
  // Read Value Length
  if( !ValueLengthField.Read<TSwap>(is) )
    {
    //assert(0 && "Should not happen");
    throw Exception("Impossible ValueLengthField");
    return is;
    }
  return is;
}

template <typename TSwap>
std::istream &ImplicitDataElement::ReadValue(std::istream &is, bool readvalues)
{
  if( is.eof() ) return is;
  const Tag itemStartItem(0xfffe,0xe000);
  assert( TagField != itemStartItem );

  /*
   * technically this should not be needed, but what if an implementor, forgot
   * to set VL = 0, then we should make sure to exit early
   */
  const Tag itemDelItem(0xfffe,0xe00d);
  if( TagField == itemDelItem )
    {
    if( ValueLengthField != 0 )
      {
      gdcmWarningMacro( "VL should be set to 0" );
      }
    ValueField = 0;
    return is;
    }
  //std::cerr << "imp cur tag=" << TagField <<  " VL=" << ValueLengthField << std::endl;
  //if( ValueLengthField > length && !ValueLengthField.IsUndefined() )
  //  {
  //  gdcmWarningMacro( "Cannot read more length than what is remaining in the file" );
  //  throw Exception( "Impossible" );
  //  }
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
      gdcmErrorMacro( "Undefined value length is impossible in non-encapsulated Transfer Syntax. Proceeding with caution" );
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
        gdcmWarningMacro( "Illegal Tag for Item starter: " << TagField << " should be: " << itemStart );
        // TODO: We READ Explicit ok...but we store Implicit !
        // Indeed when copying the VR will be saved... pretty cool eh ?
        ValueField = new SequenceOfItems;
        ValueField->SetLength(ValueLengthField); // perform realloc
        std::streampos start = is.tellg();
        try
          {
          if( !ValueIO<ExplicitDataElement,SwapperDoOp>::Read(is,*ValueField,readvalues) )
            {
            assert(0 && "Should not happen");
            }
          gdcmWarningMacro( "Illegal: Explicit SQ found in a file with "
            "TransferSyntax=Implicit for tag: " << TagField );
          }
        catch( Exception & )
          {
          // MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm
          std::streampos current = is.tellg();
          std::streamoff diff = start - current;
          is.seekg( diff, std::ios::cur );
          assert( diff == -14 );
          ValueIO<ImplicitDataElement,SwapperDoOp>::Read(is,*ValueField,readvalues);
          }
        catch( std::exception & )
          {
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
        if( !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,readvalues) )
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
    gdcmWarningMacro( "Replacing a VL. To be able to read a supposively "
      "broken Payrus file." );
    ValueLengthField = 202; // 0xca
    }
#endif
  // We have the length we should be able to read the value
  this->SetValueFieldLength( ValueLengthField, readvalues );
  bool failed;
#ifdef GDCM_WORDS_BIGENDIAN
  VR vrfield = GetVRFromTag( TagField );
  if( vrfield & VR::VRASCII || vrfield == VR::INVALID )
    {
    //assert( VRField.GetSize() == 1 );
    failed = !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,readvalues);
    }
  else
    {
    assert( vrfield & VR::VRBINARY );
    unsigned int vrsize = vrfield.GetSize();
    assert( vrsize == 1 || vrsize == 2 || vrsize == 4 || vrsize == 8 );
    if(vrfield==VR::AT) vrsize = 2;
    switch(vrsize)
      {
    case 1:
      failed = !ValueIO<ImplicitDataElement,TSwap,uint8_t>::Read(is,*ValueField,readvalues);
      break;
    case 2:
      failed = !ValueIO<ImplicitDataElement,TSwap,uint16_t>::Read(is,*ValueField,readvalues);
      break;
    case 4:
      failed = !ValueIO<ImplicitDataElement,TSwap,uint32_t>::Read(is,*ValueField,readvalues);
      break;
    case 8:
      failed = !ValueIO<ImplicitDataElement,TSwap,uint64_t>::Read(is,*ValueField,readvalues);
      break;
    default:
    failed = true;
      assert(0);
      }
    }
#else
  failed = !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,readvalues);
#endif
  if( failed )
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
  // dcmtk 3.5.4 is resilient to broken explicit SQ length and will properly
  // recompute it as long as each of the Item lengths are correct
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

//-----------------------------------------------------------------------------
template <typename TSwap>
std::istream &ImplicitDataElement::ReadWithLength(std::istream &is, VL & length, bool readvalues)
{
  ReadPreValue<TSwap>(is);
  return ReadValueWithLength<TSwap>(is, length, readvalues);
}

template <typename TSwap>
std::istream &ImplicitDataElement::ReadValueWithLength(std::istream& is, VL & length, bool readvalues)
{
  if( is.eof() ) return is;
  const Tag itemStartItem(0xfffe,0xe000);
  if( TagField == itemStartItem ) return is;

  /*
   * technically this should not be needed, but what if an implementor, forgot
   * to set VL = 0, then we should make sure to exit early
   */
  const Tag itemDelItem(0xfffe,0xe00d);
  if( TagField == itemDelItem )
    {
    if( ValueLengthField != 0 )
      {
      gdcmWarningMacro( "VL should be set to 0" );
      }
    ValueField = 0;
    return is;
    }
  //std::cerr << "imp cur tag=" << TagField <<  " VL=" << ValueLengthField << std::endl;
  if( ValueLengthField > length && !ValueLengthField.IsUndefined() )
    {
    gdcmWarningMacro( "Cannot read more length than what is remaining in the file" );
    throw Exception( "Impossible (more)" );
    }
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
      gdcmErrorMacro( "Undefined value length is impossible in non-encapsulated Transfer Syntax. Proceeding with caution" );
      ValueField = new SequenceOfFragments;
      }
    //VRField = VR::SQ;
    }
  else
    {
    if( true /*ValueLengthField < 8*/ )
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
        gdcmWarningMacro( "Illegal Tag for Item starter: " << TagField << " should be: " << itemStart );
        // TODO: We READ Explicit ok...but we store Implicit !
        // Indeed when copying the VR will be saved... pretty cool eh ?
        ValueField = new SequenceOfItems;
        ValueField->SetLength(ValueLengthField); // perform realloc
        std::streampos start = is.tellg();
        try
          {
          if( !ValueIO<ExplicitDataElement,SwapperDoOp>::Read(is,*ValueField,readvalues) )
            {
            assert(0 && "Should not happen");
            }
          gdcmWarningMacro( "Illegal: Explicit SQ found in a file with "
            "TransferSyntax=Implicit for tag: " << TagField );
          }
        catch( Exception &)
          {
          // MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm
          std::streampos current = is.tellg();
          std::streamoff diff = start - current;//could be bad, if the specific implementation does not support negative streamoff values.
          is.seekg( diff, std::ios::cur );
          assert( diff == -14 );
          ValueIO<ImplicitDataElement,SwapperDoOp>::Read(is,*ValueField,readvalues);
          }
        catch( std::exception & )
          {
          ValueLengthField = ValueField->GetLength();
          }
        return is;
        }
      else if ( item == itemPMSStart2 )
        {
        assert( 0 ); // FIXME: Sync Read/ReadWithLength
        gdcmWarningMacro( "Illegal: SQ start with " << itemPMSStart2
          << " instead of " << itemStart << " for tag: " << TagField );
        ValueField = new SequenceOfItems;
        ValueField->SetLength(ValueLengthField); // perform realloc
        if( !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,readvalues) )
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
  bool failed;
#ifdef GDCM_WORDS_BIGENDIAN
  VR vrfield = GetVRFromTag( TagField );
  if( vrfield & VR::VRASCII || vrfield == VR::INVALID )
    {
    //assert( VRField.GetSize() == 1 );
    failed = !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,readvalues);
    }
  else
    {
    assert( vrfield & VR::VRBINARY );
    unsigned int vrsize = vrfield.GetSize();
    assert( vrsize == 1 || vrsize == 2 || vrsize == 4 || vrsize == 8 );
    if(vrfield==VR::AT) vrsize = 2;
    switch(vrsize)
      {
    case 1:
      failed = !ValueIO<ImplicitDataElement,TSwap,uint8_t>::Read(is,*ValueField,readvalues);
      break;
    case 2:
      failed = !ValueIO<ImplicitDataElement,TSwap,uint16_t>::Read(is,*ValueField,readvalues);
      break;
    case 4:
      failed = !ValueIO<ImplicitDataElement,TSwap,uint32_t>::Read(is,*ValueField,readvalues);
      break;
    case 8:
      failed = !ValueIO<ImplicitDataElement,TSwap,uint64_t>::Read(is,*ValueField,readvalues);
      break;
    default:
    failed = true;
      assert(0);
      }
    }
#else
  failed = !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField,readvalues);
#endif
  if( failed )
  //if( !ValueIO<ImplicitDataElement,TSwap>::Read(is,*ValueField) )
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
    gdcmWarningMacro( "ValueLengthField was bogus" );
    ValueLengthField = dummy;
    }
#else
  assert( ValueLengthField == ValueField->GetLength() );
  assert( VRField == VR::INVALID );
#endif

  return is;
}

//-----------------------------------------------------------------------------
template <typename TSwap>
const std::ostream &ImplicitDataElement::Write(std::ostream &os) const
{
  // See PS 3.5, 7.1.3 Data Element Structure With Implicit VR
  // Write Tag
  if( !TagField.Write<TSwap>(os) )
    {
    assert(0 && "Should not happen");
    return os;
    }
  // Write Value Length
  const SequenceOfItems *sqi = dynamic_cast<const SequenceOfItems*>(&GetValue()); //GetSequenceOfItems();
  if( sqi && !ValueLengthField.IsUndefined() )
    {
    // Hum, we might have to recompute the length:
    // See TestWriter2, where an explicit SQ is converted to implicit SQ
    VL len = sqi->template ComputeLength<ImplicitDataElement>();
    //assert( len == ValueLengthField );
    if( !len.Write<TSwap>(os) )
      {
      assert(0 && "Should not happen");
      return os;
      }
    }
  else // It should be safe to simply use the ValueLengthField as stored:
    {
    // Do not allow writing file such as: dcm4che_UndefinedValueLengthInImplicitTS.dcm
    if( TagField == Tag(0x7fe0,0x0010) && ValueLengthField.IsUndefined() ) throw Exception( "VL u/f Impossible" );
    if( !ValueLengthField.Write<TSwap>(os) )
      {
      assert(0 && "Should not happen");
      return os;
      }
    }
  // Write Value
  if( ValueLengthField )
    {
    assert( ValueField );
    gdcmAssertAlwaysMacro( ValueLengthField == ValueField->GetLength() );
    assert( TagField != Tag(0xfffe, 0xe00d)
         && TagField != Tag(0xfffe, 0xe0dd) );
    if( !ValueIO<ImplicitDataElement,TSwap>::Write(os,*ValueField) )
      {
      assert(0 && "Should not happen");
      return os;
      }
    }
  return os;
}


} // end namespace gdcm


#endif // GDCMIMPLICITDATAELEMENT_TXX
