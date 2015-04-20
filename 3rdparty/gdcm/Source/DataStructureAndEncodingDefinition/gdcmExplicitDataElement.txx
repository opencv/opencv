/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMEXPLICITDATAELEMENT_TXX
#define GDCMEXPLICITDATAELEMENT_TXX

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
std::istream &ExplicitDataElement::Read(std::istream &is)
{
  ReadPreValue<TSwap>(is);
  return ReadValue<TSwap>(is);
}

template <typename TSwap>
std::istream &ExplicitDataElement::ReadPreValue(std::istream &is)
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
    // Reset ValueLengthField to avoid user error
    ValueLengthField = 0;
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
    (void)ex; //compiler warning
    ParseException pe;
    pe.SetLastElement( *this );
    throw pe;
#else
  throw ex;
#endif /* GDCM_SUPPORT_BROKEN_IMPLEMENTATION */
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
std::istream &ExplicitDataElement::ReadValue(std::istream &is, bool readvalues)
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
    if( TagField == Tag(0x7fe0,0x0010) )
      {
      // Ok this is Pixel Data fragmented...
      assert( VRField & VR::OB_OW || VRField == VR::UN );
      ValueField = new SequenceOfFragments;
      }
    else
      {
      // Support cp246 conforming file:
      // Enhanced_MR_Image_Storage_PixelSpacingNotIn_0028_0030.dcm (illegal)
      // vs
      // undefined_length_un_vr.dcm
      assert( TagField != Tag(0x7fe0,0x0010) );
      assert( VRField == VR::UN );
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
      failed = !ValueIO<ExplicitDataElement,TSwap,uint8_t>::Read(is,*ValueField,readvalues);
      break;
    case 2:
      failed = !ValueIO<ExplicitDataElement,TSwap,uint16_t>::Read(is,*ValueField,readvalues);
      break;
    case 4:
      failed = !ValueIO<ExplicitDataElement,TSwap,uint32_t>::Read(is,*ValueField,readvalues);
      break;
    case 8:
      failed = !ValueIO<ExplicitDataElement,TSwap,uint64_t>::Read(is,*ValueField,readvalues);
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
      // BUG this should be moved to the ImageReader class, only this class
      // knows what 7fe0 actually is, and should tolerate partial Pixel Data
      // element...  PMS-IncompletePixelData.dcm
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

#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
  if( SequenceOfItems *sqi = dynamic_cast<SequenceOfItems*>(&GetValue()) )
    {
    assert( ValueField->GetLength() == ValueLengthField );
    // Recompute the total length:
    if( !ValueLengthField.IsUndefined() )
      {
      // PhilipsInteraSeqTermInvLen.dcm
      // contains an extra seq del item marker, which we are not loading. Therefore the
      // total length needs to be recomputed when sqi is expressed in defined length
      VL dummy = sqi->template ComputeLength<ExplicitDataElement>();
      ValueLengthField = dummy;
      sqi->SetLength( dummy );
      gdcmAssertAlwaysMacro( dummy == ValueLengthField );
      }
    }
  else if( SequenceOfFragments *sqf = dynamic_cast<SequenceOfFragments*>(&GetValue()) )
    {
    assert( ValueField->GetLength() == ValueLengthField );
    assert( sqf->GetLength() == ValueLengthField ); (void)sqf;
    assert( ValueLengthField.IsUndefined() );
    }
#endif

  return is;
}

template <typename TSwap>
std::istream &ExplicitDataElement::ReadWithLength(std::istream &is, VL & length)
{
  return Read<TSwap>(is); (void)length;
}

//-----------------------------------------------------------------------------
template <typename TSwap>
const std::ostream &ExplicitDataElement::Write(std::ostream &os) const
{
  if( TagField == Tag(0xfffe,0xe0dd) ) throw Exception( "Impossible" );
  //if( TagField == Tag(0xfffe,0xe0dd) ) return os;
  if( !TagField.Write<TSwap>(os) )
    {
    assert( 0 && "Should not happen" );
    return os;
    }
  const Tag itemDelItem(0xfffe,0xe00d);
  if( TagField == itemDelItem )
    {
    assert(0);
    assert( ValueField == 0 );
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    if( ValueLengthField != 0 )
      {
      gdcmWarningMacro(
        "Item Delimitation Item had a length different from 0." );
      VL zero = 0;
      zero.Write<TSwap>(os);
      return os;
      }
#endif
    // else
    assert( ValueLengthField == 0 );
    if( !ValueLengthField.Write<TSwap>(os) )
      {
      assert( 0 && "Should not happen" );
      return os;
      }
    return os;
    }
  bool vr16bitsimpossible = (VRField & VR::VL16) && (ValueLengthField > (uint32_t)VL::GetVL16Max());
  if( VRField == VR::INVALID || vr16bitsimpossible )
    {
    if ( TagField.IsPrivateCreator() )
      {
      gdcmAssertAlwaysMacro( !vr16bitsimpossible );
      VR lo = VR::LO;
      if( TagField.IsGroupLength() )
        {
        lo = VR::UL;
        }
      lo.Write(os);
      ValueLengthField.Write16<TSwap>(os);
      }
    else
      {
      const VR un = VR::UN;
      un.Write(os);
      Value* v = &*ValueField;
      if( dynamic_cast<const SequenceOfItems*>(v) )
        {
        VL vl = 0xFFFFFFFF;
        assert( vl.IsUndefined() );
        vl.Write<TSwap>(os);
        }
      else
        ValueLengthField.Write<TSwap>(os);
      }
    }
  else
    {
    assert( VRField.IsVRFile() && VRField != VR::INVALID );
    if( !VRField.Write(os) )
      {
      assert( 0 && "Should not happen" );
      return os;
      }
    if( VRField & VR::VL32 )
      {
      if( !ValueLengthField.Write<TSwap>(os) )
        {
        assert( 0 && "Should not happen" );
        return os;
        }
      }
    else
      {
      // 16bits only
      if( !ValueLengthField.template Write16<TSwap>(os) )
        {
        assert( 0 && "Should not happen" );
        return os;
        }
      }
    }
  if( ValueLengthField )
    {
    // Special case, check SQ
    if ( GetVR() == VR::SQ )
      {
      gdcmAssertAlwaysMacro( dynamic_cast<const SequenceOfItems*>(&GetValue()) );
      }
//#ifndef NDEBUG
    // check consistency in Length:
    if( GetByteValue() )
      {
      assert( ValueField->GetLength() == ValueLengthField );
      }
    //else if( GetSequenceOfItems() )
    else if( const SequenceOfItems *sqi = dynamic_cast<const SequenceOfItems*>(&GetValue()) )
      {
      assert( ValueField->GetLength() == ValueLengthField );
      // Recompute the total length:
      if( !ValueLengthField.IsUndefined() )
        {
        VL dummy = sqi->template ComputeLength<ExplicitDataElement>();
        gdcmAssertAlwaysMacro( dummy == ValueLengthField );
        }
      }
    else if( GetSequenceOfFragments() )
      {
      assert( ValueField->GetLength() == ValueLengthField );
      }
//#endif
    // We have the length we should be able to write the value
    if( VRField == VR::UN && ValueLengthField.IsUndefined() )
      {
      assert( TagField == Tag(0x7fe0,0x0010) || GetValueAsSQ() );
      ValueIO<ImplicitDataElement,TSwap>::Write(os,*ValueField);
      }
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    else if( VRField == VR::INVALID && dynamic_cast<const SequenceOfItems*>(&*ValueField) )
      {
      // We have pretended so far that the Sequence was encoded as UN. Well the real
      // troubles is that we cannot store the length as explicit length, otherwise
      // we will loose the SQ, therefore change the length into undefined length
      // and add a seq del item:
      ValueIO<ImplicitDataElement,TSwap>::Write(os,*ValueField);
      if( !ValueLengthField.IsUndefined() )
        {
        // eg. TestWriter with ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm
        // seq del item is not stored, write it !
        const Tag seqDelItem(0xfffe,0xe0dd);
        seqDelItem.Write<TSwap>(os);
        VL zero = 0;
        zero.Write<TSwap>(os);
        }
      }
#endif
    else
      {
      bool failed;
      if( VRField & VR::VRASCII || VRField == VR::INVALID )
        {
        failed = !ValueIO<ExplicitDataElement,TSwap>::Write(os,*ValueField);
        }
      else
        {
        assert( VRField & VR::VRBINARY );
        unsigned int vrsize = VRField.GetSize();
        assert( vrsize == 1 || vrsize == 2 || vrsize == 4 || vrsize == 8 );
        if(VRField == VR::AT) vrsize = 2;
        switch(vrsize)
          {
        case 1:
          failed = !ValueIO<ExplicitDataElement,TSwap,uint8_t >::Write(os,*ValueField);
          break;
        case 2:
          failed = !ValueIO<ExplicitDataElement,TSwap,uint16_t>::Write(os,*ValueField);
          break;
        case 4:
          failed = !ValueIO<ExplicitDataElement,TSwap,uint32_t>::Write(os,*ValueField);
          break;
        case 8:
          failed = !ValueIO<ExplicitDataElement,TSwap,uint64_t>::Write(os,*ValueField);
          break;
        default:
          failed = true;
          assert(0);
          }
        }
      if( failed )
        {
        assert( 0 && "Should not happen" );
        }
      }
    }

  return os;
}



} // end namespace gdcm

#endif // GDCMEXPLICITDATAELEMENT_TXX
