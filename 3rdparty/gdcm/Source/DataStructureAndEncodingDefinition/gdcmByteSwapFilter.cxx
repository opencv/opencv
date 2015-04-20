/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmByteSwapFilter.h"

#include "gdcmElement.h"
#include "gdcmByteValue.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmSwapper.h"

namespace gdcm
{

//-----------------------------------------------------------------------------
//ByteSwapFilter::ByteSwapFilter()
//{
//}
//-----------------------------------------------------------------------------
ByteSwapFilter::~ByteSwapFilter()
{
}

bool ByteSwapFilter::ByteSwap()
{
  for(
    DataSet::ConstIterator it = DS.Begin();
    it != DS.End(); ++it)
    {
    const DataElement &de = *it;
    VR const & vr = de.GetVR();
    //assert( vr & VR::VRASCII || vr & VR::VRBINARY );
    const ByteValue *bv = de.GetByteValue();
    gdcm::SmartPointer<gdcm::SequenceOfItems> si = de.GetValueAsSQ();
    if( de.IsEmpty() )
      {
      }
    else if( bv && !si )
      {
      assert( !si );
      // ASCII do not need byte swap
      if( vr & VR::VRBINARY /*&& de.GetTag().IsPrivate()*/ )
        {
        //assert( de.GetTag().IsPrivate() );
        switch(vr)
          {
        case VR::AT:
          assert( 0 && "Should not happen" );
          break;
        case VR::FL:
          // FIXME: Technically FL should not be byte-swapped...
          //std::cerr << "ByteSwap FL:" << de.GetTag() << std::endl;
          SwapperDoOp::SwapArray((uint32_t*)bv->GetPointer(), bv->GetLength() / sizeof(uint32_t) );
          break;
        case VR::FD:
          assert( 0 && "Should not happen" );
          break;
        case VR::OB:
          // I think we are fine, unless this is one of those OB_OW thingy
          break;
        case VR::OF:
          assert( 0 && "Should not happen" );
          break;
        case VR::OW:
          assert( 0 && "Should not happen" );
          break;
        case VR::SL:
          SwapperDoOp::SwapArray((uint32_t*)bv->GetPointer(), bv->GetLength() / sizeof(uint32_t) );
          break;
        case VR::SQ:
          assert( 0 && "Should not happen" );
          break;
        case VR::SS:
          SwapperDoOp::SwapArray((uint16_t*)bv->GetPointer(), bv->GetLength() / sizeof(uint16_t) );
          break;
        case VR::UL:
          SwapperDoOp::SwapArray((uint32_t*)bv->GetPointer(), bv->GetLength() / sizeof(uint32_t) );
          break;
        case VR::UN:
          assert( 0 && "Should not happen" );
          break;
        case VR::US:
          SwapperDoOp::SwapArray((uint16_t*)bv->GetPointer(), bv->GetLength() / sizeof(uint16_t) );
          break;
        case VR::UT:
          assert( 0 && "Should not happen" );
          break;
        default:
          assert( 0 && "Should not happen" );
          }
        }
      }
    //else if( const SequenceOfItems *si = de.GetSequenceOfItems() )
    else if( si )
      {
      //if( de.GetTag().IsPrivate() )
        {
        //std::cerr << "ByteSwap SQ:" << de.GetTag() << std::endl;
        SequenceOfItems::ConstIterator it2 = si->Begin();
        for( ; it2 != si->End(); ++it2)
          {
          const Item &item = *it2;
          DataSet &ds = const_cast<DataSet&>(item.GetNestedDataSet()); // FIXME
          ByteSwapFilter bsf(ds);
          bsf.ByteSwap();
          }
        }
      }
    else if( const SequenceOfFragments *sf = de.GetSequenceOfFragments() )
      {
      (void)sf;
      assert( 0 && "Should not happen" );
      }
    else
      {
      assert( 0 && "error" );
      }

    }
  if( ByteSwapTag )
    {
    DataSet copy;
    DataSet::ConstIterator it = DS.Begin();
    for( ; it != DS.End(); ++it)
      {
      DataElement de = *it;
      const Tag& tag = de.GetTag();
      de.SetTag(
        Tag( SwapperDoOp::Swap( tag.GetGroup() ), SwapperDoOp::Swap( tag.GetElement() ) ) );
      copy.Insert( de );
      DS.Remove( de.GetTag() );
      }
    DS = copy;
    }

  return true;
}

}
