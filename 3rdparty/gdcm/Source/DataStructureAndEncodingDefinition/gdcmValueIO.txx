/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMVALUEIO_TXX
#define GDCMVALUEIO_TXX

#include "gdcmValueIO.h"

#include "gdcmExplicitDataElement.h"
#include "gdcmImplicitDataElement.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmByteValue.h"

namespace gdcm
{

  template <typename TDE, typename TSwap, typename TType>
  std::istream &ValueIO<TDE,TSwap,TType>::Read(std::istream &is, Value& _v, bool readvalues) {
    Value* v = &_v;
    if( ByteValue *bv = dynamic_cast<ByteValue*>(v) )
      {
      bv->template Read<TSwap,TType>(is,readvalues);
      }
    else if( SequenceOfItems *si = dynamic_cast<SequenceOfItems*>(v) )
      {
      si->template Read<TDE,TSwap>(is,readvalues);
      }
    else if( SequenceOfFragments *sf = dynamic_cast<SequenceOfFragments*>(v) )
      {
      sf->template Read<TSwap>(is,readvalues);
      }
    else
      {
      assert( 0 && "error" );
      }
    return is;
  }

  template <typename DE, typename TSwap, typename TType>
  const std::ostream &ValueIO<DE,TSwap,TType>::Write(std::ostream &os, const Value& _v) {
    const Value* v = &_v;
    if( const ByteValue *bv = dynamic_cast<const ByteValue*>(v) )
      {
      bv->template Write<TSwap,TType>(os);
      }
    else if( const SequenceOfItems *si = dynamic_cast<const SequenceOfItems*>(v) )
      {
      //VL dummy = si->ComputeLength<DE>();
      //assert( /*dummy.IsUndefined() ||*/ dummy == si->GetLength() );
      si->template Write<DE,TSwap>(os);
      }
    else if( const SequenceOfFragments *sf = dynamic_cast<const SequenceOfFragments*>(v) )
      {
      sf->template Write<TSwap>(os);
      }
    else
      {
      assert( 0 && "error" );
      }
    return os;
  }

} // end namespace gdcm

#endif // GDCMVALUEIO_TXX
