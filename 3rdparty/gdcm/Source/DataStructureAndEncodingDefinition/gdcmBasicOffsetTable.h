/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMBASICOFFSETTABLE_H
#define GDCMBASICOFFSETTABLE_H

#include "gdcmFragment.h"

namespace gdcm
{
/**
 * \brief Class to represent a BasicOffsetTable
 */

class GDCM_EXPORT BasicOffsetTable : public Fragment
{
//protected:
//  void SetTag(const Tag &t);
public:
  BasicOffsetTable() : Fragment() {}
  friend std::ostream &operator<<(std::ostream &os, const BasicOffsetTable &val);

/*
  VL GetLength() const {
    assert( !ValueLengthField.IsUndefined() );
    assert( !ValueField || ValueField->GetLength() == ValueLengthField );
    return TagField.GetLength() + ValueLengthField.GetLength()
      + ValueLengthField;
  }
*/

  template <typename TSwap>
  std::istream &Read(std::istream &is) {
    // Superclass
    const Tag itemStart(0xfffe, 0xe000);
    const Tag seqDelItem(0xfffe,0xe0dd);
    if( !TagField.Read<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
    //assert( TagField == itemStart );
    if( TagField != itemStart )
      {
      // Bug_Siemens_PrivateIconNoItem.dcm
      gdcmDebugMacro( "Could be Bug_Siemens_PrivateIconNoItem.dcm" );
      throw "SIEMENS Icon thingy";
      }
    if( !ValueLengthField.Read<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
    // Self
    SmartPointer<ByteValue> bv = new ByteValue;
    bv->SetLength(ValueLengthField);
    if( !bv->Read<TSwap>(is) )
      {
      assert(0 && "Should not happen");
      return is;
      }
    ValueField = bv;
    return is;
    }

/*
  template <typename TSwap>
  std::ostream &Write(std::ostream &os) const {
    const Tag itemStart(0xfffe, 0xe000);
    const Tag seqDelItem(0xfffe,0xe0dd);
    if( !TagField.Write<TSwap>(os) )
      {
      assert(0 && "Should not happen");
      return os;
      }
    assert( TagField == itemStart );
    if( !ValueLengthField.Write<TSwap>(os) )
      {
      assert(0 && "Should not happen");
      return os;
      }
    if( ValueLengthField )
      {
      // Self
      const ByteValue *bv = GetByteValue();
      assert( bv );
      assert( bv->GetLength() == ValueLengthField );
      if( !bv->Write<TSwap>(os) )
        {
        assert(0 && "Should not happen");
        return os;
        }
      }
    return os;
    }
*/
};
//-----------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &os, const BasicOffsetTable &val)
{
  os << " BasicOffsetTable Length=" << val.ValueLengthField << std::endl;
  if( val.ValueField )
    {
    const ByteValue *bv = val.GetByteValue();
    assert( bv );
    os << *bv;
    }

  return os;
}


} // end namespace gdcm

#endif //GDCMBASICOFFSETTABLE_H
