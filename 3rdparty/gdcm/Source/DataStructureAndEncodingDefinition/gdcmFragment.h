/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFRAGMENT_H
#define GDCMFRAGMENT_H

#include "gdcmDataElement.h"
#include "gdcmByteValue.h"
#include "gdcmSmartPointer.h"
#include "gdcmParseException.h"

namespace gdcm
{

// Implementation detail:
// I think Fragment should be a protected subclass of DataElement:
// looking somewhat like this:
/*
class GDCM_EXPORT Fragment : protected DataElement
{
public:
  using DataElement::GetTag;
  using DataElement::GetVL;
  using DataElement::SetByteValue;
  using DataElement::GetByteValue;
  using DataElement::GetValue;
*/
// Instead I am only hiding the SetTag member...

/**
 * \brief Class to represent a Fragment
 */
class GDCM_EXPORT Fragment : public DataElement
{
//protected:
//  void SetTag(const Tag &t);
public:
  Fragment() : DataElement(Tag(0xfffe, 0xe000), 0) {}
  friend std::ostream &operator<<(std::ostream &os, const Fragment &val);

  VL GetLength() const;

  VL ComputeLength() const;

  template <typename TSwap>
  std::istream &Read(std::istream &is)
    {
    ReadPreValue<TSwap>(is);
    return ReadValue<TSwap>(is);
    }

  template <typename TSwap>
  std::istream &ReadPreValue(std::istream &is)
    {
    const Tag itemStart(0xfffe, 0xe000);
    const Tag seqDelItem(0xfffe,0xe0dd);

    TagField.Read<TSwap>(is);
    if( !is )
      {
      //  BogusItemStartItemEnd.dcm
      throw Exception( "Problem #1" );
      return is;
      }
    if( !ValueLengthField.Read<TSwap>(is) )
      {
      // GENESIS_SIGNA-JPEG-CorruptFrag.dcm
      // JPEG fragment is declared to have 61902, but infact really is only 61901
      // so we end up reading 0xddff,0x00e0, and VL = 0x0 (1 byte)
      throw Exception( "Problem #2" );
      return is;
      }
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    if( TagField != itemStart && TagField != seqDelItem )
      {
      throw Exception( "Problem #3" );
      }
#endif
    return is;
    }

  template <typename TSwap>
  std::istream &ReadValue(std::istream &is)
    {
    // Superclass
    const Tag itemStart(0xfffe, 0xe000);
    const Tag seqDelItem(0xfffe,0xe0dd);
    // Self
    SmartPointer<ByteValue> bv = new ByteValue;
    bv->SetLength(ValueLengthField);
    if( !bv->Read<TSwap>(is) )
      {
      // Fragment is incomplete, but is a itemStart, let's try to push it anyway...
      gdcmWarningMacro( "Fragment could not be read" );
      //bv->SetLength(is.gcount());
      ValueField = bv;
      ParseException pe;
      pe.SetLastElement( *this );
      throw pe;
      return is;
      }
    ValueField = bv;
    return is;
    }

  template <typename TSwap>
  std::istream &ReadBacktrack(std::istream &is)
    {
    const Tag itemStart(0xfffe, 0xe000);
    const Tag seqDelItem(0xfffe,0xe0dd);

    bool cont = true;
    const std::streampos start = is.tellg();
    const int max = 10;
    int offset = 0;
    while( cont )
      {
      TagField.Read<TSwap>(is);
      assert( is );
      if( TagField != itemStart && TagField != seqDelItem )
        {
        ++offset;
        is.seekg( (std::streampos)((size_t)start - offset) );
        gdcmWarningMacro( "Fuzzy Search, backtrack: " << (start - is.tellg()) << " Offset: " << is.tellg() );
        if( offset > max )
          {
          gdcmErrorMacro( "Giving up" );
          throw "Impossible to backtrack";
          return is;
          }
        }
      else
        {
        cont = false;
        }
      }
    assert( TagField == itemStart || TagField == seqDelItem );
    if( !ValueLengthField.Read<TSwap>(is) )
      {
      return is;
      }

    // Self
    SmartPointer<ByteValue> bv = new ByteValue;
    bv->SetLength(ValueLengthField);
    if( !bv->Read<TSwap>(is) )
      {
      // Fragment is incomplete, but is a itemStart, let's try to push it anyway...
      gdcmWarningMacro( "Fragment could not be read" );
      //bv->SetLength(is.gcount());
      ValueField = bv;
      ParseException pe;
      pe.SetLastElement( *this );
      throw pe;
      return is;
      }
    ValueField = bv;
    return is;
    }


  template <typename TSwap>
  std::ostream &Write(std::ostream &os) const {
    const Tag itemStart(0xfffe, 0xe000);
    const Tag seqDelItem(0xfffe,0xe0dd);
    if( !TagField.Write<TSwap>(os) )
      {
      assert(0 && "Should not happen");
      return os;
      }
    assert( TagField == itemStart
         || TagField == seqDelItem );
    const ByteValue *bv = GetByteValue();
    // VL
    // The following piece of code is hard to read in order to support such broken file as:
    // CompressedLossy.dcm
    if( IsEmpty() )
      {
      //assert( bv );
      VL zero = 0;
      if( !zero.Write<TSwap>(os) )
        {
        assert(0 && "Should not happen");
        return os;
        }
      }
    else
      {
      assert( ValueLengthField );
      assert( !ValueLengthField.IsUndefined() );
      const VL actuallen = bv->ComputeLength();
      assert( actuallen == ValueLengthField || actuallen == ValueLengthField + 1 );
      if( !actuallen.Write<TSwap>(os) )
        {
        assert(0 && "Should not happen");
        return os;
        }
      }
    // Value
    if( ValueLengthField && bv )
      {
      // Self
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
};
//-----------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &os, const Fragment &val)
{
  os << "Tag: " << val.TagField;
  os << "\tVL: " << val.ValueLengthField;
  if( val.ValueField )
    {
    os << "\t" << *(val.ValueField);
    }

  return os;
}

} // end namespace gdcm

#endif //GDCMFRAGMENT_H
