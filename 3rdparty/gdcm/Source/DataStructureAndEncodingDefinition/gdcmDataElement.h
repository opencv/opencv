/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDATAELEMENT_H
#define GDCMDATAELEMENT_H

#include "gdcmTag.h"
#include "gdcmVL.h"
#include "gdcmVR.h"
#include "gdcmByteValue.h"
#include "gdcmSmartPointer.h"

#include <set>

namespace gdcm
{
// Data Element
// Contains multiple fields:
// -> Tag
// -> Optional VR (Explicit Transfer Syntax)
// -> ValueLength
// -> Value
// TODO: This class SHOULD be pure virtual. I don't want a user
// to shoot himself in the foot.

class SequenceOfItems;
class SequenceOfFragments;
/**
 * \brief Class to represent a Data Element
 * either Implicit or Explicit
 *
 * \details
 * DATA ELEMENT:
 * A unit of information as defined by a single entry in the data dictionary.
 * An encoded Information Object Definition (IOD) Attribute that is composed
 * of, at a minimum, three fields: a Data Element Tag, a Value Length, and a
 * Value Field. For some specific Transfer Syntaxes, a Data Element also
 * contains a VR Field where the Value Representation of that Data Element is
 * specified explicitly.
 *
 * Design:
 * \li A DataElement in GDCM always store VL (Value Length) on a 32 bits integer even when VL is 16 bits
 * \li A DataElement always store the VR even for Implicit TS, in which case VR is defaulted to VR::INVALID
 * \li For Item start/end (See 0xfffe tags), Value is NULL
 *
 * \see ExplicitDataElement ImplicitDataElement
 */
class GDCM_EXPORT DataElement
{
public:
  DataElement(const Tag& t = Tag(0), const VL& vl = 0, const VR &vr = VR::INVALID):TagField(t),ValueLengthField(vl),VRField(vr),ValueField(0) {}
  //DataElement( Attribute const &att );

  friend std::ostream& operator<<(std::ostream &_os, const DataElement &_val);

  /// Get Tag
  const Tag& GetTag() const { return TagField; }
  Tag& GetTag() { return TagField; }
  /// Set Tag
  /// Use with cautious (need to match Part 6)
  void SetTag(const Tag &t) { TagField = t; }

  /// Get VL
  const VL& GetVL() const { return ValueLengthField; }
  VL& GetVL() { return ValueLengthField; }
  /// Set VL
  /// Use with cautious (need to match Part 6), advanced user only
  /// \see SetByteValue
  void SetVL(const VL &vl) { ValueLengthField = vl; }
  void SetVLToUndefined();

  /// Get VR
  /// do not set VR::SQ on bytevalue data element
  VR const &GetVR() const { return VRField; }
  /// Set VR
  /// Use with cautious (need to match Part 6), advanced user only
  /// \pre vr is a VR::VRALL (not a dual one such as OB_OW)
  void SetVR(VR const &vr) {
    if( vr.IsVRFile() )
      VRField = vr;
  }

  /// Set/Get Value (bytes array, SQ of items, SQ of fragments):
  Value const &GetValue() const { return *ValueField; }
  Value &GetValue() { return *ValueField; }
  /// \warning you need to set the ValueLengthField explicitely
  void SetValue(Value const & vl) {
    //assert( ValueField == 0 );
    ValueField = vl;
    ValueLengthField = vl.GetLength();
  }
  /// Check if Data Element is empty
  bool IsEmpty() const { return ValueField == 0 || (GetByteValue() && GetByteValue()->IsEmpty()); }

  /// Make Data Element empty (no Value)
  void Empty() { ValueField = 0; ValueLengthField = 0; }

  /// Clear Data Element (make Value empty and invalidate Tag & VR)
  void Clear()
    {
    TagField = 0;
    VRField = VR::INVALID;
    ValueField = 0;
    ValueLengthField = 0;
    }

  // Helper:
  /// Set the byte value
  /// \warning user need to read DICOM standard for an understanding of:
  /// * even padding
  /// * \0 vs space padding
  /// By default even padding is achieved using \0 regardless of the of VR
  void SetByteValue(const char *array, VL length)
    {
    ByteValue *bv = new ByteValue(array,length);
    SetValue( *bv );
    }
  /// Return the Value of DataElement as a ByteValue (if possible)
  /// \warning: You need to check for NULL return value
  const ByteValue* GetByteValue() const {
    // Get the raw pointer from the gdcm::SmartPointer
    const ByteValue *bv = dynamic_cast<const ByteValue*>(ValueField.GetPointer());
    return bv; // Will return NULL if not ByteValue
  }

  /// Interpret the Value stored in the DataElement. This is more robust (but also more
  /// expensive) to call this function rather than the simpliest form: GetSequenceOfItems()
  /// It also return NULL when the Value is NOT of type SequenceOfItems
  /// \warning in case GetSequenceOfItems() succeed the function return this value, otherwise
  /// it creates a new SequenceOfItems, you should handle that in your case, for instance:
  /// SmartPointer<SequenceOfItems> sqi = de.GetValueAsSQ();
  SmartPointer<SequenceOfItems> GetValueAsSQ() const;

  /// Return the Value of DataElement as a Sequence Of Fragments (if possible)
  /// \warning: You need to check for NULL return value
  const SequenceOfFragments* GetSequenceOfFragments() const;
  SequenceOfFragments* GetSequenceOfFragments();

  /// return if Value Length if of undefined length
  bool IsUndefinedLength() const {
    return ValueLengthField.IsUndefined();
  }

  DataElement(const DataElement &_val)
    {
    if( this != &_val)
      {
      *this = _val;
      }
    }

  bool operator<(const DataElement &de) const
    {
    return GetTag() < de.GetTag();
    }
  DataElement &operator=(const DataElement &de)
    {
    TagField = de.TagField;
    ValueLengthField = de.ValueLengthField;
    VRField = de.VRField;
    ValueField = de.ValueField; // Pointer copy
    return *this;
    }

  bool operator==(const DataElement &de) const
    {
    bool b = TagField == de.TagField
      && ValueLengthField == de.ValueLengthField
      && VRField == de.VRField;
    if( !ValueField && !de.ValueField )
      {
      return b;
      }
    if( ValueField && de.ValueField )
      {
      return b && (*ValueField == *de.ValueField);
      }
    // ValueField != de.ValueField
    return false;
    }

  // The following fonctionalities are dependant on:
  // # The Transfer Syntax: Explicit or Implicit
  // # The Byte encoding: Little Endian / Big Endian

  /*
   * The following was inspired by a C++ idiom: Curiously Recurring Template Pattern
   * Ref: http://en.wikipedia.org/wiki/Curiously_Recurring_Template_Pattern
   * The typename TDE is typically a derived class *without* any data
   * while TSwap is a simple template parameter to achieve byteswapping (and allow factorization of
   * highly identical code)
   */
  template <typename TDE>
  VL GetLength() const {
    return static_cast<const TDE*>(this)->GetLength();
  }

  template <typename TDE, typename TSwap>
  std::istream &Read(std::istream &is) {
    return static_cast<TDE*>(this)->template Read<TSwap>(is);
  }

  template <typename TDE, typename TSwap>
  std::istream &ReadOrSkip(std::istream &is, std::set<Tag> const &skiptags) {
    (void)skiptags;
    return static_cast<TDE*>(this)->template Read<TSwap>(is);
  }

  template <typename TDE, typename TSwap>
  std::istream &ReadPreValue(std::istream &is, std::set<Tag> const &skiptags) {
    (void)skiptags;
    return static_cast<TDE*>(this)->template ReadPreValue<TSwap>(is);
  }
  template <typename TDE, typename TSwap>
  std::istream &ReadValue(std::istream &is, std::set<Tag> const &skiptags) {
    (void)skiptags;
    return static_cast<TDE*>(this)->template ReadValue<TSwap>(is);
  }
  template <typename TDE, typename TSwap>
  std::istream &ReadValueWithLength(std::istream &is, VL & length, std::set<Tag> const &skiptags) {
    (void)skiptags;
    return static_cast<TDE*>(this)->template ReadValueWithLength<TSwap>(is, length);
  }

  template <typename TDE, typename TSwap>
  std::istream &ReadWithLength(std::istream &is, VL &length) {
    return static_cast<TDE*>(this)->template ReadWithLength<TSwap>(is,length);
  }

  template <typename TDE, typename TSwap>
  const std::ostream &Write(std::ostream &os) const {
    return static_cast<const TDE*>(this)->template Write<TSwap>(os);
  }

protected:
  Tag TagField;
  // This is the value read from the file, might be different from the length of Value Field
  VL ValueLengthField; // Can be 0xFFFFFFFF

  // Value Representation
  VR VRField;
  typedef SmartPointer<Value> ValuePtr;
  ValuePtr ValueField;

  void SetValueFieldLength( VL vl, bool readvalues );
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const DataElement &val)
{
  os << val.TagField;
  os << "\t" << val.VRField;
  os << "\t" << val.ValueLengthField;
  if( val.ValueField )
    {
    val.ValueField->Print( os << "\t" );
    }
  return os;
}

inline bool operator!=(const DataElement& lhs, const DataElement& rhs)
{
  return ! ( lhs == rhs );
}

} // end namespace gdcm

#endif //GDCMDATAELEMENT_H
