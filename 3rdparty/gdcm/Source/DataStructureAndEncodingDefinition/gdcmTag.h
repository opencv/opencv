/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTAG_H
#define GDCMTAG_H

#include "gdcmTypes.h"

#include <iostream>
#include <iomanip>

namespace gdcm
{

/**
 * \brief Class to represent a DICOM Data Element (Attribute) Tag (Group, Element).
 * Basically an uint32_t which can also be expressed as two uint16_t (group and
 * element)
 * \note
 * DATA ELEMENT TAG:
 * A unique identifier for a Data Element composed of an ordered pair of
 * numbers (a Group Number followed by an Element Number).  GROUP NUMBER: The
 * first number in the ordered pair of numbers that makes up a Data Element
 * Tag.
 * ELEMENT NUMBER: The second number in the ordered pair of numbers that
 * makes up a Data Element Tag.
 */
class GDCM_EXPORT Tag
{
public:
  /// \brief Constructor with 2*uint16_t
  Tag(uint16_t group, uint16_t element) {
    ElementTag.tags[0] = group; ElementTag.tags[1] = element;
  }
  /// \brief Constructor with 1*uint32_t
  /// Prefer the cstor that takes two uint16_t
  Tag(uint32_t tag = 0) {
    SetElementTag(tag);
  }

  friend std::ostream& operator<<(std::ostream &_os, const Tag &_val);
  friend std::istream& operator>>(std::istream &_is, Tag &_val);

  /// \brief Returns the 'Group number' of the given Tag
  uint16_t GetGroup() const { return ElementTag.tags[0]; }
  /// \brief Returns the 'Element number' of the given Tag
  uint16_t GetElement() const { return ElementTag.tags[1]; }
  /// \brief Sets the 'Group number' of the given Tag
  void SetGroup(uint16_t group) { ElementTag.tags[0] = group; }
  /// \brief Sets the 'Element number' of the given Tag
  void SetElement(uint16_t element) { ElementTag.tags[1] = element; }
  /// \brief Sets the 'Group number' & 'Element number' of the given Tag
  void SetElementTag(uint16_t group, uint16_t element) {
    ElementTag.tags[0] = group; ElementTag.tags[1] = element;
  }

  /// \brief Returns the full tag value of the given Tag
  uint32_t GetElementTag() const {
#ifndef GDCM_WORDS_BIGENDIAN
    return (ElementTag.tag<<16) | (ElementTag.tag>>16);
#else
    return ElementTag.tag;
#endif
  }

  /// \brief Sets the full tag value of the given Tag
  void SetElementTag(uint32_t tag) {
#ifndef GDCM_WORDS_BIGENDIAN
    tag = ( (tag<<16) | (tag>>16) );
#endif
    ElementTag.tag = tag;
  }

  /// Returns the Group or Element of the given Tag, depending on id (0/1)
  const uint16_t &operator[](const unsigned int &_id) const
    {
    assert(_id<2);
    return ElementTag.tags[_id];
    }
  /// Returns the Group or Element of the given Tag, depending on id (0/1)
  uint16_t &operator[](const unsigned int &_id)
    {
    assert(_id<2);
    return ElementTag.tags[_id];
    }

  Tag &operator=(const Tag &_val)
    {
    ElementTag.tag = _val.ElementTag.tag;
    return *this;
    }

  bool operator==(const Tag &_val) const
    {
    return ElementTag.tag == _val.ElementTag.tag;
    }
  bool operator!=(const Tag &_val) const
    {
    return ElementTag.tag != _val.ElementTag.tag;
    }

  /// DICOM Standard expects the Data Element to be sorted by Tags
  /// All other comparison can be constructed from this one and operator ==
  // FIXME FIXME FIXME TODO
  // the following is pretty dumb. Since we have control over who is group
  // and who is element, we should reverse them in little endian and big endian case
  // since what we really want is fast comparison and not garantee that group is in #0
  // ...
  bool operator<(const Tag &_val) const
    {
#ifndef GDCM_WORDS_BIGENDIAN
    if( ElementTag.tags[0] < _val.ElementTag.tags[0] )
      return true;
    if( ElementTag.tags[0] == _val.ElementTag.tags[0]
      && ElementTag.tags[1] <  _val.ElementTag.tags[1] )
      return true;
    return false;
#else
    // Plain comparison is enough!
    return ( ElementTag.tag < _val.ElementTag.tag );
#endif
    }
  bool operator<=(const Tag &t2) const
    {
    const Tag &t1 = *this;
    return t1 == t2 || t1 < t2;
    }

  Tag(const Tag &_val)
    {
    ElementTag.tag = _val.ElementTag.tag;
    }
  /// return the length of tag (read: size on disk)
  uint32_t GetLength() const { return 4; }

  /// STANDARD DATA ELEMENT: A Data Element defined in the DICOM Standard,
  /// and therefore listed in the DICOM Data Element Dictionary in PS 3.6.
  /// Is the Tag from the Public dict...well the implementation is buggy
  /// it does not prove the element is indeed in the dict...
  bool IsPublic() const { return !(ElementTag.tags[0] % 2); }

  /// PRIVATE DATA ELEMENT: Additional Data Element, defined by an
  /// implementor, to communicate information that is not contained in
  /// Standard Data Elements. Private Data elements have odd Group Numbers.
  bool IsPrivate() const { return !IsPublic(); }

  //-----------------------------------------------------------------------------
  /// Read a tag from binary representation
  template <typename TSwap>
  std::istream &Read(std::istream &is)
    {
    if( is.read(ElementTag.bytes, 4) )
      TSwap::SwapArray(ElementTag.tags, 2);
    return is;
    }

  /// Write a tag in binary rep
  template <typename TSwap>
  const std::ostream &Write(std::ostream &os) const
    {
    uint16_t copy[2];
    copy[0]= ElementTag.tags[0];
    copy[1]= ElementTag.tags[1];
    TSwap::SwapArray(copy, 2);
    return os.write((char*)(&copy), 4);
    }

  /// Return the Private Creator Data Element tag of a private data element
  Tag GetPrivateCreator() const
    {
    // See PS 3.5 - 7.8.1 PRIVATE DATA ELEMENT TAGS
    // eg: 0x0123,0x1425 -> 0x0123,0x0014
    if( IsPrivate() && !IsPrivateCreator() )
      {
      Tag r = *this;
      r.SetElement( (uint16_t)(GetElement() >> 8) );
      return r;
      }
    if( IsPrivateCreator() ) return *this;
    return Tag(0x0,0x0);
    }
  /// Set private creator:
  void SetPrivateCreator(Tag const &t)
    {
    // See PS 3.5 - 7.8.1 PRIVATE DATA ELEMENT TAGS
    // eg: 0x0123,0x0045 -> 0x0123,0x4567
    assert( t.IsPrivate() /*&& t.IsPrivateCreator()*/ );
    const uint16_t element = (uint16_t)(t.GetElement() << 8);
    const uint16_t base = (uint16_t)(GetElement() << 8);
    SetElement( (uint16_t)((base >> 8) + element) );
    SetGroup( t.GetGroup() );
    }

  /// Returns if tag is a Private Creator (xxxx,00yy), where xxxx is odd number
  /// and yy in [0x10,0xFF]
  bool IsPrivateCreator() const
    {
    return IsPrivate() && (GetElement() <= 0xFF && GetElement() >= 0x10);
    }

  /// return if the tag is considered to be an illegal tag
  bool IsIllegal() const
    {
    // DICOM reserved those groups:
    return GetGroup() == 0x0001 || GetGroup() == 0x0003 || GetGroup() == 0x0005 || GetGroup() == 0x0007
    // This is a very special case, in private group, one cannot use element [0x01,0x09] ...
//      || (IsPrivate() && !IsPrivateCreator() && !IsGroupLength());
      || (IsPrivate() && GetElement() > 0x0 && GetElement() < 0x10 );
    }

  /// return whether the tag correspond to a group length tag:
  bool IsGroupLength() const
    {
    return GetElement() == 0x0;
    }

  /// e.g 6002,3000 belong to groupXX: 6000,3000
  bool IsGroupXX(const Tag &t) const
    {
    if( t.GetElement() == GetElement() )
      {
      if( t.IsPrivate() ) return false;
      uint16_t group = (uint16_t)((GetGroup() >> 8 ) << 8);
      return group == t.GetGroup();
      }
    return false;
    }

  /// Read from a comma separated string.
  /// This is a highly user oriented function, the string should be formated as:
  /// 1234,5678 to specify the tag (0x1234,0x5678)
  /// The notation comes from the DICOM standard, and is handy to use from a
  /// command line program
  bool ReadFromCommaSeparatedString(const char *str);
  
  /// Read From XML formatted tag value eg. tag = "12345678"
  /// It comes in useful when reading tag values from XML file(in NativeDICOMModel)
  bool ReadFromContinuousString(const char *str);

  /// Print tag value with no separating comma: eg. tag = "12345678"
  /// It comes in useful when reading tag values from XML file(in NativeDICOMModel)
  std::string PrintAsContinuousString() const;

  /// Same as PrintAsContinuousString, but hexadecimal [a-f] are printed using upper case
  std::string PrintAsContinuousUpperCaseString() const;

  /// Read from a pipe separated string (GDCM 1.x compat only). Do not use in newer code
  /// \see ReadFromCommaSeparatedString
  bool ReadFromPipeSeparatedString(const char *str);

  /// Print as a pipe separated string (GDCM 1.x compat only). Do not use in newer code
  /// \see ReadFromPipeSeparatedString
  std::string PrintAsPipeSeparatedString() const;

private:
  union { uint32_t tag; uint16_t tags[2]; char bytes[4]; } ElementTag;
};
//-----------------------------------------------------------------------------
inline std::istream& operator>>(std::istream &_is, Tag &_val)
{
  char c;
  _is >> c;
  uint16_t a, b;
  _is >> std::hex >> a;
  //_is >> std::hex >> _val[0];
  //_is >> std::hex >> _val.ElementTag.tags[0];
  _is >> c;
  //_is >> _val[1];
  //_is >> std::hex >> _val.ElementTag.tags[1];
  _is >> std::hex >> b;
  _is >> c;
  _val.SetGroup( a );
  _val.SetElement( b );
  return _is;
}

inline std::ostream& operator<<(std::ostream &_os, const Tag &_val)
{
  _os.setf( std::ios::right);
  _os << std::hex << '(' << std::setw( 4 ) << std::setfill( '0' )
    << _val[0] << ',' << std::setw( 4 ) << std::setfill( '0' )
    << _val[1] << ')' << std::setfill( ' ' ) << std::dec;
  return _os;
}

} // end namespace gdcm

#endif //GDCMTAG_H
