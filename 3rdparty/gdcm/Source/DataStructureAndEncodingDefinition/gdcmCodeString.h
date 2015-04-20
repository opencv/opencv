/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCODESTRING_H
#define GDCMCODESTRING_H

#include "gdcmString.h"

namespace gdcm
{

/**
 * \brief CodeString
 * This is an implementation of DICOM VR: CS
 * The cstor will properly Trim so that operator== is correct.
 *
 * \note the cstor of CodeString will Trim the string on the fly so as
 * to remove the extra leading and ending spaces. However it will not
 * perform validation on the fly (CodeString obj can contains invalid
 * char such as lower cases). This design was chosen to be a little tolerant
 * to broken DICOM implementation, and thus allow user to compare lower
 * case CS from there input file without the need to first rewrite them
 * to get rid of invalid character (validation is a different operation from
 * searching, querying).
 * \warning when writing out DICOM file it is highly recommended to perform
 * the IsValid() call, at least to check that the length of the string match
 * the definition in the standard.
 */
// Note to myself: because note all wrapped language support exception
// we could not support throwing an exception during object construction.
class GDCM_EXPORT CodeString
{
  friend std::ostream& operator<< (std::ostream& os, const CodeString& str);
  friend bool operator==(const CodeString &ref, const CodeString& cs);
  friend bool operator!=(const CodeString &ref, const CodeString& cs);
  typedef String<'\\',16> InternalClass;
public:
  typedef InternalClass::value_type             value_type;
  typedef InternalClass::pointer                pointer;
  typedef InternalClass::reference              reference;
  typedef InternalClass::const_reference        const_reference;
  typedef InternalClass::size_type              size_type;
  typedef InternalClass::difference_type        difference_type;
  typedef InternalClass::iterator               iterator;
  typedef InternalClass::const_iterator         const_iterator;
  typedef InternalClass::reverse_iterator       reverse_iterator;
  typedef InternalClass::const_reverse_iterator const_reverse_iterator;

  /// CodeString constructors.
  CodeString(): Internal() {}
  CodeString(const value_type* s): Internal(s) { Internal = Internal.Trim(); }
  CodeString(const value_type* s, size_type n): Internal(s, n) {
    Internal = Internal.Trim(); }
  CodeString(const InternalClass& s, size_type pos=0, size_type n=InternalClass::npos):
    Internal(s, pos, n) { Internal = Internal.Trim(); }

  /// Check if CodeString obj is correct..
  bool IsValid() const;

  /// Return the full code string as std::string
  std::string GetAsString() const {
    return Internal;
  }

  /// Return the size of the string
  size_type Size() const { return Internal.size(); }

protected:
  std::string TrimInternal() const {
    return Internal.Trim();
  }

private:
  String<'\\',16> Internal;
};

inline std::ostream& operator<< (std::ostream& os, const CodeString& str)
{
  os << str.Internal;
  return os;
}

inline bool operator==(const CodeString &ref, const CodeString& cs)
{
  return ref.Internal == cs.Internal;
}
inline bool operator!=(const CodeString &ref, const CodeString& cs)
{
  return ref.Internal != cs.Internal;
}


} // end namespace gdcm

#endif //GDCMCODESTRING_H
