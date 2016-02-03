/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMTYPE_H
#define GDCMTYPE_H

#include "gdcmTypes.h"

#include <iostream>

namespace gdcm
{

/**
 * \brief Type
 * \note
 * PS 3.5
 * 7.4 DATA ELEMENT TYPE
 * 7.4.1 TYPE 1 REQUIRED DATA ELEMENTS
 * 7.4.2 TYPE 1C CONDITIONAL DATA ELEMENTS
 * 7.4.3 TYPE 2 REQUIRED DATA ELEMENTS
 * 7.4.4 TYPE 2C CONDITIONAL DATA ELEMENTS
 * 7.4.5 TYPE 3 OPTIONAL DATA ELEMENTS
 *
 * The intent of Type 2 Data Elements is to allow a zero length to be conveyed
 * when the operator or application does not know its value or has a specific
 * reason for not specifying its value. It is the intent that the device should
 * support these Data Elements.
 */
class GDCM_EXPORT Type
{
public:
  typedef enum {
    T1 = 0,
    T1C,
    T2,
    T2C,
    T3,
    UNKNOWN
  } TypeType;

  Type(TypeType type = UNKNOWN) : TypeField(type) { }

  operator TypeType () const { return TypeField; }
  friend std::ostream &operator<<(std::ostream &os, const Type &vr);

  static const char *GetTypeString(TypeType type);
  static TypeType GetTypeType(const char *type);

private:
  TypeType TypeField;
};
//-----------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &_os, const Type &val)
{
  _os << Type::GetTypeString(val.TypeField);
  return _os;
}

} // end namespace gdcm

#endif //GDCMTYPE_H
