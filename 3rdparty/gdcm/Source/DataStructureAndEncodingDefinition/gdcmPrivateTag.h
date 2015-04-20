/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPRIVATETAG_H
#define GDCMPRIVATETAG_H

#include "gdcmTag.h"
#include "gdcmVR.h"
#include "gdcmDataElement.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

#include <string.h> // strlen
#include <ctype.h> // tolower

namespace gdcm
{

/**
 * \brief Class to represent a Private DICOM Data Element (Attribute) Tag (Group, Element, Owner)
 * \note private tag have element value in: [0x10,0xff], for instance 0x0009,0x0000 is NOT a private tag
 */

// TODO: We could save some space since we only store 8bits for element
class GDCM_EXPORT PrivateTag : public Tag
{
  friend std::ostream& operator<<(std::ostream &_os, const PrivateTag &_val);
public:
  PrivateTag(uint16_t group = 0, uint16_t element = 0, const char *owner = ""):Tag(group,element),Owner(owner ? LOComp::Trim(owner) : "") {
    // truncate the high bits
    SetElement( (uint8_t)element );
  }
  PrivateTag( Tag const & t, const char *owner = ""):Tag(t),Owner(owner ? LOComp::Trim(owner) : "") {
    // truncate the high bits
    SetElement( (uint8_t)t.GetElement());
  }

  const char *GetOwner() const { return Owner.c_str(); }
  void SetOwner(const char *owner) { if(owner) Owner = LOComp::Trim(owner); }

  bool operator<(const PrivateTag &_val) const;

  /// Read PrivateTag from a string. Element number will be truncated
  /// to 8bits. Eg: "1234,5678,GDCM" is private tag: (1234,78,"GDCM")
  bool ReadFromCommaSeparatedString(const char *str);

  DataElement GetAsDataElement() const;

private:
  // SIEMENS MED, GEMS_PETD_01 ...
  std::string Owner;
};

inline std::ostream& operator<<(std::ostream &os, const PrivateTag &val)
{
  //assert( !val.Owner.empty() );
  os.setf( std::ios::right );
  os << std::hex << '(' << std::setw( 4 ) << std::setfill( '0' )
    << val[0] << ',' << std::setw( 2 ) << std::setfill( '0' )
    << val[1] << ',';
  os << val.Owner;
  os << ')' << std::setfill( ' ' ) << std::dec;
  return os;
}

} // end namespace gdcm

#endif //GDCMPRIVATETAG_H
