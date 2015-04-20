/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMABSTRACTSYNTAX_H
#define GDCMABSTRACTSYNTAX_H

#include "gdcmTypes.h"
#include "gdcmUIDs.h"
#include "gdcmDataElement.h"

namespace gdcm
{

namespace network
{

/**
 * \brief AbstractSyntax
 * Table 9-14
 * ABSTRACT SYNTAX SUB-ITEM FIELDS
 */
class AbstractSyntax
{
public:
  AbstractSyntax();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;

  void SetName( const char *name ) { UpdateName( name ); }
  const char *GetName() const { return Name.c_str(); }

  // accept a UIDs::TSType also...
  void SetNameFromUID( UIDs::TSName tsname );
  //now that the PresentationContext messes around with UIDs and returns a string
  //use that string as well.
  //void SetNameFromUIDString( const std::string& inUIDName );

  size_t Size() const;

  void Print(std::ostream &os) const;

  bool operator==(const AbstractSyntax & as) const
    {
    return Name == as.Name;
    }

  DataElement GetAsDataElement() const;

private:
  void UpdateName( const char *name );
  static const uint8_t ItemType;
  static const uint8_t Reserved2;
  uint16_t ItemLength; // len of
  std::string /*AbstractSyntax*/ Name; // UID
};

} // end namespace network
} // end namespace gdcm

#endif //GDCMABSTRACTSYNTAX_H
