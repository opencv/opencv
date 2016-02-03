/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTABLEENTRY_H
#define GDCMTABLEENTRY_H

#include "gdcmType.h"

#include <string>

namespace gdcm
{

/**
 * \brief TableEntry
 */
class TableEntry
{
public:
  TableEntry(const char *attribute = 0,
    Type const &type = Type(), const char * des = 0 ) :
    Attribute(attribute ? attribute : ""),TypeField(type),Description(des ? des : "") {}
  ~TableEntry() {}

private:
  std::string Attribute;
  Type TypeField;
  std::string Description;
};

} // end namespace gdcm

#endif //GDCMTABLEENTRY_H
