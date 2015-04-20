/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMMODULEENTRY_H
#define GDCMMODULEENTRY_H

#include "gdcmTypes.h"
#include "gdcmType.h"

#include <string>

namespace gdcm
{
/**
 * \brief Class for representing a ModuleEntry
 * \note bla
 * \sa DictEntry
 */
class GDCM_EXPORT ModuleEntry
{
public:
  ModuleEntry(const char *name = "", const char *type = "3", const char *description = ""):Name(name)/*,Type(type)*/,DescriptionField(description) {
    DataElementType = Type::GetTypeType(type);
  }
  virtual ~ModuleEntry() {} // important
  friend std::ostream& operator<<(std::ostream& _os, const ModuleEntry &_val);

  void SetName(const char *name) { Name = name; }
  const char *GetName() const { return Name.c_str(); }

  void SetType(const Type &type) { DataElementType = type; }
  const Type &GetType() const { return DataElementType; }

  /*
   * WARNING: 'Description' is currently a std::string, but it might change in the future
   * do not expect it to remain the same, and always use the ModuleEntry::Description typedef
   * instead.
   */
  typedef std::string Description;
  void SetDescription(const char *d) { DescriptionField = d; }
  const Description & GetDescription() const { return DescriptionField; }

protected:
  // PS 3.3 repeats the name of an attribute, but often contains typos
  // for now we will not use this info, but instead access the DataDict instead
  std::string Name;

  // An attribute, encoded as a Data Element, may or may not be required in a
  // Data Set, depending on that Attribute's Data Element Type.
  Type DataElementType;

  // TODO: for now contains the raw description (with enumerated values, defined terms...)
  Description DescriptionField;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& _os, const ModuleEntry &_val)
{
  _os << _val.Name << "\t" << _val.DataElementType << "\t" << _val.DescriptionField;
  return _os;
}

typedef ModuleEntry MacroEntry;


} // end namespace gdcm

#endif //GDCMMODULEENTRY_H
