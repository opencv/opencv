/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPDBELEMENT_H
#define GDCMPDBELEMENT_H

#include "gdcmTag.h"
#include "gdcmVM.h"
#include "gdcmVR.h"
#include "gdcmByteValue.h"
#include "gdcmSmartPointer.h"

namespace gdcm
{
/**
 * \brief Class to represent a PDB Element
 * \see PDBHeader
 */
class GDCM_EXPORT PDBElement
{
public:
  PDBElement() {}

  friend std::ostream& operator<<(std::ostream &os, const PDBElement &val);

  /// Set/Get Name
  const char *GetName() const { return NameField.c_str(); }
  void SetName(const char *name) { NameField = name; }

  /// Set/Get Value
  const char *GetValue() const { return ValueField.c_str(); }
  void SetValue(const char *value) { ValueField = value; }

  bool operator==(const PDBElement &de) const
    {
    return ValueField == de.ValueField
      && NameField == de.NameField;
    }

protected:
  std::string NameField;
  std::string ValueField;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const PDBElement &val)
{
  os << val.NameField;
  os << " \"";
  os << val.ValueField;
  os << "\"";

  return os;
}

} // end namespace gdcm

#endif //GDCMPDBELEMENT_H
