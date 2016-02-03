/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMIODENTRY_H
#define GDCMIODENTRY_H

#include "gdcmUsage.h"
#include "gdcmType.h"

#include <string>

namespace gdcm
{
/**
 * \brief Class for representing a IODEntry
 * \note
A.1.3 IOD Module Table and Functional Group Macro Table
This Section of each IOD defines in a tabular form the Modules comprising the IOD. The following
information must be specified for each Module in the table:
- The name of the Module or Functional Group
- A reference to the Section in Annex C which defines the Module or Functional Group
- The usage of the Module or Functional Group; whether it is:
- Mandatory (see A.1.3.1) , abbreviated M
- Conditional (see A.1.3.2) , abbreviated C
- User Option (see A.1.3.3) , abbreviated U
The Modules referenced are defined in Annex C.
A.1.3.1 MANDATORY MODULES
For each IOD, Mandatory Modules shall be supported per the definitions, semantics and requirements
defined in Annex C.
PS 3.3 - 2008
Page 96
- Standard -
A.1.3.2 CONDITIONAL MODULES
Conditional Modules are Mandatory Modules if specific conditions are met. If the specified conditions are
not met, this Module shall not be supported; that is, no information defined in that Module shall be sent.
A.1.3.3 USER OPTION MODULES
User Option Modules may or may not be supported. If an optional Module is supported, the Attribute
Types specified in the Modules in Annex C shall be supported.
 * \sa DictEntry
 */
class GDCM_EXPORT IODEntry
{
public:
  IODEntry(const char *name = "", const char *ref = "", const char *usag = ""):Name(name),Ref(ref),usage(usag) {
  }
  friend std::ostream& operator<<(std::ostream& _os, const IODEntry &_val);

  void SetIE(const char *ie) { IE = ie; }
  const char *GetIE() const { return IE.c_str(); }

  void SetName(const char *name) { Name = name; }
  const char *GetName() const { return Name.c_str(); }

  void SetRef(const char *ref) { Ref = ref; }
  const char *GetRef() const { return Ref.c_str(); }

  void SetUsage(const char *usag) { usage = usag; }
  const char *GetUsage() const { return usage.c_str(); }
  Usage::UsageType GetUsageType() const;

private:
  std::string IE;

  std::string Name;

  std::string Ref;

  std::string usage;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& _os, const IODEntry &_val)
{
  _os << _val.IE << "\t" << _val.Name << "\t" << _val.Ref << "\t" << _val.usage;
  return _os;
}

} // end namespace gdcm

#endif //GDCMIODENTRY_H
