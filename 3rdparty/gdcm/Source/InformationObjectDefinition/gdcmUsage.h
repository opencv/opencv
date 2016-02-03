/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMUSAGE_H
#define GDCMUSAGE_H

#include "gdcmTypes.h"

#include <iostream>

namespace gdcm
{

/**
 * \brief Usage
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

A.1.3.2 CONDITIONAL MODULES
Conditional Modules are Mandatory Modules if specific conditions are met. If the specified conditions are
not met, this Module shall not be supported; that is, no information defined in that Module shall be sent.
A.1.3.3 USER OPTION MODULES
User Option Modules may or may not be supported. If an optional Module is supported, the Attribute
Types specified in the Modules in Annex C shall be supported.
 */
class GDCM_EXPORT Usage
{
public:
  typedef enum {
    Mandatory, // (see A.1.3.1) , abbreviated M
    Conditional, // (see A.1.3.2) , abbreviated C
    UserOption, // (see A.1.3.3) , abbreviated U
    Invalid
  } UsageType;

  Usage(UsageType type = Invalid) : UsageField(type) { }

  operator UsageType () const { return UsageField; }
  friend std::ostream &operator<<(std::ostream &os, const Usage &vr);

  static const char *GetUsageString(UsageType type);
  static UsageType GetUsageType(const char *type);

private:
  UsageType UsageField;
};
//-----------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &_os, const Usage &val)
{
  _os << Usage::GetUsageString(val.UsageField);
  return _os;
}

} // end namespace gdcm

#endif //GDCMUSAGE_H
