/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCSAHEADERDICTENTRY_H
#define GDCMCSAHEADERDICTENTRY_H

#include "gdcmVR.h"
#include "gdcmVM.h"

#include <string>
#include <iostream>
#include <iomanip>

#include <cstring>

namespace gdcm
{
/**
 * \brief Class to represent an Entry in the Dict
 * Does not really exist within the DICOM definition, just a way to minimize
 * storage and have a mapping from gdcm::Tag to the needed information
 * \note bla
 * TODO FIXME: Need a PublicCSAHeaderDictEntry...indeed CSAHeaderDictEntry has a notion of retired which
 * does not exist in PrivateCSAHeaderDictEntry...
 *
 * \see gdcm::Dict
 */
class GDCM_EXPORT CSAHeaderDictEntry
{
public:
  CSAHeaderDictEntry(const char *name = "", VR const &vr = VR::INVALID, VM const &vm = VM::VM0, const char *desc = ""):Name(name),ValueRepresentation(vr),ValueMultiplicity(vm),Description(desc) {
  }

  friend std::ostream& operator<<(std::ostream& _os, const CSAHeaderDictEntry &_val);

  /// Set/Get VR
  const VR &GetVR() const { return ValueRepresentation; }
  void SetVR(const VR & vr) { ValueRepresentation = vr; }

  /// Set/Get VM
  const VM &GetVM() const { return ValueMultiplicity; }
  void SetVM(VM const & vm) { ValueMultiplicity = vm; }

  /// Set/Get Name
  const char *GetName() const { return Name.c_str(); }
  void SetName(const char* name) { Name = name; }

  /// Set/Get Description
  const char *GetDescription() const { return Description.c_str(); }
  void SetDescription(const char* desc) { Description = desc; }

  bool operator<(const CSAHeaderDictEntry &entry) const
    {
    return strcmp(GetName(),entry.GetName()) < 0;
    }

private:
  std::string Name;
  VR ValueRepresentation;
  VM ValueMultiplicity;
  std::string Description;
  std::string Type; // TODO
};


//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& os, const CSAHeaderDictEntry &val)
{
  if( val.Name.empty() )
    {
    os << "[No name]";
    }
  else
    {
    os << val.Name;
    }
  os << "\t" << val.ValueRepresentation << "\t" << val.ValueMultiplicity;
  if( !val.Description.empty() )
    {
    os << "\t" << val.Description;
    }
  return os;
}

} // end namespace gdcm

#endif //GDCMCSAHEADERDICTENTRY_H
