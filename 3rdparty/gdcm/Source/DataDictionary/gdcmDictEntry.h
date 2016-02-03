/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDICTENTRY_H
#define GDCMDICTENTRY_H

#include "gdcmVR.h"
#include "gdcmVM.h"

#include <string>
#include <iostream>
#include <iomanip>

namespace gdcm
{
/**
 * \brief Class to represent an Entry in the Dict
 * Does not really exist within the DICOM definition, just a way to minimize
 * storage and have a mapping from gdcm::Tag to the needed information
 * \note bla
 * TODO FIXME: Need a PublicDictEntry...indeed DictEntry has a notion of retired which
 * does not exist in PrivateDictEntry...
 *
 * \see gdcm::Dict
 */
class GDCM_EXPORT DictEntry
{
public:
  DictEntry(const char *name = "", const char *keyword = "", VR const &vr = VR::INVALID, VM const &vm = VM::VM0, bool ret = false):
    Name(name),
    Keyword(keyword),
    ValueRepresentation(vr),
    ValueMultiplicity(vm),
    Retired(ret),
    GroupXX(false),
    ElementXX(false)
  {
  }

  friend std::ostream& operator<<(std::ostream& _os, const DictEntry &_val);

  /// Set/Get VR
  const VR &GetVR() const { return ValueRepresentation; }
  void SetVR(const VR & vr) { ValueRepresentation = vr; }
//  bool IsValid() const { return ValueRepresentation != VR::VR_END; }
//	  !Name.empty() /*&& ValueRepresentation && ValueMultiplicity*/; }

  /// Set/Get VM
  const VM &GetVM() const { return ValueMultiplicity; }
  void SetVM(VM const & vm) { ValueMultiplicity = vm; }

  /// Set/Get Name
  const char *GetName() const { return Name.c_str(); }
  void SetName(const char* name) { Name = name; }

  /// same as GetName but without spaces...
  const char *GetKeyword() const { return Keyword.c_str(); }
  void SetKeyword(const char* keyword) { Keyword = keyword; }

  /// Set/Get Retired flag
  bool GetRetired() const { return Retired; }
  void SetRetired(bool retired) { Retired = retired; }

  // <entry group="50xx" element="0005" vr="US" vm="1" retired="true" version="3">
  /// Set whether element is shared in multiple groups (Curve/Overlay typically)
  void SetGroupXX(bool v) { GroupXX = v; }

  // <entry group="0020" element="31xx" vr="CS" vm="1-n" retired="true" version="2">
  /// Set whether element is shared in multiple elements (Source Image IDs typically)
  void SetElementXX(bool v) { ElementXX = v; }

  /// Return whether the name of the DataElement can be considered to be unique.
  /// As of 2008 all elements name were unique (except the expclitely 'XX' ones)
  bool IsUnique() const { return ElementXX == false && GroupXX == false; }

private:
  //
  friend class Dict;
  static bool CheckKeywordAgainstName(const char *name, const char *keyword);

private:
  std::string Name;
  std::string Keyword;
  VR ValueRepresentation;
  VM ValueMultiplicity;
  bool Retired : 1;
  bool GroupXX : 1;
  bool ElementXX : 1;
};

#if 0
class GDCM_EXPORT PrivateDictEntry : public DictEntry
{
public:
  PrivateDictEntry(const char *name = "", VR::VRType const &vr = VR::INVALID, VM::VMType const &vm = VM::VM0 , bool ret = false, const char *owner = ""):DictEntry(name,vr,vm,ret),Owner(owner) {}
  PrivateDictEntry(const char *name, const char *vr, const char *vm):DictEntry(name,vr,vm) {}

  const char *GetOwner() const { return Owner.c_str(); }
  void SetOwner(const char *owner) { Owner = owner; }

private:
  // SIEMENS MED, GEMS_PETD_01 ...
  std::string Owner;
};
#endif

//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& os, const DictEntry &val)
{
  if( val.Name.empty() )
    {
    os << "[No name]";
    }
  else
    {
    os << val.Name;
    }
  if( val.Keyword.empty() )
    {
    os << "[No keyword]";
    }
  else
    {
    os << val.Keyword;
    }
  os << "\t" << val.ValueRepresentation << "\t" << val.ValueMultiplicity;
  if( val.Retired )
    {
    os << "\t(RET)";
    }
  return os;
}

} // end namespace gdcm

#endif //GDCMDICTENTRY_H
