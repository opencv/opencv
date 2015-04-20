/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMMODULES_H
#define GDCMMODULES_H

#include "gdcmTypes.h"
#include "gdcmModule.h"

#include <map>

namespace gdcm
{
/**
 * \brief Class for representing a Modules
 * \note bla
 * \sa Module
 */
class GDCM_EXPORT Modules
{
public:
  typedef std::map<std::string, Module> ModuleMapType;

  Modules() {}
  friend std::ostream& operator<<(std::ostream& _os, const Modules &_val);

  void Clear() { ModulesInternal.clear(); }

  // A Module is inserted based on it's ref
  void AddModule(const char *ref, const Module & module )
    {
    assert( ref && *ref );
    assert( ModulesInternal.find( ref ) == ModulesInternal.end() );
    ModulesInternal.insert(
      ModuleMapType::value_type(ref, module));
    }
  const Module &GetModule(const char *name) const
    {
    assert( name && *name );
    ModuleMapType::const_iterator it = ModulesInternal.find( name );
    assert( it != ModulesInternal.end() );
    assert( it->first == name );
    return it->second;
    }

  bool IsEmpty() const { return ModulesInternal.empty(); }

private:
  ModuleMapType ModulesInternal;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& _os, const Modules &_val)
{
  Modules::ModuleMapType::const_iterator it = _val.ModulesInternal.begin();
  for(;it != _val.ModulesInternal.end(); ++it)
    {
    const std::string &name = it->first;
    const Module &m = it->second;
    _os << name << " " << m << '\n';
    }

  return _os;
}



} // end namespace gdcm

#endif //GDCMMODULES_H
