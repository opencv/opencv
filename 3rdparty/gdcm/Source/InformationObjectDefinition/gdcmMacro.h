/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMMACRO_H
#define GDCMMACRO_H

#include "gdcmTypes.h"
#include "gdcmTag.h"
#include "gdcmMacroEntry.h"

#include <map>
#include <vector>

namespace gdcm
{

class DataSet;
class Usage;
/**
 * \brief Class for representing a Macro
 * \note Attribute Macro:
 * a set of Attributes that are described in a single table that is referenced
 * by multiple Module or other tables.
 * \sa Module
 */
class GDCM_EXPORT Macro
{
public:
  typedef std::map<Tag, MacroEntry> MapModuleEntry;
  typedef std::vector<std::string> ArrayIncludeMacrosType;

  //typedef MapModuleEntry::const_iterator ConstIterator;
  //typedef MapModuleEntry::iterator Iterator;
  //ConstIterator Begin() const { return ModuleInternal.begin(); }
  //Iterator Begin() { return ModuleInternal.begin(); }
  //ConstIterator End() const { return ModuleInternal.end(); }
  //Iterator End() { return ModuleInternal.end(); }

  Macro() {}
  friend std::ostream& operator<<(std::ostream& _os, const Macro&_val);

  void Clear() { ModuleInternal.clear(); }

  /// Will add a ModuleEntry direcly at root-level. See Macro for nested-included level.
  void AddMacroEntry(const Tag& tag, const MacroEntry & module)
    {
    ModuleInternal.insert(
      MapModuleEntry::value_type(tag, module));
    }

  /// Find or Get a ModuleEntry. ModuleEntry are either search are root-level
  /// or within nested-macro included in module.
  bool FindMacroEntry(const Tag &tag) const;
  const MacroEntry& GetMacroEntry(const Tag &tag) const;

  void SetName( const char *name) { Name = name; }
  const char *GetName() const { return Name.c_str(); }

  // Verify will print on std::cerr for error
  // Upon success will return true, false otherwise
  bool Verify(const DataSet& ds, Usage const & usage) const;

private:
  //Module &operator=(const Module &_val); // purposely not implemented
  //Module(const Module &_val); // purposely not implemented

  MapModuleEntry ModuleInternal;
  std::string Name;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& _os, const Macro &_val)
{
  _os << _val.Name << '\n';
  Macro::MapModuleEntry::const_iterator it = _val.ModuleInternal.begin();
  for(;it != _val.ModuleInternal.end(); ++it)
    {
    const Tag &t = it->first;
    const MacroEntry &de = it->second;
    _os << t << " " << de << '\n';
    }

  return _os;
}

} // end namespace gdcm

#endif //GDCMMACRO_H
