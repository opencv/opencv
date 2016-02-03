/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmModule.h"
#include "gdcmDataSet.h"
#include "gdcmUsage.h"
//#include "gdcmDefs.h"
#include "gdcmMacros.h" // Macros
//#include "gdcmGlobal.h"

namespace gdcm
{

bool Module::FindModuleEntryInMacros(Macros const &macros, const Tag &tag) const
{
  (void)macros;
  MapModuleEntry::const_iterator it = ModuleInternal.find(tag);
  if( it != ModuleInternal.end() )
    {
    return true;
    }
  // Need to search within Nested-Included Macro:
  // start with easy case:
  if( ArrayIncludeMacros.empty() ) return false;

  // we need to loop over all Included-Macro:
  //static const Global &g = Global::GetInstance();
  //static const Defs &defs = g.GetDefs();
  //static const Macros &macros = defs.GetMacros();

#if 0
  for( ArrayIncludeMacrosType::const_iterator it = ArrayIncludeMacros.begin();
    it != ArrayIncludeMacros.end(); ++it)
    {
    const std::string &name = *it;
    const Macro &macro = macros.GetMacro( name.c_str() );
    if( macro.FindMacroEntry( tag ) )
      {
      return true;
      }
    }
#endif
  // Not found anywhere :(
  return false;
}

const ModuleEntry& Module::GetModuleEntryInMacros(Macros const &macros, const Tag &tag) const
{
  MapModuleEntry::const_iterator it = ModuleInternal.find(tag);
  if( it != ModuleInternal.end() )
    {
    assert( it->first == tag );
    return it->second;
    }
  // Need to search within Nested-Included Macro:
  // start with easy case:
  if( ArrayIncludeMacros.empty() )
    {
    throw "Could not find Module for Tag requested";
    }

  // we need to loop over all Included-Macro:
  //static const Global &g = Global::GetInstance();
  //static const Defs &defs = g.GetDefs();
  //static const Macros &macros = defs.GetMacros();

  for( ArrayIncludeMacrosType::const_iterator it2 = ArrayIncludeMacros.begin();
    it2 != ArrayIncludeMacros.end(); ++it2)
    {
    const std::string &name = *it2;
    const Macro &macro= macros.GetMacro( name.c_str() );
    if( macro.FindMacroEntry( tag ) )
      {
      return macro.GetMacroEntry(tag);
      }
    }
  // Not found anywhere :(
  throw "Could not find Module for Tag requested";
}

bool Module::Verify(const DataSet& ds, Usage const & usage) const
{
  bool success = true;
  if( usage == Usage::UserOption ) return success;
  Module::MapModuleEntry::const_iterator it = ModuleInternal.begin();
  for(;it != ModuleInternal.end(); ++it)
    {
    const Tag &tag = it->first;
    const ModuleEntry &me = it->second;
    const gdcm::Type &type = me.GetType();
    if( ds.FindDataElement( tag ) )
      {
      // element found
      const DataElement &de = ds.GetDataElement( tag );
      if ( de.IsEmpty() && (type == Type::T1 || type == Type::T1C ) )
        {
        gdcmWarningMacro( "T1 element cannot be empty: " << de );
        success = false;
        }
      }
    else
      {
      if( type == Type::T1 || type == Type::T1C )
        {
        gdcmWarningMacro( "DataSet is missing tag: " << tag );
        gdcmWarningMacro( "ModuleEntry specify: " << me );
        gdcmWarningMacro( "Usage is: " << usage );
        success = false;
        }
      }
    }

  return success;
}

}
