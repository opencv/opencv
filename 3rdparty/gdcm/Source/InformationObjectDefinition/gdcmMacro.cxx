/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmMacro.h"
#include "gdcmDataSet.h"
#include "gdcmUsage.h"
//#include "gdcmDefs.h"
#include "gdcmModuleEntry.h" // MacroEntry
//#include "gdcmGlobal.h"

namespace gdcm
{

bool Macro::FindMacroEntry(const Tag &tag) const
{
  MapModuleEntry::const_iterator it = ModuleInternal.find(tag);
  if( it != ModuleInternal.end() )
    {
    return true;
    }
  // Not found anywhere :(
  return false;
}

const MacroEntry& Macro::GetMacroEntry(const Tag &tag) const
{
  MapModuleEntry::const_iterator it = ModuleInternal.find(tag);
  if( it != ModuleInternal.end() )
    {
    assert( it->first == tag );
    return it->second;
    }
  // Not found anywhere :(
  throw "Could not find Module for Tag requested";
}

bool Macro::Verify(const DataSet& ds, Usage const & usage) const
{
  bool success = true;
  if( usage == Usage::UserOption ) return success;
  Macro::MapModuleEntry::const_iterator it = ModuleInternal.begin();
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
