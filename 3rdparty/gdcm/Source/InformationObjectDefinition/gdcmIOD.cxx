/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIOD.h"

#include "gdcmDataSet.h"
#include "gdcmModule.h"
#include "gdcmModules.h"
#include "gdcmDefs.h"

namespace gdcm
{

Type IOD::GetTypeFromTag(const Defs &defs, const Tag& tag) const
{
  Type ret;
  const IOD &iod = *this;
  static const Modules &modules = defs.GetModules();
  static const Macros &macros = defs.GetMacros();

  const size_t niods = iod.GetNumberOfIODs();
  // Iterate over each iod entry in order:
  bool found = false;
  for(unsigned int idx = 0; !found && idx < niods; ++idx)
    {
    const IODEntry &iodentry = iod.GetIODEntry(idx);
    const char *ref = iodentry.GetRef();
    //Usage::UsageType ut = iodentry.GetUsageType();

    const Module &module = modules.GetModule( ref );
    if( module.FindModuleEntryInMacros(macros, tag ) )
      {
      const ModuleEntry &module_entry = module.GetModuleEntryInMacros(macros,tag);
      ret = module_entry.GetType();
      found = true;
      }
    }

  return ret;
}

}
