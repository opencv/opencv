/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDEFS_H
#define GDCMDEFS_H

#include "gdcmModules.h"
#include "gdcmMacros.h"
#include "gdcmIODs.h"

#include <string>

namespace gdcm
{
class DataSet;
class File;
class MediaStorage;
/**
 * \brief FIXME I do not like the name 'Defs'
 * \note bla
 */
class GDCM_EXPORT Defs
{
public:
  Defs();
  ~Defs();

  const Modules &GetModules() const { return Part3Modules; }
  Modules &GetModules() { return Part3Modules; }

  /// Users should not directly use Macro. Macro are simply a way for DICOM WG to re-use Tables.
  /// Macros are conviently wraped within Modules. See gdcm::Module API directly
  const Macros &GetMacros() const { return Part3Macros; }
  Macros &GetMacros() { return Part3Macros; }

  const IODs & GetIODs() const { return Part3IODs; }
  IODs & GetIODs() { return Part3IODs; }

  bool IsEmpty() const { return GetModules().IsEmpty(); }

  bool Verify(const File& file) const;

  // \deprecated DO NOT USE
  bool Verify(const DataSet& ds) const;

  Type GetTypeFromTag(const File& file, const Tag& tag) const;

  static const char *GetIODNameFromMediaStorage(MediaStorage const &ms);

  const IOD& GetIODFromFile(const File& file) const;

protected:
  friend class Global;
  void LoadDefaults();
  void LoadFromFile(const char *filename);

private:
  // Part 3 stuff:
  Macros Part3Macros;
  Modules Part3Modules;
  IODs Part3IODs;

  Defs &operator=(const Defs &val); // purposely not implemented
  Defs(const Defs &val); // purposely not implemented
};


} // end namespace gdcm

#endif //GDCMDEFS_H
