/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// Implementation detail was shamelessly borowed from the VTK excellent
// implementation of debug leak manager singleton:
/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkDebugLeaks.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMGLOBAL_H
#define GDCMGLOBAL_H

#include "gdcmTypes.h"

namespace gdcm
{
class GlobalInternal;
class Dicts;
class Defs;
/**
 * \brief Global
 * \note
 * Global should be included in any translation unit
 * that will use Dict or that implements the singleton
 * pattern.  It makes sure that the Dict singleton is created
 * before and destroyed after all other singletons in GDCM.
 *
 */
class GDCM_EXPORT Global // why expose the symbol I think I only need to expose the instance...
{
  friend std::ostream& operator<<(std::ostream &_os, const Global &g);
public:
  Global();
  ~Global();

  /// retrieve the default/internal dicts (Part 6)
  /// This dict is filled up at load time
  Dicts const &GetDicts() const;
  Dicts &GetDicts();

  /// retrieve the default/internal (Part 3)
  /// You need to explicitely call LoadResourcesFiles before
  Defs const &GetDefs() const;

  /// return the singleton instance
  static Global& GetInstance();

  /// Load all internal XML files, resource path need to have been
  /// set before calling this member function (see Append/Prepend members func)
  /// \warning not thread safe !
  bool LoadResourcesFiles();

  /// Append path at the end of the path list
  /// \warning not thread safe !
  bool Append(const char *path);

  /// Prepend path at the beginning of the path list
  /// \warning not thread safe !
  bool Prepend(const char *path);

protected:
  /// Locate a resource file
  const char *Locate(const char *resfile) const;

private:
  Global &operator=(const Global &_val); // purposely not implemented
  Global(const Global &_val); // purposely not implemented
  // PIMPL:
  // but we could have also directly exposed a Dicts *Internals;
  static GlobalInternal *Internals;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const Global &g)
{
  (void)g;
  return os;
}

// This instance will show up in any translation unit that uses
// Global or that has a singleton.  It will make sure
// Global is initialized before it is used and is the last
// static object destroyed.
static Global GlobalInstance;

} // end namespace gdcm

#endif //GDCMGLOBAL_H
