/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFILENAME_H
#define GDCMFILENAME_H

#include "gdcmTypes.h"

#include <string>

namespace gdcm
{
/**
 * \brief Class to manipulate file name's
 * \note OS independant representation of a filename (to query path, name and extension from a filename)
 */
class GDCM_EXPORT Filename
{
public:
  Filename(const char* filename = ""):FileName(filename ? filename : ""),Path(),Conversion() {}

  /// Return the full filename
  const char *GetFileName() const { return FileName.c_str(); }
  /// Return only the path component of a filename
  const char *GetPath();
  /// return only the name part of a filename
  const char *GetName();
  /// return only the extension part of a filename
  const char *GetExtension();
  /// Convert backslash (windows style) to UNIX style slash.
  const char *ToUnixSlashes();
  /// Convert foward slash (UNIX style) to windows style slash.
  const char *ToWindowsSlashes();

  /// Join two paths
  /// NOT THREAD SAFE
  static const char *Join(const char *path, const char *filename);

  /// return whether the filename is empty
  bool IsEmpty() const { return FileName.empty(); }

  /// Simple operator to allow
  /// Filename myfilename( "..." );
  /// const char * s = myfilename;
  operator const char * () const { return GetFileName(); }

  // FIXME: I don't like this function
  // It hides the realpath call (maybe usefull)
  // and it forces file to exist on the disk whereas Filename
  // should be independant from file existence.
  bool IsIdentical(Filename const &fn) const;

  /// Does the filename ends with a particular string ?
  bool EndWith(const char ending[]) const;

private:
  std::string FileName;
  std::string Path;
  std::string Conversion;
};

} // end namespace gdcm

#endif //GDCMFILENAME_H
