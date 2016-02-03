/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileSet.h"
#include "gdcmSystem.h"

namespace gdcm
{
  bool FileSet::AddFile(const char *filename)
    {
    if( System::FileExists(filename) )
      {
      Files.push_back( filename );
      return true;
      }
    return false;
    }
  void FileSet::SetFiles(FilesType const &files)
    {
    FilesType::const_iterator it = files.begin();
    for( ; it != files.end(); ++it )
      {
      AddFile( it->c_str() );
      }
    }

}
