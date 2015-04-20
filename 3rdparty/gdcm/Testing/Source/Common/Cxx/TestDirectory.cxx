/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDirectory.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"

#include <stdlib.h> // atoi

int TestOneDirectory(const char *path, bool recursive = false )
{
  if( !gdcm::System::FileIsDirectory(path) )
    {
    std::cerr << path << " is not a directory" << std::endl;
    return 1;
    }

  gdcm::Directory d;
  d.Load( path, recursive );
  //d.Print( std::cout );

  if( d.GetToplevel() != path )
    {
    std::cerr << d.GetToplevel() << " != " << path << std::endl;
    return 1;
    }
  gdcm::Directory::FilenamesType const &files = d.GetFilenames();
  for(gdcm::Directory::FilenamesType::const_iterator it = files.begin(); it != files.end(); ++it )
    {
    const char *filename = it->c_str();
    if( !gdcm::System::FileExists(filename) )
      {
      return 1;
      }
    }

  return 0;
}

int TestDirectory(int argc, char *argv[])
{
  int res = 0;
  if( argc > 1 )
    {
    bool recursive = false;
    if ( argc > 2 )
      {
      recursive = (atoi(argv[2]) > 0 ? true : false);
      }
    res += TestOneDirectory( argv[1], recursive);
    }
  else
    {
    const char *path = gdcm::Testing::GetDataRoot();
    res += TestOneDirectory( path );
    }

  //res += TestOneDirectory( "" );

  return res;
}
