/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmGlobal.h"
#include "gdcmDicts.h"
#include "gdcmDefs.h"
#include "gdcmFilename.h"

#include <limits.h> // PATH_MAX
#include <string.h> // strcpy
#ifdef _WIN32
#include <windows.h> // MAX_PATH
#endif

namespace gdcm
{

// Must NOT be initialized.  Default initialization to zero is
// necessary.
unsigned int GlobalCount;

class GlobalInternal
{
public:
  GlobalInternal():GlobalDicts(),GlobalDefs() {}
  Dicts GlobalDicts; // Part 6 + Part 4 elements
// TODO need H table for TransferSyntax / MediaStorage / Part 3 ...
  Defs GlobalDefs;

  // Ressource paths:
  // By default only construct two paths:
  // - The official install dir (need to keep in sinc with cmakelist variable
  // - a dynamic one, so that gdcm is somewhat rellocatable
  // - on some system where it make sense the path where the Resource should be located
  void LoadDefaultPaths()
    {
    assert( RessourcePaths.empty() );
    const char filename2[] = GDCM_CMAKE_INSTALL_PREFIX "/" GDCM_INSTALL_DATA_DIR "/XML/";
    RessourcePaths.push_back( filename2 );
    const char filename3[] = GDCM_CMAKE_INSTALL_PREFIX " " GDCM_API_VERSION "/" GDCM_INSTALL_DATA_DIR "/XML/";
    RessourcePaths.push_back( filename3 );
    const char *curprocfn = System::GetCurrentProcessFileName();
    if( curprocfn )
      {
      Filename fn( curprocfn );
      std::string str = fn.GetPath();
      str += "/../" GDCM_INSTALL_DATA_DIR "/XML/";
      RessourcePaths.push_back( str );
      }
    const char *respath = System::GetCurrentResourcesDirectory();
    if( respath )
      {
      RessourcePaths.push_back( respath );
      }
#ifdef GDCM_BUILD_TESTING
    // Needed for backward compat and dashboard
    const char src_path[] = GDCM_SOURCE_DIR "/Source/InformationObjectDefinition/";
    RessourcePaths.push_back( src_path );
#endif
    }
  std::vector<std::string> RessourcePaths;
};

Global::Global()
{
  if(++GlobalCount == 1)
    {
    assert( Internals == NULL ); // paranoid
    Internals = new GlobalInternal;
    assert( Internals->GlobalDicts.IsEmpty() );
    // Fill in with default values now !
    // at startup time is safer as later call might be from different thread
    // thus initialization of std::map would be all skrew up
    Internals->GlobalDicts.LoadDefaults();
    assert( Internals->GlobalDefs.IsEmpty() );
    // Same goes for GlobalDefs:
    //Internals->GlobalDefs.LoadDefaults();
    Internals->LoadDefaultPaths();
    }
}

Global::~Global()
{
  if(--GlobalCount == 0)
    {
    //Internals->GlobalDicts.Unload();
    delete Internals;
    Internals = NULL; // paranoid
    }
}

bool Global::LoadResourcesFiles()
{
  assert( Internals != NULL ); // paranoid
  const char *filename = Locate( "Part3.xml" );
  if( filename )
    {
    if( Internals->GlobalDefs.IsEmpty() )
      Internals->GlobalDefs.LoadFromFile(filename);
    return true;
    }
  // resource manager was not set properly
  return false;
}

bool Global::Append(const char *path)
{
  if( !System::FileIsDirectory(path) )
    {
    return false;
    }
  Internals->RessourcePaths.push_back( path );
  return true;
}

bool Global::Prepend(const char *path)
{
  if( !System::FileIsDirectory(path) )
    {
    return false;
    }
  Internals->RessourcePaths.insert( Internals->RessourcePaths.begin(), path );
  return true;
}

const char *Global::Locate(const char *resfile) const
{
#ifdef _WIN32
  static char path[MAX_PATH];
#else
  static char path[PATH_MAX];
#endif

  std::vector<std::string>::const_iterator it = Internals->RessourcePaths.begin();
  for( ; it != Internals->RessourcePaths.end(); ++it)
    {
    const std::string &p = *it;
    gdcmDebugMacro( "Trying to locate in: " << p );
    std::string fullpath = p + "/" + resfile;
    if( System::FileExists(fullpath.c_str()) )
      {
      // we found a match
      // check no invalid write access possible:
      if( fullpath.size() >= sizeof(path) )
        {
        gdcmDebugMacro( "Impossible happen: path is too long" );
        return NULL;
        }
      strcpy(path, fullpath.c_str() );
      return path;
      }
    }
  // no match sorry  :(
  return NULL;
}

Dicts const &Global::GetDicts() const
{
  assert( !Internals->GlobalDicts.IsEmpty() );
  return Internals->GlobalDicts;
}

Dicts &Global::GetDicts()
{
  assert( !Internals->GlobalDicts.IsEmpty() );
  return Internals->GlobalDicts;
}

Defs const &Global::GetDefs() const
{
  assert( !Internals->GlobalDefs.IsEmpty() );
  return Internals->GlobalDefs;
}

Global& Global::GetInstance()
{
  return GlobalInstance;
}

// Purposely not initialized.  ClassInitialize will handle it.
GlobalInternal * Global::Internals;


} // end namespace gdcm
