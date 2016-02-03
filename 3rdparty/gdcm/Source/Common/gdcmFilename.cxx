/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFilename.h"
#include <limits.h>
#include <stdlib.h> // realpath
#include <assert.h>
#include <string.h>

namespace gdcm
{

/**
 * Return path of a full filename (no trailing slashes).
 * Warning: returned path is converted to Unix slashes format.
 */
const char *Filename::GetPath()
{
  std::string fn = ToUnixSlashes();

  std::string::size_type slash_pos = fn.rfind("/");
  if(slash_pos != std::string::npos)
    {
    Path = fn.substr(0, slash_pos);
    }
  else
    {
    Path = "";
    }

  return Path.c_str();
}


/**
 * Return file name of a full filename (i.e. file name without path).
 */
const char *Filename::GetName()
{
  std::string filename = FileName;
  assert( !filename.empty() );
#if defined(_WIN32)
  std::string::size_type slash_pos = filename.find_last_of("/\\");
#else
  std::string::size_type slash_pos = filename.find_last_of("/");
#endif
  if(slash_pos != std::string::npos)
    {
    return &FileName[0] + slash_pos + 1;
    }

  return &FileName[0];
}

const char *Filename::ToWindowsSlashes()
{
  Conversion = FileName;
  //assert( !Conversion.empty() );
  for (std::string::iterator it = Conversion.begin(); it != Conversion.end(); ++it )
    {
    if( *it == '/' )
      {
      *it = '\\';
      }
    }

  return Conversion.c_str();
}

// convert windows slashes to unix slashes
const char *Filename::ToUnixSlashes()
{
  Conversion = FileName;
  //std::string::size_type s = Conversion.find("\\");
  //assert( s == std::string::npos );
  assert( !Conversion.empty() );
  for (std::string::iterator it = Conversion.begin(); it != Conversion.end(); ++it )
    {
    if( *it == '\\' )
      {
      assert( it+1 == Conversion.end() || *(it+1) != ' ' ); // is it an escaped space ?
      *it = '/';
      }
    }

  return Conversion.c_str();
}

#if defined(_WIN32) && (defined(_MSC_VER) || defined(__WATCOMC__) || defined(__BORLANDC__) || defined(__MINGW32__))
#include <windows.h>

inline void Realpath(const char *path, std::string & resolved_path)
{
  char *ptemp;
  char fullpath[MAX_PATH];
  if( GetFullPathName(path, sizeof(fullpath), fullpath, &ptemp) )
    {
    Filename fn( fullpath );
    resolved_path = fn.ToUnixSlashes();
    }
  else
    {
    resolved_path = "";
    }
}
#else
/* The maximum length of a file name.  */
#if defined(PATH_MAX)
# define GDCM_FILENAME_MAXPATH PATH_MAX
#elif defined(MAXPATHLEN)
# define GDCM_FILENAME_MAXPATH MAXPATHLEN
#else
# define GDCM_FILENAME_MAXPATH 16384
#endif

inline void Realpath(const char *path, std::string & resolved_path)
{
  char resolved_name[GDCM_FILENAME_MAXPATH];

  char *ret = realpath(path, resolved_name);
  if( ret )
    {
    resolved_path = resolved_name;
    }
  else
    {
    resolved_path = "";
    }
}
#endif

const char *Filename::GetExtension()
{
  std::string name = GetName();
  std::string::size_type dot_pos = name.rfind(".");
  if(dot_pos != std::string::npos)
    {
    return GetName() + dot_pos;
    }

  return 0;
}


bool Filename::IsIdentical(Filename const &fn) const
{
  std::string realpath1;
  std::string realpath2;
  Realpath(FileName.c_str(), realpath1);
  Realpath(fn.GetFileName(), realpath2);
  gdcmDebugMacro( "IsIdentical: " << realpath1 << " vs " << realpath2 );
  return realpath1 == realpath2;
}

const char *Filename::Join(const char *path, const char *filename)
{
  static std::string s; // warning C4640: 's' : construction of local static object is not thread-safe
  s = path;
  s += '/';
  s += filename;
  return s.c_str();
}

bool Filename::EndWith(const char ending[]) const
{
  if( !ending ) return false;
  const char *str = FileName.c_str();
  size_t str_len = FileName.size();
  size_t ending_len = strlen(ending);

  if(ending_len > str_len)
    return false;

  return 0 == strncmp( str + str_len - ending_len, ending, ending_len );
}

} // end namespace gdcm
