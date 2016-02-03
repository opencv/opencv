/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFilenameGenerator.h"
#include "gdcmTrace.h"

#include <cstring> // strchr
#include <stdio.h> // snprintf
#ifdef _WIN32
#define snprintf _snprintf
#endif

namespace gdcm
{

//-----------------------------------------------------------------------------
FilenameGenerator::SizeType FilenameGenerator::GetNumberOfFilenames() const
{
  return Filenames.size();
}

//-----------------------------------------------------------------------------
void FilenameGenerator::SetNumberOfFilenames(SizeType nfiles)
{
  Filenames.resize( nfiles );
}

//-----------------------------------------------------------------------------
const char * FilenameGenerator::GetFilename(SizeType n) const
{
  if( n < Filenames.size() )
    return Filenames[n].c_str();
  return NULL;
}

//-----------------------------------------------------------------------------
bool FilenameGenerator::Generate()
{
  if( Pattern.empty() && Prefix.empty() )
    {
    return false;
    }
  else if( Pattern.empty() && !Prefix.empty() ) // no pattern but a prefix
    {
    const SizeType numfiles = Filenames.size();
    for( SizeType i = 0; i < numfiles; ++i)
      {
      std::ostringstream os;
      os << Prefix;
      os << i;
      Filenames[i] = os.str();
      }
    return true;
    }
  else if( !Pattern.empty() )
    {
    std::string::size_type pat_len = Pattern.size();
    const SizeType padding = 10; // FIXME is this large enough for all cases ?
    const SizeType internal_len = pat_len + padding;
    const SizeType numfiles = Filenames.size();
    if( numfiles == 0 )
      {
      gdcmDebugMacro( "Need to specify the number of files" );
      // I am pretty sure this is an error:
      return false;
      }
    const char *pattern = Pattern.c_str();
    int num_percent = 0;
    while( (pattern = strchr( pattern, '%')) )
      {
      ++pattern;
      ++num_percent;
      }
    if ( num_percent != 1 )
      {
      // Bug: what if someone wants to output file such as %%%02 ... oh well
      gdcmDebugMacro( "No more than one % in string formating please" );
      return false;
      }
    bool success = true;
    char *internal = new char[internal_len];
    for( SizeType i = 0; i < numfiles && success; ++i)
      {
      int res = snprintf( internal, internal_len, Pattern.c_str(), i );
      assert( res >= 0 );
      success = (SizeType)res < internal_len;
      if( Pattern.empty() )
        {
        Filenames[i] = internal;
        }
      else
        {
        Filenames[i] = Prefix + internal;
        }
      //assert( Filenames[i].size() == res ); // upon success only
      }
    delete[] internal;
    if( !success )
      {
      Filenames.clear();
      // invalidate size too ??
      }
    return success;
    }
  return false;
}

} // namespace gdcm
