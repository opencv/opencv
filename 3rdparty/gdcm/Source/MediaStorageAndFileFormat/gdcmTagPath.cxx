/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTagPath.h"
#include "gdcmTag.h"

#include <vector>
#include <string.h> // strlen
#include <stdlib.h> // abort
#include <stdio.h> // sscanf

namespace gdcm
{

/*
 * Implementation detail: a tag path is simply a vector<Tag>.
 * with the following convention:
 * First tag is a valid tag,
 * Second tag is an item number (or 0)
 * Third tag is a valid tag
 * ...
 * and so on an so forth
 */

TagPath::TagPath():Path()
{
}

TagPath::~TagPath()
{
}

void TagPath::Print(std::ostream &os) const
{
  unsigned int flip = 0;
  std::vector<Tag>::const_iterator it = Path.begin();
  for(; it != Path.end(); ++it)
    {
    if( flip % 2 == 0 )
      {
      os << *it << "/";
      }
    else // item number
      {
      // assert( it->GetElementTag() < 255 ); // how many item max can we have ?
      os << it->GetElementTag() << "/";
      }
    ++flip;
    }
}

bool TagPath::IsValid(const char *path)
{
  TagPath tp;
  return tp.ConstructFromString(path);
}

bool TagPath::ConstructFromTagList(Tag const *l, unsigned int n)
{
  //Path = std::vector<Tag>(l,l+n);
  Path.clear();
  for(unsigned int i = 0; i < n; ++i)
    {
    Path.push_back( l[i] );
    if( i+1 < n )
      {
      Path.push_back( 0 );
      }
    }
  return true;
}

bool TagPath::ConstructFromString(const char *path)
{
  Path.clear();
  if(!path) return false;
  unsigned int flip = 0;
  size_t pos = 0;
  const size_t len = strlen(path);
  if(!len) return false;
  // Need to start with a /
  if( path[pos] == '/' )
    {
    ++pos;
    }
  else
    {
    return false;
    }
  while( pos != len )
    {
    Tag t;
    if( flip % 2 == 0 )
      {
      if( t.ReadFromCommaSeparatedString( path+pos ) )
        {
        pos += 4 + 4 + 1;
        Path.push_back( t );
        }
      else
        {
        return false;
        }
      }
    else
      {
      unsigned int value = 0;
      if( path[pos] == '*' )
        {
        t.SetElementTag( 0 );
        pos++;
        Path.push_back( t );
        }
      else if( sscanf(path+pos, "%d/", &value) == 1 )
        {
        }
      }
    ++flip;
    if( pos != len && path[pos] == '/' ) ++pos;
    //else assert(0);
    }
  return true;
}

bool TagPath::Push(Tag const & t)
{
  if( Path.size() % 2 == 0 )
    {
    Path.push_back( t );
    return true;
    }
  return false;
}

bool TagPath::Push(unsigned int itemnum)
{
  if( Path.size() % 2 == 1 )
    {
    Path.push_back( itemnum );
    return true;
    }
  return false;
}

}
