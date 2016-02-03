/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDicts.h"
#include "gdcmGlobal.h"
#include "gdcmDict.h"

#include <set>

int TestDicts(int, char *[])
{
  gdcm::Dicts dicts;
  //const gdcm::Dict &d = dicts.GetPublicDict();
  //std::cout << d << std::endl;

  const gdcm::Global& g = gdcm::GlobalInstance;
  // get the Part 6 dicts from it:
  const gdcm::Dicts &ds = g.GetDicts();
  const gdcm::Dict &pub = ds.GetPublicDict();
  gdcm::Dict::ConstIterator it = pub.Begin();
  int ret = 0;
  std::set<std::string> names;
  for( ; it != pub.End(); ++it)
    {
    const gdcm::Tag &t = it->first;
    const gdcm::DictEntry &de = it->second;
    // A couple of tests:
    if( t.GetElement() == 0x0 )
      {
      // Check group length
      if( de.GetVR() != gdcm::VR::UL || de.GetVM() != gdcm::VM::VM1 )
        {
        std::cerr << "Group length issue: Problem with tag: " << t << " " << de << std::endl;
        ++ret;
        }
      }
    // I need a test that check there is no duplicate name for data elements since python-gdcm
    // will rely on it
    if( names.count( de.GetName() ) != 0 )
      {
      //std::cerr << "Name issue: Problem with tag: " << t << " " << de << std::endl;
      //++ret;
      }
    names.insert( de.GetName() );
    }

  return ret;
}
