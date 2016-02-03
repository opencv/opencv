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
#include "gdcmDict.h"
#include "gdcmDefs.h"

int TestGlobal(int, char *[])
{
  // case 1
  // Get the global singleton:
  gdcm::Trace::DebugOn();
  gdcm::Global& g = gdcm::Global::GetInstance();
  if( !g.LoadResourcesFiles() )
    {
    std::cerr << "Could not LoadResourcesFiles" << std::endl;
    return 1;
    }
  gdcm::Trace::DebugOff();
  // get the Part 6 dicts from it:
  const gdcm::Dicts &ds = g.GetDicts();
  const gdcm::Dict &pub = ds.GetPublicDict();
  const gdcm::PrivateDict &priv = ds.GetPrivateDict();

  // case 2
  gdcm::Dicts dicts;
  const gdcm::Dict &d1 = dicts.GetPublicDict();

  // case 3
  gdcm::Dict d2;

  // This one will be empty:
  std::cout << "Empty dict:" << std::endl;
  std::cout << d1 << std::endl;
  if( !d1.IsEmpty() )
    {
    return 1;
    }
  // This one will be empty:
  std::cout << "Empty dict:" << std::endl;
  std::cout << d2 << std::endl;
  if( !d2.IsEmpty() )
    {
    return 1;
    }
  // This should should be filled in:
  std::cout << "Global pub dict:" << std::endl;
  //std::cout << pub << std::endl;
  if( pub.IsEmpty() )
    {
    return 1;
    }
  // This should should be filled in:
  std::cout << "Global priv dict:" << std::endl;
  //std::cout << priv << std::endl;
  if( priv.IsEmpty() )
    {
    return 1;
    }
#if 0
  // FIXME I do not understand what was wrong before...
  // I had to change the PrivateTag to use lower case to support this:
  const gdcm::DictEntry& de1 = priv.GetDictEntry( gdcm::PrivateTag(0x2001,0x0001,"Philips Imaging DD 001") );
  std::cout << de1 << std::endl;

  const gdcm::DictEntry& de2 = priv.GetDictEntry( gdcm::PrivateTag(0x2001,0x0001,"PHILIPS IMAGING DD 001") );
  std::cout << de2 << std::endl;
  if( &de1 != &de2 )
    {
    return 1;
    }

  const gdcm::DictEntry& de3 = priv.GetDictEntry( gdcm::PrivateTag(0x2001,0x0003,"Philips Imaging DD 001") );
  std::cout << de3 << std::endl;

  const gdcm::DictEntry& de4 = priv.GetDictEntry( gdcm::PrivateTag(0x2001,0x0003,"PHILIPS IMAGING DD 001") );
  std::cout << de4 << std::endl;
  if( &de4 != &de3 )
    {
    return 1;
    }
#endif

#if 0
  const char *empty = "";
  std::string s = empty;
  std::cout << s.empty() << std::endl;
  const gdcm::DictEntry& de = pub.GetDictEntry( gdcm::Tag(0x0028,0x0015) );
  const char *v = de.GetName();
  //assert( v );
  std::cout << "TOTO:" << de << std::endl;
#endif

  const gdcm::Defs &defs = g.GetDefs();

  const gdcm::Modules &modules = defs.GetModules();
  std::cout << modules << std::endl;

  const gdcm::Macros &macros = defs.GetMacros();
  std::cout << macros << std::endl;

  const gdcm::IODs &iods = defs.GetIODs();
  std::cout << iods << std::endl;


  return 0;
}
