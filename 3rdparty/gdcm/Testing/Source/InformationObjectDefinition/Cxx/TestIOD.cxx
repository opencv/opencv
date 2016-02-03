/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIOD.h"
#include "gdcmDefs.h"
#include "gdcmGlobal.h"
#include "gdcmDicts.h"
#include "gdcmDict.h"
#include "gdcmType.h"
#include "gdcmIODs.h"

int TestIOD(int, char *[])
{
  using namespace gdcm;
  gdcm::Global& g = gdcm::Global::GetInstance();
  if( !g.LoadResourcesFiles() )
    {
    std::cerr << "Could not LoadResourcesFiles" << std::endl;
    return 1;
    }

  static const Defs &defs = g.GetDefs();
  static const gdcm::Dicts &dicts = g.GetDicts();
  static const IODs &iods = defs.GetIODs();
  static const gdcm::Dict &pubdict = dicts.GetPublicDict();

  //const IOD& iod = defs.GetIODFromFile(*F);

    IODs::IODMapTypeConstIterator it = iods.Begin();
    for( ; it != iods.End(); ++it )
      {
      const IODs::IODName &name = it->first;
      (void)name;
      const IOD &iod = it->second;

      gdcm::Dict::ConstIterator dictit = pubdict.Begin();
      for(; dictit != pubdict.End(); ++dictit)
        {
        const gdcm::Tag &tag = dictit->first;
        gdcm::Type t = iod.GetTypeFromTag(defs, tag);
        (void)t;
        //std::cout << t << std::endl;
        }
      }

  return 0;
}
