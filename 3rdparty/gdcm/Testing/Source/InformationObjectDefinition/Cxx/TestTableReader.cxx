/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTableReader.h"
#include "gdcmModules.h"

// generated file:
#include "gdcmTables.h"

void TestReadTable(const char *filename)
{
  gdcm::Defs defs;
  gdcm::TableReader tr(defs);
  tr.SetFilename(filename);
  tr.Read();


  const gdcm::Modules &modules = defs.GetModules();
  std::cout << modules << std::endl;

  const gdcm::Macros &macros = defs.GetMacros();
  std::cout << macros << std::endl;

  const gdcm::IODs &iods = defs.GetIODs();
  std::cout << iods << std::endl;
}

int TestTableReader(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    TestReadTable(filename);
    return 0;
    }

  // else
  int i = 0;
  const char *filename;
  while( (filename = gdcmTables[i]) )
    {
    TestReadTable( filename );
    ++i;
    }

  return 0;
}
