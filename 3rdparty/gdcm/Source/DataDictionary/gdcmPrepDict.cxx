/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/* Executable meant to preprocess the DICOM V3 Dictionary
 * CMake regular expression are great but as far as cmake 2.2 is concerned there is
 * no math operation which makes the preprocess stage extremely difficult to write
 * Therefore we write our own C++ code that will do the same, then cmake will generate the
 * executable which will generate the source code (and all dependencies are properly set).
 */

#include "gdcmDictConverter.h"

int main(int argc, char *argv[])
{
  if( argc < 4 )
    return 1;

  // Can use GDCM_SOURCE_DIR
  // std::string path = GDCM_SOURCE_DIR;
  // path += "/Dicts/";
  // path += DictFilename;
  // std::cerr << "open: " << path << std::endl;
  const char *filename = argv[1]; // Full path to the dict
  const char *outfilename = argv[2]; // Full path to output the dict
  const char *dictname = argv[3]; // Full path to output the dict
  //std::cerr << "open: " << filename << std::endl;

  gdcm::DictConverter dc;
  //dc.SetOutputType(gdcm::DictConverter::DICT_DEBUG);
  dc.SetInputFileName(filename);
  dc.SetOutputFileName(outfilename);
  dc.SetDictName(dictname);
  dc.Convert();

  return 0;
}
