/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 * Executable meant to preprocess a GroupName dictionary, entry should look like this:
 * 0000 CMD Command
 * ...
 */

#include <iostream>
#include <fstream>
#include <sstream>  // for std::getline
#include <assert.h>
#include <stdio.h>
#include <iomanip> // std::setw

void write_header(std::ofstream &of)
{
  of << "#include \"gdcmTypes.h\"\n"
    "#include \"gdcmGroupDict.h\"\n\n"
    "namespace gdcm {\n\n"
    "typedef struct\n{\n"
    "  uint16_t group;\n"
    "  const char *abbreviation;\n"
    "  const char *name;\n"
    "} GROUP_ENTRY;\n\n"
    "static GROUP_ENTRY groupname[] = {\n";
}

void write_footer(std::ofstream &of)
{
  of << "\t{0,0,0} // will not be added to the dict \n"
    "};\n\n"
    "void GroupDict::FillDefaultGroupName()\n"
    "{\n"
    "  unsigned int i = 0;\n"
    "  GROUP_ENTRY n = groupname[i];\n"
    "  while( n.name != 0 )\n"
    "  {\n"
    "    Insert( n.group, n.abbreviation, n.name );\n"
    "    n = groupname[++i];\n"
    "  }\n"
    "}\n\n"
    "} // namespace gdcm\n";
}

bool check_abbr(std::string &abbr)
{
  std::string::const_iterator it = abbr.begin();
  for(; it != abbr.end(); ++it)
    {
    if ( *it < 'A' || *it > 'Z' ) return false;
    }
  return true;
}

int main(int argc, char *argv[])
{
  if( argc < 3 )
    return 1;

  const char *filename = argv[1]; // Full path to the dict
  const char *outfilename = argv[2]; // Full path to output the dict
  //std::cerr << "open: " << filename << std::endl;
  std::ifstream from(filename, std::ios::binary);
  std::ofstream into(outfilename,std::ios::binary);
  if(!from)
    {
    std::cerr << "Problem opening the from file" << std::endl;
    return 1;
    }
  if(!into)
    {
    std::cerr << "Problem opening the into file" << std::endl;
    return 1;
    }

  write_header(into);
  int error = 0;
  std::string line;
  while(std::getline(from, line))
    {
     if( !line.empty() )
       {
       std::string::iterator e(line.end()-1);
       if( *e == '\r' ) line.erase(e);
       }
    unsigned int group; // Group Number
    std::string abbr; // NHI Abbreviation (when known) - not part of DICOM standard -
    std::string meaning; // Meaning          (when known) - not part of DICOM standard -
    std::istringstream is(line);
    is >> std::hex >> group;
    if ( group > 0xffff)
      return 1;
    is >> abbr;
    if( !check_abbr(abbr) )
      return 1;
    // skip any whitespace before calling getline
    is >> std::ws;
    // get all the remaining characters
    std::getline(is,meaning);
    into << "\t{0x" << std::hex << std::setw(4) << std::setfill('0') << group << ",\"" << abbr << "\",\"" << meaning << "\"},\n";
    }
  write_footer(into);

  from.close();
  into.close();

  return error;
}
