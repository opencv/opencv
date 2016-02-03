/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <sstream>
#include <fstream>
#include <iostream>

// fstat
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// open
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// mmap
#include <sys/mman.h>

#include "gdcmFile.h"
#include "gdcmObject.h"
#include "gdcmDataSet.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmSmartPointer.h"
#include "gdcmDeflateStream.h"
#include "gdcmDumper.h"
#include "gdcmDirectory.h"
#include "gdcmSystem.h"


int DoOperation(std::string const & path)
{
  //std::cout << path << "\n";
  std::ifstream is( path.c_str(), std::ios::binary );
  is.seekg(0, std::ios::end );
  std::streampos size = is.tellg();
  is.seekg(0,std::ios::beg);
  std::vector<char> buffer;
  buffer.resize(size);
  is.read(&buffer[0], size );
  char k = buffer[(size_t)size-1]; // at least read one char to avoid compiler optimization
  is.close();

  return 0;
}

int main(int argc, char *argv[])
{
  if( argc < 2 )
    {
    return 1;
    }
  std::string filename = argv[1];

  int res = 0;
  if( gdcm::System::FileIsDirectory( filename.c_str() ) )
    {
    gdcm::Directory d;
    d.Load(filename);
    gdcm::Directory::FilenamesType const &filenames = d.GetFilenames();
    for( gdcm::Directory::FilenamesType::const_iterator it = filenames.begin(); it != filenames.end(); ++it )
      {
      res += DoOperation(*it);
      }
    }
  else
    {
    res += DoOperation(filename);
    }

  return res;
}
