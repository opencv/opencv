/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSorter.h"
#include "gdcmTesting.h"

int TestSorter(int argc, char *argv[])
{
  // Black box:
  gdcm::Directory::FilenamesType fns;
  gdcm::Sorter s;
  // No sort function and fns is empty
  if( !s.Sort( fns ) )
    {
    return 1;
    }

  // White box:
  const char *directory = gdcm::Testing::GetDataRoot();
  if( argc == 2 )
    {
    directory = argv[1];
    }
  gdcm::Directory d;
  unsigned int nfiles = d.Load( directory ); // no recursion
  d.Print( std::cout );
  std::cout << "done retrieving file list. " << nfiles << " files found." <<  std::endl;

/* bool b = s.Sort( d.GetFilenames() );

  if( !b )
    {
    std::cerr << "Failed to sort:" << directory << std::endl;
    return 1;
    }

  std::cout << "Sorting succeeded:" << std::endl;
  s.Print( std::cout );
*/
  return 0;
}
