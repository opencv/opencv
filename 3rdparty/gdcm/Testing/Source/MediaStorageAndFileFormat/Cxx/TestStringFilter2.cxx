/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmStringFilter.h"
#include "gdcmReader.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmTesting.h"
#include "gdcmTrace.h"

static int TestStringFilt(const char *filename)
{
  gdcm::StringFilter sf;
  gdcm::Reader r;
  r.SetFileName( filename );
  if( !r.Read() )
    {
    return 1;
    }
  gdcm::DataSet const& ds = r.GetFile().GetDataSet();
  sf.SetFile( r.GetFile() );

  int ret = 0;
  gdcm::DataSet::ConstIterator it = ds.Begin();
  for( ; it != ds.End(); ++it)
    {
    const gdcm::DataElement &ref = *it;
    std::pair<std::string, std::string> s = sf.ToStringPair( ref );
    if( !s.second.empty() || ref.GetVL() == 0 )
      {
      std::cout << s.first << " -> " << s.second << std::endl;
      std::string s2 = sf.FromString( ref.GetTag(), s.second.c_str(), s.second.size() );
      //std::cout << s.first << " -> " << s2 << std::endl;
      }
    else if( !ref.GetByteValue() ) // It means it's a SQ
      {
      std::cout << "SQ:" << ref.GetTag() << std::endl;
      }
    else if( ref.GetTag().IsPrivate() )
      {
      //std::cout << "Private:" << ref.GetTag() << std::endl;
      std::string s2 = sf.FromString( ref.GetTag(), s.second.c_str(), s.second.size() );
      }
    else
      {
      std::cerr << "Not supported: " << ref << std::endl;
      //ret += 1;
      }
    }

  return ret;
}

int TestStringFilter2(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestStringFilt(filename);
    }

  // else
  // First of get rid of warning/debug message
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestStringFilt( filename );
    ++i;
    }

  return r;
}
