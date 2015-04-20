/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmScanner.h"
#include "gdcmDirectory.h"
#include "gdcmSystem.h"
#include "gdcmTesting.h"
#include "gdcmTrace.h"

// dcmdump /path/to/image/*.dcm 2>&/dev/null| grep 0020 | grep "000e\|000d" | sort | uniq
//
// $ find   /images/ -type f -exec dcmdump -s +P 0010,0010 {} \;

int TestScannerExtra()
{
  const char *extradataroot = gdcm::Testing::GetDataExtraRoot();
  if( !extradataroot )
    {
    return 1;
    }
  if( !gdcm::System::FileIsDirectory(extradataroot) )
    {
    std::cerr << "No such directory: " << extradataroot <<  std::endl;
    return 1;
    }

  gdcm::Directory d;
  unsigned int nfiles = d.Load( extradataroot, true ); // no recursion
  std::cout << "done retrieving file list. " << nfiles << " files found." <<  std::endl;

  gdcm::Scanner s;
  const gdcm::Tag t1(0x0020,0x000d); // Study Instance UID
  const gdcm::Tag t2(0x0020,0x000e); // Series Instance UID
  const gdcm::Tag t3(0x0010,0x0010); // Patient's Name
  const gdcm::Tag t4(0x0004,0x5678); // DUMMY element
  const gdcm::Tag t5(0x0028,0x0010); // Rows
  const gdcm::Tag t6(0x0028,0x0011); // Columns
  s.AddTag( t1 );
  s.AddTag( t2 );
  s.AddTag( t3 );
  s.AddTag( t4 );
  s.AddTag( t5 );
  s.AddTag( t6 );
  bool b = s.Scan( d.GetFilenames() );
  if( !b )
    {
    std::cerr << "Scanner failed" << std::endl;
    return 1;
    }
  //s.Print( std::cout );
  return 0;
}

int TestScanner1(int argc, char *argv[])
{
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  const char *directory = gdcm::Testing::GetDataRoot();
  if( argc == 2 )
    {
    directory = argv[1];
    }

  if( !gdcm::System::FileIsDirectory(directory) )
    {
    std::cerr << "No such directory: " << directory <<  std::endl;
    return 1;
    }

  gdcm::Directory d;
  unsigned int nfiles = d.Load( directory ); // no recursion
//  d.Print( std::cout );
  std::cout << "done retrieving file list. " << nfiles << " files found." <<  std::endl;

  gdcm::Scanner s;
  const gdcm::Tag t1(0x0020,0x000d); // Study Instance UID
  const gdcm::Tag t2(0x0020,0x000e); // Series Instance UID
  const gdcm::Tag t3(0x0010,0x0010); // Patient's Name
  const gdcm::Tag t4(0x0004,0x5678); // DUMMY element
  const gdcm::Tag t5(0x0028,0x0010); // Rows
  const gdcm::Tag t6(0x0028,0x0011); // Columns
  const gdcm::Tag t7(0x0008,0x0016); //
  s.AddTag( t1 );
  s.AddTag( t2 );
  s.AddTag( t3 );
  s.AddTag( t4 );
  s.AddTag( t5 );
  s.AddTag( t6 );
  s.AddTag( t7 );
  bool b = s.Scan( d.GetFilenames() );
  if( !b )
    {
    std::cerr << "Scanner failed" << std::endl;
    return 1;
    }
//  s.Print( std::cout );

  gdcm::Directory::FilenamesType const & files = s.GetFilenames();
  if( files != d.GetFilenames() )
    {
    return 1;
    }

  int numerrors = 0;
  for( gdcm::Directory::FilenamesType::const_iterator it = files.begin(); it != files.end(); ++it )
    {
    const char *filename = it->c_str();
    const char* value = s.GetValue(filename, t7);
    const char *msstr = gdcm::Testing::GetMediaStorageFromFile(filename);
    if( msstr && strcmp(msstr, "1.2.840.10008.1.3.10" ) == 0 )
      {
      // would need to check (0002,0002)...
      }
    if( value && msstr )
      {
      if( strcmp( value, msstr ) != 0 )
        {
        std::cerr << "Problem with file " << filename << std::endl;
        std::cerr << value << "/" << msstr << std::endl;
        ++numerrors;
        }
      }
    else
      {
      if ( !msstr )
        {
        std::cerr << "Problem with file " << filename << std::endl;
        if( value ) std::cerr << "Value found: " << value << std::endl;
        ++numerrors;
        }
      }
    }

  if( numerrors )
    return numerrors;

  // Check dummy filename:
  bool iskey = s.IsKey( "gdcm.rocks.invalid.name" );
  if( iskey )
    {
    std::cout << "IsKey returned: " << iskey << std::endl;
    return 1;
    }

  // Let's get the value for tag t1 in first file:
  gdcm::Scanner::MappingType const &mt = s.GetMappings();
  std::string sfilename;
  sfilename = gdcm::Testing::GetDataRoot();
  sfilename+= "/test.acr";
{
  //const char *filename = d.GetFilenames()[0].c_str();
  const char *filename = sfilename.c_str();
  // The following breaks with Papyrus file: PET-cardio-Multiframe-Papyrus.dcm
  unsigned int i = 0;
  gdcm::Scanner::MappingType::const_iterator it = mt.find(filename);
  assert( it != mt.end() );
  while( it == mt.end() )
    {
    ++i;
    if( i == d.GetFilenames().size() )
      {
      return 1;
      }
    filename = d.GetFilenames()[i].c_str();
    it = mt.find(filename);
    }
  std::cout << "Mapping for " << filename << " is :" << std::endl;
  const gdcm::Scanner::TagToValue &tv = it->second;
  //const std::string &filename = d.GetFilenames()[0];
  gdcm::Scanner::TagToValue::const_iterator it2 = tv.find( t5 );
  if( it2 == tv.end() || t5 != it2->first )
    {
    std::cerr << "Could not find tag:" << t5 << std::endl;
    return 1;
    }
  const char * t1value = it2->second;
  std::cout << filename << " -> " << t1 << " = " << (*t1value ? t1value : "none" ) << std::endl;
}

  const gdcm::Directory::FilenamesType &filenames = d.GetFilenames();
{
  gdcm::Directory::FilenamesType::const_iterator it = filenames.begin();
  for(; it != filenames.end(); ++it)
    {
    const char *filename = it->c_str();
    const gdcm::Tag &reftag = t6;
    const char *value =  s.GetValue( filename, reftag );
    if( value )
      {
      assert( value );
      std::cout << filename << " has " << reftag << " = " << value << std::endl;
      }
    else
      {
      std::cout << filename << " has " << reftag << " = no value or not DICOM" << std::endl;
      }
    }
}
/*
{
  std::vector<const char *> keys = s.GetKeys();
  for( std::vector<const char *>::const_iterator it = keys.begin(); it != keys.end(); ++it)
    {
    const char *filename = *it;
    const gdcm::Directory::FilenamesType::const_iterator it2
      = std::find(filenames.begin(), filenames.end(), filename);
    if( it2 == filenames.end() )
      {
      return 1;
      }
    if( !s.IsKey( filename ) )
      {
      return 1;
      }
    }
}
*/

  // puposely discard gdcmDataExtra test, this is just an 'extra' test...
  int b2 = TestScannerExtra(); (void)b2;


  return 0;
}
