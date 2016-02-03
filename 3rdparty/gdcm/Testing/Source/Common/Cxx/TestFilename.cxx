/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmTesting.h"

#include <iostream>
#include <fstream>
#include <cstdlib> // EXIT_FAILURE

/*!
 * \test TestFilename
 * bla coucou
 */
int TestFilename(int argc, char *argv[])
{
  (void)argc;(void)argv;
  std::string path = "/gdcm/is/a/dicom/";
  std::string name = "library.dcm";
  std::string fullpath = path;
  fullpath += '/';
  fullpath += name;
  gdcm::Filename f( fullpath.c_str() );
  std::cout << f.GetPath() << std::endl;
  std::cout << f.GetName() << std::endl;
  std::cout << f.GetExtension() << std::endl;
  std::cout << f << std::endl;

  if( f.GetPath() != path )
    {
    std::cerr << "Wrong path" << std::endl;
    return 1;
    }
  if( f.GetName() != name)
    {
    std::cerr << "Wrong name" << std::endl;
    return 1;
    }
  if( f.GetExtension() != std::string( ".dcm" ) )
    {
    std::cerr << "Wrong extension" << std::endl;
    return 1;
    }
//  if( std::string( "/tmp/debug.dcm" ) != f )
//    {
//    return 1;
//    }

  std::string dataroot = gdcm::Testing::GetDataRoot();
  std::string current = dataroot +  "/test.acr";
  if( !gdcm::System::FileExists( current.c_str() ) )
    {
    std::cerr << "File does not exist: " << current << std::endl;
    return 1;
    }
  std::cerr << "Current:" << current << std::endl;
  gdcm::Filename fn0(current.c_str());
  std::cerr << fn0.GetPath() << std::endl;
  std::string current2 = fn0.GetPath();
  current2 += "/./";
  current2 += fn0.GetName();
  std::cerr << current2 << std::endl;
  if( current2 == current )
    {
    return 1;
    }
  gdcm::Filename fn2(current2.c_str());
  if( !fn0.IsIdentical(fn2))
    {
    return 1;
    }

  {
  const char *curprocfn = gdcm::System::GetCurrentProcessFileName();
  if( curprocfn )
    {
    gdcm::Filename fn( curprocfn );
    std::string str = fn.GetPath();
    std::cout << str << std::endl;
    }
  }

{
#ifdef GDCM_HAVE_WCHAR_IFSTREAM
  const wchar_t ifn[] = L"UnicodeFileName.dcm";
  const wchar_t* fn = gdcm::Testing::GetTempFilenameW(ifn);
  std::ofstream outputFileStream( fn );
  if ( ! outputFileStream.is_open() )
    {
    std::cerr << "Failed to read UTF-16: " << fn << std::endl;
    return EXIT_FAILURE;
    }
  outputFileStream.close();
#else
  //char ifn2[] = "α.dcm"; //MM: I do not think this is legal C++...
  const char ifn2[] = "\xCE\xB1.dcm"; // this is the proper way to write it (portable)
  const char ifn1[] = {
  (char)0xCE,
  (char)0xB1,
  '.',
  'd',
  'c',
  'm',
  0}; // trailing NULL char
  std::string sfn1 = gdcm::Testing::GetTempFilename(ifn1);
  const char *csfn1 = sfn1.c_str();
  std::string sfn2 = gdcm::Testing::GetTempFilename(ifn2);
  const char *csfn2 = sfn2.c_str();
  std::ofstream outputFileStream( csfn1 );
  if ( ! outputFileStream.is_open() )
    {
    std::cerr << "Failed to create UTF-8 file: " << csfn1 << std::endl;
    return EXIT_FAILURE;
    }
  const char secret[]= "My_secret_pass_phrase";
  outputFileStream << secret;
  outputFileStream.close();
  if( !gdcm::System::FileExists(csfn1) )
    {
    std::cerr << "File does not exist: " << csfn1 << std::endl;
    return EXIT_FAILURE;
    }

  // Now open version 2 (different encoding)
  std::ifstream inputFileStream( csfn2 );
  if ( ! inputFileStream.is_open() )
    {
    std::cerr << "Failed to open UTF-8 file: " << csfn2 << std::endl;
    return EXIT_FAILURE;
    }
  std::string ssecret;
  inputFileStream >> ssecret;
  inputFileStream.close();
  if( ssecret != secret )
    {
    std::cerr << "Found: " << ssecret << " should have been " << secret << std::endl;
    return EXIT_FAILURE;
    }

  if( !gdcm::System::RemoveFile(csfn1) )
    {
    std::cerr << "Could not remvoe #1: " << csfn1 << std::endl;
    return EXIT_FAILURE;
    }
  // cannot remove twice the same file:
  if( gdcm::System::RemoveFile(csfn2) )
    {
    std::cerr << "Could remvoe #2 a second time...seriously " << csfn2 << std::endl;
    return EXIT_FAILURE;
    }
#endif
}

{

//#define TESTLONGPATHNAMES
#ifdef TESTLONGPATHNAMES
	//why are we testing this?  This is the operating system's deal, not GDCM's.
	//GDCM is not responsible for long path names, and cannot fix this issue.
	//if we want to move this to a configuration option (ie, test for long pathnames),
	//then we can--otherwise, windows users should just beware of this issue.
	//This path limit has been the case since Windows 95, and is unlikely to change any time soon.

  // Apparently there is an issue with long pathanem i nWin32 system:
  // http://msdn.microsoft.com/en-us/library/aa365247(VS.85).aspx#maxpath
  // The only way to work around the 260 byte limitation it appears as if we
  // have to deal with universal naming convention (UNC) path.
  const char subdir[] =
    "very/long/pathname/foobar/hello_world/toreproduceabugindpkg/pleaseconsider/"
    "very/long/pathname/foobar/hello_world/toreproduceabugindpkg/pleaseconsider/"
    "very/long/pathname/foobar/hello_world/toreproduceabugindpkg/pleaseconsider/"
    "very/long/pathname/foobar/hello_world/toreproduceabugindpkg/pleaseconsider/";
  const char *directory_ = gdcm::Testing::GetTempDirectory(subdir);
#ifdef _WIN32
  gdcm::Filename mydir( directory_ );
  std::string unc = "\\\\?\\";//on Windows, to use UNC, you need to:
  //a) append this string
  //b) use a network drive (ie, the gdcm file is made on a network drive) that
  //c) you have access to.
  //I don't think this is a good or useful test. mmr
  unc += mydir.ToWindowsSlashes();
  const char *directory = unc.c_str();
#else
  const char *directory = directory_;
#endif
  if( !gdcm::System::MakeDirectory(directory))
    {
    std::cerr << "Failed to create directory with long path: " << directory << std::endl;
    return EXIT_FAILURE;
    }
  std::string sfn = gdcm::Testing::GetTempFilename( "dummy.dcm", subdir );
  std::cerr << "Long path is: " << sfn.size() << std::endl;
  std::cerr << "Long path is: " << sfn << std::endl;
  if( sfn.size() > 260 )
    {
    const char *fn = sfn.c_str();
    // Should demontrate the issue
  std::ofstream outputFileStream( fn );
  if ( ! outputFileStream.is_open() )
    {
    std::cerr << "Failed to create file with long path: " << fn << std::endl;
    return EXIT_FAILURE;
    }
  outputFileStream.close();
  if( !gdcm::System::FileExists(fn) )
    {
    std::cerr << "File does not exist: " << fn << std::endl;
    return EXIT_FAILURE;
    }
  if( !gdcm::System::RemoveFile(fn) )
    {
    std::cerr << "Could not remvoe: " << fn << std::endl;
    return EXIT_FAILURE;
    }

  }
else
{
    std::cerr << "seriously " << fn << std::endl;
    return EXIT_FAILURE;
}
#endif //TESTLONGPATHNAMES
}

  return 0;
}
