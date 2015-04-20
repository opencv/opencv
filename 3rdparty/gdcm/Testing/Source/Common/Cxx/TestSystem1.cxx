/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSystem.h"
#include "gdcmTesting.h"
#include "gdcmFilename.h"
#include <iostream>
#include <fstream>

#include <string.h> // strlen

#include <time.h>

int TestGetTimeOfDay()
{
  time_t t = time(0);
  char date[22];
  if( !gdcm::System::GetCurrentDateTime(date) )
    {
    std::cerr << "Error" << std::endl;
    return 1;
    }
  char time_date[22];
  if( !gdcm::System::FormatDateTime(time_date, t) )
    {
    std::cerr << "Error" << std::endl;
    return 1;
    }
  //std::cout << date << std::endl;
  //std::cout << time_date << std::endl;

  if ( strncmp( date, time_date, strlen("20090511172802.") ) != 0 )
    {
    std::cerr << "Error" << std::endl;
    return 1;
    }
  return 0;
}

int TestSystem1(int, char *[])
{
  const char s1[] = "HELLO, wORLD !";
  const char s2[] = "Hello, World !";
  if( gdcm::System::StrCaseCmp(s1,s2) != 0 )
    {
    return 1;
    }
  if( gdcm::System::StrNCaseCmp(s1,s2, strlen(s1)) != 0 )
    {
    return 1;
    }
  const char s3[] = "Hello, World ! ";
  if( gdcm::System::StrCaseCmp(s1,s3) == 0 )
    {
    return 1;
    }
  if( gdcm::System::StrNCaseCmp(s1,s3, strlen(s1)) != 0 )
    {
    return 1;
    }
  if( gdcm::System::StrNCaseCmp(s1,s3, strlen(s3)) == 0 )
    {
    return 1;
    }

  // struct stat {
  // off_t         st_size;     /* total size, in bytes */
  // }

  //unsigned long size1 = sizeof(off_t);
  unsigned long size2 = sizeof(size_t);
  unsigned long size4 = sizeof(std::streamsize);
#if 0
  if( size1 > size2 )
    {
    std::cerr << "size_t is not appropriate on this system" << std::endl; // fails on some macosx
    return 1;
    }
  unsigned long size3 = sizeof(uintmax_t);
  if( size2 != size3 )
    {
    std::cerr << "size_t is diff from uintmax_t: " << size2 << " " << size3 << std::endl;
    return 1;
    }
#endif
  if( size2 != size4 )
    {
    std::cerr << "size_t is diff from std::streamsize: " << size2 << " " << size4 << std::endl;
    return 1;
    }

  char datetime[22];
  bool bres = gdcm::System::GetCurrentDateTime(datetime);
  if( !bres )
    {
    std::cerr << "bres" << std::endl;
    return 1;
    }
  assert( datetime[21] == 0 );
  std::cerr << datetime << std::endl;

  const char *cwd = gdcm::System::GetCWD();
  std::cerr << "cwd:" << cwd << std::endl;
  // GDCM_EXECUTABLE_OUTPUT_PATH "/../" "/Testing/Source/Common/Cxx"

  /*
   * I can do this kind of testing here since I know testing:
   * - cannot be installed (no rule in cmakelists)
   * - they cannot be moved around since cmake is not relocatable
   * thus this is safe to assume that current process directory is actually the executable output
   * path as computed by cmake:
   *
   * TODO: there can be trailing slash...
   */
  const char *path = gdcm::System::GetCurrentProcessFileName();
  if( !path )
    {
    std::cerr << "Missing implemnetation for GetCurrentProcessFileName" << std::endl;
    return 1;
    }
  gdcm::Filename fn( path );
//std::cerr << path << std::endl;
  if( strncmp(GDCM_EXECUTABLE_OUTPUT_PATH, fn.GetPath(), strlen(GDCM_EXECUTABLE_OUTPUT_PATH)) != 0 )
    {
    std::cerr << GDCM_EXECUTABLE_OUTPUT_PATH << " != " << fn.GetPath() << std::endl;
    gdcm::Filename fn_debug1( GDCM_EXECUTABLE_OUTPUT_PATH );
    gdcm::Filename fn_debug2( fn.GetPath() );
    std::cerr << fn_debug1.GetFileName() << " , " << fn_debug2.GetFileName() << std::endl;
    std::cerr << std::boolalpha << fn_debug1.IsIdentical( fn_debug2 ) << std::endl;
    return 1;
    }
  // gdcmCommonTests
  const char exename[] = "gdcmCommonTests";
  if( strncmp(exename, fn.GetName(), strlen(exename)) != 0 )
    {
    std::cerr << exename << " != " << fn.GetName() << std::endl;
    return 1;
    }

{
  char date[22];
  if( !gdcm::System::GetCurrentDateTime( date ) )
    {
    std::cerr << "GetCurrentDateTime: " << date << std::endl;
    return 1;
    }
  assert( date[21] == 0 );
  time_t timep; long milliseconds;
  if( !gdcm::System::ParseDateTime(timep, milliseconds, date) )
    {
    std::cerr << "Could not re-parse: " << date << std::endl;
    return 1;
    }
  char date2[22];
  if( !gdcm::System::FormatDateTime(date2, timep, milliseconds) )
    {
    return 1;
    }
  assert( date2[21] == 0 );

  if( strcmp( date, date2 ) != 0 )
    {
    std::cerr << "date1=" << date << std::endl;
    std::cerr << "date2=" << date2 << std::endl;
    return 1;
    }
}
  // Skip millisecond this time:
{
  char date[22+1];
  if( !gdcm::System::GetCurrentDateTime( date ) )
    {
    std::cerr << "GetCurrentDateTime: " << date << std::endl;
    return 1;
    }
  date[14] = 0;
  std::cout << date << std::endl;
  time_t timep;
  if( !gdcm::System::ParseDateTime(timep, date) )
    {
    std::cerr << "ParseDateTime" << std::endl;
    return 1;
    }
  char date2[22+1];
  date2[22] = 0;
  if( !gdcm::System::FormatDateTime(date2, timep) )
    {
    std::cerr << "FormatDateTime" << std::endl;
    return 1;
    }

  // FormatDateTime always print millisecond, only compare the date up to the millisecond:
  if( strncmp( date, date2, strlen(date) ) != 0 )
    {
    std::cerr << "date1=" << date << std::endl;
    std::cerr << "date2=" << date2 << std::endl;
    return 1;
    }
}

  // Check some valid date
{
  // YYYYMMDDHHMMSS.FFFFFF&ZZXX
  static const char *dates[] = {
    "2001",
    "200101",
    "20010102",
    "2001010203",
    "200101020304",
    "20010102030405",
    "20010102030405",
    "20010102030405.01",
    "20010102030405.0101",
    "20010102030405.010101",
  };
  for(int i = 0; i < 10; ++i )
    {
    const char *date = dates[i];
    time_t timep; long milliseconds;
    if( !gdcm::System::ParseDateTime(timep, milliseconds, date) )
      {
      std::cerr << "Should accept: " << date << std::endl;
      return 1;
      }
    }
}
  // Check some invalid date
{
  // YYYYMMDDHHMMSS.FFFFFF&ZZXX
  static const char *dates[] = {
    "200",
    "200121",
    "20010142",
    "2001010233",
    "200101020374",
    "20010102030485",
    "20010102030405.",
    "20010102030405-0000000",
    "20010102030405.0000001",
    "20010102030405.0000001",
  };
  for(int i = 0; i < 10; ++i )
    {
    const char *date = dates[i];
    time_t timep; long milliseconds;
    if( gdcm::System::ParseDateTime(timep, milliseconds, date) )
      {
      char buffer[22];
      gdcm::System::FormatDateTime(buffer, timep, milliseconds);
      std::cerr << "Should not accept: " << date << std::endl;
      std::cerr << "rendered as: " << buffer << std::endl;
      return 1;
      }
    }
}

  //const char long_str8[] = " 0123456789";
  //long l = 0;
  //int n = sscanf( long_str8, "%8ld", &l );
  //std::cout << "Long:" << l << std::endl;

  char hostname[255+1];
  hostname[255] = 0;
  if( gdcm::System::GetHostName( hostname ) )
    {
    std::cout << "Host:" <<  hostname << std::endl;
    }
  else
  {
  return 1;
  }

  //time_t t = gdcm::System::FileTime("/etc/debian_version");
  //char date3[22];
  //gdcm::System::FormatDateTime(date3, t);
  //std::cout << date3 << std::endl;

  const char fixed_date[] = "20090428172557.515500";
  if( strlen( fixed_date ) != 21 )
  {
    std::cerr << "fixed_date" << std::endl;
    return 1;
  }
  time_t fixed_timep;
  long fixed_milliseconds;
  if( !gdcm::System::ParseDateTime(fixed_timep, fixed_milliseconds, fixed_date) )
  {
    std::cerr << "ParseDateTime" << std::endl;
  return 1;
  }
  if( fixed_milliseconds != 515500 )
{
    std::cerr << "fixed_milliseconds" << std::endl;
return 1;
}
  char fixed_date2[22];
  if( !gdcm::System::FormatDateTime(fixed_date2, fixed_timep, fixed_milliseconds) )
{
    std::cerr << "FormatDateTime" << std::endl;
  return 1;
}
assert( fixed_date2[21] == 0 );
  if( strcmp( fixed_date, fixed_date2 ) != 0 )
{
    std::cerr << "fixed_date | fixed_date2" << std::endl;
  return 1;
}

  const char invalid_date1[] = "20090428172557.";
if( gdcm::System::ParseDateTime(fixed_timep, fixed_milliseconds, invalid_date1) )
{
std::cerr << "should reject:" << invalid_date1 << std::endl;
return 1;
}
  const char invalid_date2[] = "200";
if( gdcm::System::ParseDateTime(fixed_timep, fixed_milliseconds, invalid_date2) )
{
std::cerr << "should reject:" << invalid_date2 << std::endl;
return 1;
}
//  const char invalid_date3[] = "17890714172557";
//if( gdcm::System::ParseDateTime(fixed_timep, fixed_milliseconds, invalid_date3) )
//{
//std::cerr << "should reject:" << invalid_date3 << std::endl;
//char buffer[22];
//gdcm::System::FormatDateTime( buffer, fixed_timep, fixed_milliseconds );
//std::cerr << "Found" <<  buffer << std::endl;
//return 1;
//}
//  const char invalid_date4[] = "19891714172557";
//if( gdcm::System::ParseDateTime(fixed_timep, fixed_milliseconds, invalid_date4) )
//{
//std::cerr << "should reject:" << invalid_date4 << std::endl;
//char buffer[22];
//gdcm::System::FormatDateTime( buffer, fixed_timep, fixed_milliseconds );
//std::cerr << "Found" <<  buffer << std::endl;
//
//return 1;
//}
//  const char invalid_date5[] = "19890014172557";
//if( gdcm::System::ParseDateTime(fixed_timep, fixed_milliseconds, invalid_date5) )
//{
//std::cerr << "should reject:" << invalid_date5 << std::endl;
//char buffer[22];
//gdcm::System::FormatDateTime( buffer, fixed_timep, fixed_milliseconds );
//std::cerr << "Found" <<  buffer << std::endl;
//
//return 1;
//}
  const char valid_date1[] = "19890714172557";
if( !gdcm::System::ParseDateTime(fixed_timep, fixed_milliseconds, valid_date1) )
{
std::cerr << "should accept:" << valid_date1 << std::endl;
return 1;
}
  int res = 0;
  res +=  TestGetTimeOfDay();

  const char * testfilesize = gdcm::Testing::GetTempFilename( "filesize.bin" );
if( gdcm::System::FileExists( testfilesize ) )
{
  gdcm::System::RemoveFile(testfilesize);
}

  size_t ss1 = gdcm::System::FileSize( testfilesize );
if( ss1 != 0 )
{
std::cerr << "found:" << ss1 << std::endl;
  ++res;
}

  std::ofstream os( testfilesize, std::ios::binary );
  const char coucou[] = "coucou";
  os << coucou;
  os.flush();
  os.close();

  size_t ss2 = gdcm::System::FileSize( testfilesize );
  if( ss2 != strlen( coucou ) )
{
std::cerr << "found:" << ss2 << std::endl;
  res++;
}


  const char *codeset = gdcm::System::GetLocaleCharset();
if( !codeset )
{
std::cerr << "Could nto find Charset on your system. Please report." << std::endl;
res++;
}

  return res;
}
