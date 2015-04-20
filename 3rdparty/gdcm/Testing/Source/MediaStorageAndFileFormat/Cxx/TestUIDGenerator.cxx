/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmUIDGenerator.h"

#include <iostream>
#include <string>
#include <set>

#include <string.h>

int TestUIDGeneratorValid()
{
  gdcm::UIDGenerator uid;
  uid.SetRoot( "1.2.3.4.0.0.1" );
  const char *s = uid.Generate();
  if( !gdcm::UIDGenerator::IsValid( s ) )
    {
    return 1;
    }
  const char invalid0[] = ".0.123";
  if( gdcm::UIDGenerator::IsValid( invalid0 ) )
    {
    return 1;
    }
   const char invalid1[] = "abcd";
  if( gdcm::UIDGenerator::IsValid( invalid1 ) )
    {
    return 1;
    }
  const char invalid2[] = "1.2.3.4.0.0.123.a";
  if( gdcm::UIDGenerator::IsValid( invalid2 ) )
    {
    return 1;
    }
  const char invalid3[] = "1.2.3.4.0.0.123..";
  if( gdcm::UIDGenerator::IsValid( invalid3 ) )
    {
    return 1;
    }
  const char invalid4[] = "1.2.3.4.0.0..123";
  if( gdcm::UIDGenerator::IsValid( invalid4 ) )
    {
    return 1;
    }
  const char invalid5[] = "1.2.3.4.00.123";
  if( gdcm::UIDGenerator::IsValid( invalid5 ) )
    {
    return 1;
    }
  const char invalid6[] = "1.2.3.4.00.123.";
  if( gdcm::UIDGenerator::IsValid( invalid6 ) )
    {
    return 1;
    }
  const char invalid7[] = "1234567890.1234567890.1234567890.1234567890.1234567890.1234567890";
  if( gdcm::UIDGenerator::IsValid( invalid7 ) )
    {
    return 1;
    }
  const char invalid8[] = "1234567890.1234567890.1234567890.1234567890.1234567890/123456789";
  if( gdcm::UIDGenerator::IsValid( invalid8 ) )
    {
    return 1;
    }
  const char invalid9[] = "";
  if( gdcm::UIDGenerator::IsValid( invalid9 ) )
    {
    return 1;
    }
  const char invalid10[] = ".";
  if( gdcm::UIDGenerator::IsValid( invalid10 ) )
    {
    return 1;
    }
  return 0; // no error
}

int TestUIDGenerator(int , char *[])
{
  gdcm::UIDGenerator uid;
  std::cout << gdcm::UIDGenerator::GetGDCMUID() << std::endl;
  std::cout << uid.GetRoot() << std::endl;
  if( strcmp( gdcm::UIDGenerator::GetGDCMUID(), uid.GetRoot() ) != 0 )
    {
    return 1;
    }
  /*
   * Purposely take a very long root, to test the robustness of the generator
   * since we are left with fewer bytes to still generate uniq UID
   */
  // let's test 27 bytes root:
  const char myroot[] = "9876543210.9876543210.98765"; // 26 bytes is the length of GDCM root
  //if( strlen(myroot) != 26 )
  //  {
  //  return 1;
  //  }
  uid.SetRoot( myroot );
  std::cerr << "before generate" << std::endl;
  const char *s = uid.Generate();
  std::cerr << "after generate" << std::endl;
  std::cout << "s:" << s << std::endl;
  if( strcmp( myroot, uid.GetRoot() ) != 0 )
    {
    std::cerr << "1 failed" << std::endl;
    return 1;
    }
  if( strcmp( gdcm::UIDGenerator::GetGDCMUID(), myroot ) == 0 )
    {
    std::cerr << "2 failed" << std::endl;
    return 1;
    }
  if( strncmp( s, uid.GetRoot(), strlen( uid.GetRoot() ) ) != 0 )
    {
    std::cerr << "3 failed" << std::endl;
    return 1;
    }

/*
  std::string s0 = "123456";
  std::cout << (s0.c_str() + s0.find_first_not_of('0')) << std::endl;
  std::string s1 = "0123456";
  std::cout << (s1.c_str() + s1.find_first_not_of('0')) << std::endl;
  std::string s2 = "00123456";
  std::cout << (s2.c_str() + s2.find_first_not_of('0')) << std::endl;
  std::string s3 = "000";
  if( s3.find_first_not_of('0') != std::string::npos )
    std::cout << (s3.c_str() + s3.find_first_not_of('0')) << std::endl;
*/

  // Threading issue, make sure that two different UIDs cannot generate same UID
  gdcm::UIDGenerator uid1;
  gdcm::UIDGenerator uid2;
  const unsigned int n = 100;
  std::set<std::string> uids;
  for(unsigned int i = 0; i < n; ++i)
    {
    const char *unique1 = uid1.Generate();
    const char *unique2 = uid2.Generate();
    if( !unique1 || !unique2 ) return 1;
    std::cout << unique1 << std::endl;
    std::cout << unique2 << std::endl;
    if ( uids.count(unique1) == 1 )
      {
      std::cerr << "Already found: " << unique1 << std::endl;
      return 1;
      }
    uids.insert( unique1 );
    if ( uids.count(unique2) == 1 )
      {
      std::cerr << "Already found: " << unique2 << std::endl;
      return 1;
      }
    uids.insert( unique2 );
    if( strcmp(unique1 , unique2 ) == 0 )
      {
      // That would be very bad !
      return 1;
      }
    }
  int ret = 0;
  ret += TestUIDGeneratorValid();

  return ret;
}
