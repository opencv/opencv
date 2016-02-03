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
#include "gdcmTrace.h"
#include "gdcmSystem.h"

#include <bitset>
#include <cstring>

// FIXME...
#if defined(_WIN32) || defined(__CYGWIN__)
#define HAVE_UUIDCREATE
#else
#define HAVE_UUID_GENERATE
#endif

#ifdef HAVE_UUID_GENERATE
#include "gdcm_uuid.h"
#endif

#ifdef GDCM_HAVE_RPC_H
#include <rpc.h>
#endif

namespace gdcm
{

/*
 * This is just plain bad luck. GDCM UID root is 26 byte long
 * And all implementation of the DCE UUID (Theodore Y. Ts'o)
 * are based on a uint128_t (unsigned char [16]). Which
 * means that converted to a base 10 number they require at
 * least 39 bytes to fit in memory, since the highest possible
 * number is 256**16 - 1 = 340282366920938463463374607431768211455
 * Unfortunately root + '.' + suffix should be at most 64 bytes
 *
 * So to get a full UUID implementation as per RFC 4122
 * http://www.ietf.org/rfc/rfc4122.txt we need a shorter
 * root...
 *
 */
const char UIDGenerator::GDCM_UID[] = "1.2.826.0.1.3680043.2.1143";
std::string UIDGenerator::Root = GetGDCMUID();
// The following contains the *encoded* hardware address (not the raw as in ipconfig/ifconfig)
std::string UIDGenerator::EncodedHardwareAddress; // = System::GetHardwareAddress();

const char *UIDGenerator::GetRoot() { return Root.c_str(); }
void UIDGenerator::SetRoot(const char * root) {
  assert( IsValid( root ) );
  Root = root;
}

const char *UIDGenerator::GetGDCMUID()
{
  return GDCM_UID;
}

/*
 * http://www.isthe.com/chongo/tech/comp/fnv/
 */
#define FNV1_64_INIT ((uint64_t)0xcbf29ce484222325ULL)
struct fnv_hash
{
  static uint64_t
  hash(const char* pBuffer, size_t nByteLen)
    {
    uint64_t nHashVal    = FNV1_64_INIT,
             nMagicPrime = 0x00000100000001b3ULL;

    unsigned char* pFirst = ( unsigned char* )( pBuffer ),
                 * pLast  = pFirst + nByteLen;

    while( pFirst < pLast )
      {
      nHashVal ^= *pFirst++,
      nHashVal *= nMagicPrime;
      }

    return nHashVal;
    }
};

/*
Implementation note: You cannot set a root of more than 26 bytes (which should already
enough for most people).
Since implementation is only playing with the first 8bits of the upper
*/
const char* UIDGenerator::Generate()
{
  Unique = GetRoot();
  // We choose here a value of 26 so that we can still have 37 bytes free to
  // set the suffix part which is sufficient to store a 2^(128-8+1)-1 number
  if( Unique.empty() || Unique.size() > 62 ) // 62 is simply the highest possible limit
    {
    // I cannot go any further...
    return NULL;
    }
  unsigned char uuid[16];
  bool r = UIDGenerator::GenerateUUID(uuid);
  // This should only happen in some obscure cases. Since the creation of UUID failed
  // I should try to go any further and make sure the user's computer crash and burn
  // right away
  if( !r ) return 0;
  char randbytesbuf[64];
  size_t len = System::EncodeBytes(randbytesbuf, uuid, sizeof(uuid));
  assert( len < 64 ); // programmer error
  Unique += "."; // This dot is compulsary to separate root from suffix
  if( Unique.size() + len > 64 )
    {
    int idx = 0;
    bool found = false;
    std::bitset<8> x;
    while( !found && idx < 16 ) /* 16 is insane ... oh well */
      {
      // too bad ! randbytesbuf is too long, let's try to truncate the high bits a little
      x = uuid[idx];
      unsigned int i = 0;
      while( ( Unique.size() + len > 64 ) && i < 8 )
        {
        x[7-i] = 0;
        uuid[idx] = (unsigned char)x.to_ulong();
        len = System::EncodeBytes(randbytesbuf, uuid, sizeof(uuid));
        ++i;
        }
      if( ( Unique.size() + len > 64 ) && i == 8 )
        {
        // too bad only reducing the 8 bits from uuid[idx] was not enought,
        // let's set to zero the following bits...
        idx++;
        }
      else
        {
        // cool we found enough to stop
        found = true;
        }
      }
    if( !found )
      {
      // Technically this could only happen when root has a length >= 64 ... is it
      // even remotely possible ?
      gdcmWarningMacro( "Root is too long for current implementation" );
      return NULL;
      }
    }
  // can now safely use randbytesbuf as is, no need to truncate any more:
  Unique += randbytesbuf;

  assert( IsValid( Unique.c_str() ) );

  return Unique.c_str();
}


/* return true on success */
bool UIDGenerator::GenerateUUID(unsigned char *uuid_data)
{
#if defined(HAVE_UUID_GENERATE)
  uuid_t g;
  uuid_generate(g);
  memcpy(uuid_data, g, sizeof(uuid_t));
#elif defined(HAVE_UUID_CREATE)
  uint32_t rv;
  uuid_t g;
  uuid_create(&g, &rv);
  if (rv != uuid_s_ok)
    return false;
  memcpy(uuid_data, &g, sizeof(uuid_t));
#elif defined(HAVE_UUIDCREATE)
  if (FAILED(UuidCreate((UUID *)uuid_data)))
    {
    return false;
    }
#else
#error should not happen
#endif
  return true;
}

bool UIDGenerator::IsValid(const char *uid_)
{
  /*
  9.1 UID ENCODING RULES
  The DICOM UID encoding rules are defined as follows:
  - Each component of a UID is a number and shall consist of one or more digits. The first digit of
  each component shall not be zero unless the component is a single digit.
  Note: Registration authorities may distribute components with non-significant leading zeroes. The leading
  zeroes should be ignored when being encoded (ie. 00029 would be encoded 29).
  - Each component numeric value shall be encoded using the characters 0-9 of the Basic G0 Set
  of the International Reference Version of ISO 646:1990 (the DICOM default character
  repertoire).
  - Components shall be separated by the character "." (2EH).
  - If ending on an odd byte boundary, except when used for network negotiation (See PS 3.8),
  one trailing NULL (00H), as a padding character, shall follow the last component in order to
  align the UID on an even byte boundary.
  - UID's, shall not exceed 64 total characters, including the digits of each component, separators
  between components, and the NULL (00H) padding character if needed.
  */

  /*
   * FIXME: This is not clear in the standard, but I believe a trailing '.' is not allowed since
   * this is considered as a separator for components
   */

  std::string uid = uid_;
  if( uid.size() > 64 || uid.empty() )
    {
    return false;
    }
  if( uid[0] == '.' || uid[uid.size()-1] == '.' ) // important to do that first
    {
    return false;
    }
  std::string::size_type i = 0;
  for(; i < uid.size(); ++i)
    {
    if( uid[i] == '.' ) // if test is true we are garantee that next char is valid (see previous check)
      {
      // check that next character is neither '0' (except single number) not '.'
      if( uid[i+1] == '.' )
        {
        return false;
        }
      else if( uid[i+1] == '0' ) // character is garantee to exist since '.' is not last char
        {
        // Need to check first if we are not at the end of string
        if( i+2 != uid.size() && uid[i+2] != '.' )
          {
          return false;
          }
        }
      }
    else if ( !isdigit( (unsigned char)uid[i] ) )
      {
      return false;
      }
    }
  // no error found !
  return true;
}


} // end namespace gdcm
