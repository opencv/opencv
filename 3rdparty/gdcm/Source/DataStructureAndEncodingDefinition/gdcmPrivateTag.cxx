/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPrivateTag.h"
#include "gdcmTrace.h"
#include "gdcmSystem.h" // FIXME

#include <stdio.h> // sscanf
#include <limits> // numeric_limits

namespace gdcm
{
  bool PrivateTag::ReadFromCommaSeparatedString(const char *str)
    {
    if( !str ) return false;
    unsigned int group = 0, element = 0;
    std::string owner;
    owner.resize( strlen(str) ); // str != NULL
    if( sscanf(str, "%04x,%04x,%[^\"]", &group , &element, &owner[0] ) != 3
      || group > std::numeric_limits<uint16_t>::max()
      || element > std::numeric_limits<uint16_t>::max()
      /*|| strlen(owner.c_str()) == 0*/ ) // can't use owner.empty()
      {
      gdcmDebugMacro( "Problem reading Private Tag: " << str );
      return false;
      }
    SetGroup( (uint16_t)group );
    // This is not considered an error to specify element as 1010 for example.
    // just keep the lower bits of element:
    SetElement( (uint8_t)element );
    SetOwner( owner.c_str() );
    if( !*GetOwner() )
      {
      gdcmDebugMacro( ": " << str );
      return false;
      }
    return true;
    }

  bool PrivateTag::operator<(const PrivateTag &_val) const
    {
    const Tag & t1 = *this;
    const Tag & t2 = _val;
    if( t1 == t2 )
      {
      const char *s1 = Owner.c_str();
      const char *s2 = _val.GetOwner();
      assert( s1 );
      assert( s2 );
      if( *s1 )
        assert( s1[strlen(s1)-1] != ' ' );
      if( *s2 )
        assert( s2[strlen(s2)-1] != ' ' );
      bool res = strcmp(s1, s2) < 0;
#ifdef DEBUG_DUPLICATE
      if( *s1 && *s2 && gdcm::System::StrCaseCmp(s1,s2) == 0 && strcmp(s1,s2) != 0 )
        {
        // FIXME:
        // Typically this should only happen with the "Philips MR Imaging DD 001" vs "PHILIPS MR IMAGING DD 001"
        // or "Philips Imaging DD 001" vr "PHILIPS IMAGING DD 001"
        //assert( strcmp(Owner.c_str(), _val.GetOwner()) == 0 );
        //return true;
        const bool res2 = gdcm::System::StrCaseCmp(s1,s2) < 0;
        res = res2;
        assert( 0 );
        }
#endif
      return res;
      }
    else return t1 < t2;
    }

DataElement PrivateTag::GetAsDataElement() const
{
  DataElement de;
  de.SetTag( *this );
  de.SetVR( VR::LO );
  std::string copy = Owner;
  if( copy.size() % 2 ) copy.push_back( ' ' );
  de.SetByteValue( copy.c_str(), (uint32_t)copy.size() );
  return de;
}

} // end namespace gdcm
