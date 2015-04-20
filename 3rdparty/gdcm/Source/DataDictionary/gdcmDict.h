/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDICT_H
#define GDCMDICT_H

#include "gdcmTypes.h"
#include "gdcmTag.h"
#include "gdcmPrivateTag.h"
#include "gdcmDictEntry.h"
#include "gdcmSystem.h"

#include <iostream>
#include <iomanip>
#include <map>

/*
 * FIXME / TODO
 * I need to seriously rewrite this mess. a class template should work for both a public
 * and a private dict
 */

namespace gdcm
{
// Data Element Tag
/**
 * \brief Class to represent a map of DictEntry
 * \note bla
 * TODO FIXME: For Element == 0x0 need to return
 * Name = Group Length
 * ValueRepresentation = UL
 * ValueMultiplicity = 1
 */
class GDCM_EXPORT Dict
{
public:
  typedef std::map<Tag, DictEntry> MapDictEntry;
  typedef MapDictEntry::iterator Iterator;
  typedef MapDictEntry::const_iterator ConstIterator;
  //static DictEntry GroupLengthDictEntry; // = DictEntry("Group Length",VR::UL,VM::VM1);

  Dict():DictInternal() {
    assert( DictInternal.empty() );
  }

  friend std::ostream& operator<<(std::ostream& _os, const Dict &_val);

  ConstIterator Begin() const { return DictInternal.begin(); }
  ConstIterator End() const { return DictInternal.end(); }

  bool IsEmpty() const { return DictInternal.empty(); }
  void AddDictEntry(const Tag &tag, const DictEntry &de)
    {
#ifndef NDEBUG
    MapDictEntry::size_type s = DictInternal.size();
#endif
    DictInternal.insert(
      MapDictEntry::value_type(tag, de));
    assert( s < DictInternal.size() );
    }

  const DictEntry &GetDictEntry(const Tag &tag) const
    {
    MapDictEntry::const_iterator it =
      DictInternal.find(tag);
    if (it == DictInternal.end())
      {
#ifdef UNKNOWNPUBLICTAG
      // test.acr
      if( tag != Tag(0x28,0x15)
        && tag != Tag(0x28,0x16)
        && tag != Tag(0x28,0x199)
        // gdcmData/TheralysGDCM1.dcm
        && tag != Tag(0x20,0x1)
        // gdcmData/0019004_Baseline_IMG1.dcm
        && tag != Tag(0x8348,0x339)
        && tag != Tag(0xb5e8,0x338)
        // gdcmData/dicomdir_Acusson_WithPrivate_WithSR
        && tag != Tag(0x40,0xa125)
      )
        {
        assert( 0 && "Impossible" );
        }
#endif
      it = DictInternal.find( Tag(0xffff,0xffff) );
      return it->second;
      }
    assert( DictInternal.count(tag) == 1 );
    return it->second;
    }

  /// Function to return the Keyword from a Tag
  const char *GetKeywordFromTag(Tag const & tag) const
    {
    MapDictEntry::const_iterator it =
      DictInternal.find(tag);
    if (it == DictInternal.end())
      {
      return NULL;
      }
    assert( DictInternal.count(tag) == 1 );
    return it->second.GetKeyword();
    }

  /// Lookup DictEntry by keyword. Even if DICOM standard defines keyword
  /// as being unique. The lookup table is built on Tag. Therefore
  /// looking up a DictEntry by Keyword is more inefficient than looking up
  /// by Tag.
  const DictEntry &GetDictEntryByKeyword(const char *keyword, Tag & tag) const
    {
    MapDictEntry::const_iterator it =
      DictInternal.begin();
    if( keyword )
      {
      for(; it != DictInternal.end(); ++it)
        {
        if( strcmp( keyword, it->second.GetKeyword() ) == 0 )
          {
          // Found a match !
          tag = it->first;
          break;
          }
        }
      }
    else
      {
      it = DictInternal.end();
      }
    if (it == DictInternal.end())
      {
      tag = Tag(0xffff,0xffff);
      it = DictInternal.find( tag );
      return it->second;
      }
    assert( DictInternal.count(tag) == 1 );
    return it->second;
    }

  /// Inefficient way of looking up tag by name. Technically DICOM
  /// does not garantee uniqueness (and Curve / Overlay are there to prove it). But
  /// most of the time name is in fact uniq and can be uniquely link to a tag
  const DictEntry &GetDictEntryByName(const char *name, Tag & tag) const
    {
    MapDictEntry::const_iterator it =
      DictInternal.begin();
    if( name )
      {
      for(; it != DictInternal.end(); ++it)
        {
        if( strcmp( name, it->second.GetName() ) == 0 )
          {
          // Found a match !
          tag = it->first;
          break;
          }
        }
      }
    else
      {
      it = DictInternal.end();
      }
    if (it == DictInternal.end())
      {
      tag = Tag(0xffff,0xffff);
      it = DictInternal.find( tag );
      return it->second;
      }
    assert( DictInternal.count(tag) == 1 );
    return it->second;
    }

protected:
  friend class Dicts;
  void LoadDefault();

private:
  Dict &operator=(const Dict &_val); // purposely not implemented
  Dict(const Dict &_val); // purposely not implemented

  MapDictEntry DictInternal;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& os, const Dict &val)
{
  Dict::MapDictEntry::const_iterator it = val.DictInternal.begin();
  for(;it != val.DictInternal.end(); ++it)
    {
    const Tag &t = it->first;
    const DictEntry &de = it->second;
    os << t << " " << de << '\n';
    }

  return os;
}

// TODO
// For private dict, element < 0x10 should automatically defined:
// Name = "Private Creator"
// ValueRepresentation = LO
// ValueMultiplicity = 1
// Owner = ""

/**
 * \brief Private Dict
 */
class GDCM_EXPORT PrivateDict
{
  typedef std::map<PrivateTag, DictEntry> MapDictEntry;
  friend std::ostream& operator<<(std::ostream& os, const PrivateDict &val);
public:
  PrivateDict() {}
  ~PrivateDict() {}
  void AddDictEntry(const PrivateTag &tag, const DictEntry &de)
    {
#ifndef NDEBUG
    MapDictEntry::size_type s = DictInternal.size();
#endif
    DictInternal.insert(
      MapDictEntry::value_type(tag, de));
// The following code should only be used when manually constructing a Private.xml file by hand
// it will get rid of VR::UN duplicate (ie. if a VR != VR::Un can be found)
#if defined(NDEBUG) && 0
    if( s == DictInternal.size() )
      {
      MapDictEntry::iterator it =
        DictInternal.find(tag);
      assert( it != DictInternal.end() );
      DictEntry &duplicate = it->second;
      assert( de.GetVR() == VR::UN || duplicate.GetVR() == VR::UN );
      assert( de.GetVR() != duplicate.GetVR() );
      if( duplicate.GetVR() == VR::UN )
        {
        assert( de.GetVR() != VR::UN );
        duplicate.SetVR( de.GetVR() );
        duplicate.SetVM( de.GetVM() );
        assert( GetDictEntry(tag).GetVR() != VR::UN );
        assert( GetDictEntry(tag).GetVR() == de.GetVR() );
        assert( GetDictEntry(tag).GetVM() == de.GetVM() );
        }
      return;
      }
#endif
    assert( s < DictInternal.size() /*&& std::cout << tag << "," << de << std::endl*/ );
    }
  /// Remove entry 'tag'. Return true on success (element was found
  /// and remove). return false if element was not found.
  bool RemoveDictEntry(const PrivateTag &tag)
    {
    MapDictEntry::size_type s =
      DictInternal.erase(tag);
    assert( s == 1 || s == 0 );
    return s == 1;
    }
  bool FindDictEntry(const PrivateTag &tag) const
    {
    MapDictEntry::const_iterator it =
      DictInternal.find(tag);
    if (it == DictInternal.end())
      {
      return false;
      }
    return true;
    }
  const DictEntry &GetDictEntry(const PrivateTag &tag) const
    {
    // if 0x10 -> return Private Creator
    MapDictEntry::const_iterator it =
      DictInternal.find(tag);
    if (it == DictInternal.end())
      {
      //assert( 0 && "Impossible" );
      it = DictInternal.find( PrivateTag(0xffff,0xffff,"GDCM Private Sentinel" ) );
      assert (it != DictInternal.end());
      return it->second;
      }
    assert( DictInternal.count(tag) == 1 );
    return it->second;
    }


  void PrintXML() const
    {
    MapDictEntry::const_iterator it = DictInternal.begin();
    std::cout << "<dict edition=\"2008\">\n";
    for(;it != DictInternal.end(); ++it)
      {
      const PrivateTag &t = it->first;
      const DictEntry &de = it->second;
      std::cout << "  <entry group=\"" << std::hex << std::setw(4)
        << std::setfill('0') << t.GetGroup() << "\"" <<
        " element=\"xx" << std::setw(2) << std::setfill('0')<< t.GetElement() << "\"" << " vr=\""
        << de.GetVR() << "\" vm=\"" << de.GetVM() << "\" owner=\""
        << t.GetOwner();
      const char *name = de.GetName();
      if( *name == 0 )
        {
        std::cout << "\"/>\n";
        }
      else
        {
        std::cout << "\" name=\"" << de.GetName() << "\"/>\n";
        }
      }
    std::cout << "</dict>\n";
    }

  bool IsEmpty() const { return DictInternal.empty(); }
protected:
  friend class Dicts;
  void LoadDefault();

private:
  PrivateDict &operator=(const PrivateDict &_val); // purposely not implemented
  PrivateDict(const PrivateDict &_val); // purposely not implemented

  MapDictEntry DictInternal;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& os, const PrivateDict &val)
{
  PrivateDict::MapDictEntry::const_iterator it = val.DictInternal.begin();
  for(;it != val.DictInternal.end(); ++it)
    {
    const PrivateTag &t = it->first;
    const DictEntry &de = it->second;
    os << t << " " << de << '\n';
    }

  return os;
}

} // end namespace gdcm

#endif //GDCMDICT_H
