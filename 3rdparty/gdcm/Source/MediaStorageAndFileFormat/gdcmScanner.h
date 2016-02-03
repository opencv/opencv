/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSCANNER_H
#define GDCMSCANNER_H

#include "gdcmDirectory.h"
#include "gdcmSubject.h"
#include "gdcmTag.h"
#include "gdcmPrivateTag.h"
#include "gdcmSmartPointer.h"

#include <map>
#include <set>
#include <string>

#include <string.h> // strcmp

namespace gdcm
{
class StringFilter;

/**
 * \brief Scanner
 * This filter is meant for quickly browsing a FileSet (a set of files on
 * disk). Special consideration are taken so as to read the mimimum amount of
 * information in each file in order to retrieve the user specified set of
 * DICOM Attribute.
 *
 * This filter is dealing with both VRASCII and VRBINARY element, thanks to the
 * help of StringFilter
 *
 * \warning IMPORTANT In case of file where tags are not ordered (illegal as
 * per DICOM specification), the output will be missing information
 *
 * \note implementation details. All values are stored in a std::set of
 * std::string. Then the address of the cstring underlying the std::string is
 * used in the std::map.
 *
 * This class implement the Subject/Observer pattern trigger the following events:
 * \li ProgressEvent
 * \li StartEvent
 * \li EndEvent
 */
class GDCM_EXPORT Scanner : public Subject
{
  friend std::ostream& operator<<(std::ostream &_os, const Scanner &s);
public:
  Scanner():Values(),Filenames(),Mappings() {}
  ~Scanner();

  /// struct to map a filename to a value
  /// Implementation note:
  /// all std::map in this class will be using const char * and not std::string
  /// since we are pointing to existing std::string (hold in a std::vector)
  /// this avoid an extra copy of the byte array.
  /// Tag are used as Tag class since sizeof(tag) <= sizeof(pointer)
  typedef std::map<Tag, const char*> TagToValue;
  //typedef std::map<Tag, ConstCharWrapper> TagToValue; //StringMap;
  //typedef TagToStringMap TagToValue;
  typedef TagToValue::value_type TagToValueValueType;

  /// Add a tag that will need to be read. Those are root level skip tags
  void AddTag( Tag const & t );
  void ClearTags();

  // Work in progress do not use:
  void AddPrivateTag( PrivateTag const & t );

  /// Add a tag that will need to be skipped. Those are root level skip tags
  void AddSkipTag( Tag const & t );
  void ClearSkipTags();

  /// Start the scan !
  bool Scan( Directory::FilenamesType const & filenames );

  Directory::FilenamesType const &GetFilenames() const { return Filenames; }

  /// Print result
  void Print( std::ostream & os ) const;

  /// Check if filename is a key in the Mapping table.
  /// returns true only of file can be found, which means
  /// the file was indeed a DICOM file that could be processed
  bool IsKey( const char * filename ) const;

  /// Return the list of filename that are key in the internal map,
  /// which means those filename were properly parsed
  Directory::FilenamesType GetKeys() const;

  // struct to store all the values found:
  typedef std::set< std::string > ValuesType;

  /// Get all the values found (in lexicographic order)
  ValuesType const & GetValues() const { return Values; }

  /// Get all the values found (in lexicographic order) associated with Tag 't'
  ValuesType GetValues(Tag const &t) const;

  /// Get all the values found (in a vector) associated with Tag 't'
  /// This function is identical to GetValues, but is accessible from the wrapped
  /// layer (python, C#, java)
  Directory::FilenamesType GetOrderedValues(Tag const &t) const;

  /* ltstr is CRITICAL, otherwise pointers value are used to do the key comparison */
  struct ltstr
    {
    bool operator()(const char* s1, const char* s2) const
      {
      assert( s1 && s2 );
      return strcmp(s1, s2) < 0;
      }
    };
  typedef std::map<const char *,TagToValue, ltstr> MappingType;
  typedef MappingType::const_iterator ConstIterator;
  ConstIterator Begin() const { return Mappings.begin(); }
  ConstIterator End() const { return Mappings.end(); }

  /// Mappings are the mapping from a particular tag to the map, mapping filename to value:
  MappingType const & GetMappings() const { return Mappings; }

  /// Get the std::map mapping filenames to value for file 'filename'
  TagToValue const & GetMapping(const char *filename) const;

  /// Will loop over all files and return the first file where value match the reference value
  /// 'valueref'
  const char *GetFilenameFromTagToValue(Tag const &t, const char *valueref) const;

  /// Will loop over all files and return a vector of std::strings of filenames
  /// where value match the reference value 'valueref'
  Directory::FilenamesType GetAllFilenamesFromTagToValue(Tag const &t, const char *valueref) const;

  /// See GetFilenameFromTagToValue(). This is simply GetFilenameFromTagToValue followed
  // by a call to GetMapping()
  TagToValue const & GetMappingFromTagToValue(Tag const &t, const char *value) const;

  /// Retrieve the value found for tag: t associated with file: filename
  /// This is meant for a single short call. If multiple calls (multiple tags)
  /// should be done, prefer the GetMapping function, and then reuse the TagToValue
  /// hash table.
  /// \warning Tag 't' should have been added via AddTag() prior to the Scan() call !
  const char* GetValue(const char *filename, Tag const &t) const;

  /// for wrapped language: instanciate a reference counted object
  static SmartPointer<Scanner> New() { return new Scanner; }

protected:
  void ProcessPublicTag(StringFilter &sf, const char *filename);
private:
  // struct to store all uniq tags in ascending order:
  typedef std::set< Tag > TagsType;
  typedef std::set< PrivateTag > PrivateTagsType;
  std::set< Tag > Tags;
  std::set< PrivateTag > PrivateTags;
  std::set< Tag > SkipTags;
  ValuesType Values;
  Directory::FilenamesType Filenames;

  // Main struct that will hold all mapping:
  MappingType Mappings;

  double Progress;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const Scanner &s)
{
  s.Print( os );
  return os;
}

#if defined(SWIGPYTHON) || defined(SWIGCSHARP) || defined(SWIGJAVA)
/*
 * HACK: I need this temp class to be able to manipulate a std::map from python,
 * swig does not support wrapping of simple class like std::map...
 */
class SWIGTagToValue
{
public:
  SWIGTagToValue(Scanner::TagToValue const &t2v):Internal(t2v),it(t2v.begin()) {}
  const Scanner::TagToValueValueType& GetCurrent() const { return *it; }
  const Tag& GetCurrentTag() const { return it->first; }
  const char *GetCurrentValue() const { return it->second; }
  void Start() { it = Internal.begin(); }
  bool IsAtEnd() const { return it == Internal.end(); }
  void Next() { ++it; }
private:
  const Scanner::TagToValue& Internal;
  Scanner::TagToValue::const_iterator it;
};
#endif /* SWIG */

/**
 * \example ScanDirectory.cs
 * This is a C# example on how to use Scanner
 */

} // end namespace gdcm

#endif //GDCMSCANNER_H
