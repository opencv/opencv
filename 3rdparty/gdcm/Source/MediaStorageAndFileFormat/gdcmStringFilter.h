/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSTRINGFILTER_H
#define GDCMSTRINGFILTER_H

#include "gdcmDataElement.h"
#include "gdcmDicts.h"
#include "gdcmFile.h"

namespace gdcm
{

/**
 * \brief StringFilter
 * StringFilter is the class that make gdcm2.x looks more like gdcm1 and transform the binary blob
 * contained in a DataElement into a string, typically this is a nice feature to have for wrapped language
 */
class GDCM_EXPORT StringFilter
{
public:
  StringFilter();
  ~StringFilter();

  ///
  void UseDictAlways(bool) {}

  /// Allow user to pass in there own dicts
  void SetDicts(const Dicts &dicts);

  /// Convert to string the ByteValue contained in a DataElement. The
  /// DataElement must be coming from the actual DataSet associated with File
  /// (see SetFile).
  std::string ToString(const DataElement& de) const;

  /// Directly from a Tag:
  std::string ToString(const Tag& t) const;

  /// Convert to string the ByteValue contained in a DataElement
  /// the returned elements are:
  /// pair.first : the name as found in the dictionary of DataElement
  /// pari.second : the value encoded into a string (US,UL...) are properly converted
  std::pair<std::string, std::string> ToStringPair(const DataElement& de) const;
  /// Directly from a Tag:
  std::pair<std::string, std::string> ToStringPair(const Tag& t) const;

  GDCM_LEGACY(std::string FromString(const Tag&t, const char * value, VL const & vl))

  /// Convert to string the char array defined by the pair (value,len)
  std::string FromString(const Tag&t, const char * value, size_t len);

  /// Set/Get File
  void SetFile(const File& f) { F = f; }
  File &GetFile() { return *F; }
  const File &GetFile() const { return *F; }

  /// Execute the XPATH query to find a value (as string)
  /// return false when attribute is not found (or an error in the XPATH query)
  /// You need to make sure that your XPATH query is syntatically correct
  bool ExecuteQuery(std::string const &query, std::string & value) const;

protected:
  std::pair<std::string, std::string> ToStringPair(const Tag& t, DataSet const &ds) const;
  bool ExecuteQuery(std::string const &query, DataSet const &ds, std::string & value) const;

private:
  std::pair<std::string, std::string> ToStringPairInternal(const DataElement& de, DataSet const &ds) const;
  SmartPointer<File> F;
};

} // end namespace gdcm

#endif //GDCMSTRINGFILTER_H
