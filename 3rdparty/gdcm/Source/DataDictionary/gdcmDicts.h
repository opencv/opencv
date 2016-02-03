/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDICTS_H
#define GDCMDICTS_H

#include "gdcmDict.h"
#include "gdcmCSAHeaderDict.h"

#include <string>

namespace gdcm
{
/**
 * \brief Class to manipulate the sum of knowledge (all the dict user load)
 * \note bla
 */
class GDCM_EXPORT Dicts
{
  friend std::ostream& operator<<(std::ostream &_os, const Dicts &d);
public:
  Dicts();
  ~Dicts();

  /// works for both public and private dicts:
  /// owner is null for public dict
  /// \warning owner need to be set to appropriate owner for call to work. see
  // DataSet::GetPrivateCreator
  /// NOT THREAD SAFE
  const DictEntry &GetDictEntry(const Tag& tag, const char *owner = NULL) const;

  const DictEntry &GetDictEntry(const PrivateTag& tag) const;

  //enum PublicTypes {
  //  DICOMV3_DICT,
  //  ACRNEMA_DICT,
  //  NIH_DICT
  //};
  const Dict &GetPublicDict() const;

  const PrivateDict &GetPrivateDict() const;
  PrivateDict &GetPrivateDict();

  const CSAHeaderDict &GetCSAHeaderDict() const;

  bool IsEmpty() const { return GetPublicDict().IsEmpty(); }

protected:
  typedef enum {
    PHILIPS,
    GEMS,
    SIEMENS
  //  ...
  } ConstructorType;
  static const char *GetConstructorString(ConstructorType type);

  friend class Global;
  void LoadDefaults();

private:
  // Public dict:
  Dict PublicDict;

  // Private Dicts:
  PrivateDict ShadowDict;

  CSAHeaderDict CSADict;
  Dicts &operator=(const Dicts &_val); // purposely not implemented
  Dicts(const Dicts &_val); // purposely not implemented
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const Dicts &d)
{
  (void)d;
  return os;
}


} // end namespace gdcm

#endif //GDCMDICTS_H
