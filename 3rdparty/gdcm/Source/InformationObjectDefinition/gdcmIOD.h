/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMIOD_H
#define GDCMIOD_H

#include "gdcmTypes.h"
#include "gdcmTag.h"
#include "gdcmIODEntry.h"

#include <vector>

namespace gdcm
{
class DataSet;
class Defs;

/**
 * \brief Class for representing a IOD
 * \note bla
 *
 * \sa Dict
 */
class GDCM_EXPORT IOD
{
public:
  typedef std::vector<IODEntry> MapIODEntry;
  typedef MapIODEntry::size_type SizeType;

  IOD() {}
  friend std::ostream& operator<<(std::ostream& _os, const IOD &_val);

  void Clear() { IODInternal.clear(); }

  void AddIODEntry(const IODEntry & iode)
    {
    IODInternal.push_back(iode);
    }

  SizeType GetNumberOfIODs() const {
    return IODInternal.size();
  }

  const IODEntry& GetIODEntry(SizeType idx) const
    {
    return IODInternal[idx];
    }

  Type GetTypeFromTag(const Defs &defs, const Tag& tag) const;

private:
  //IOD &operator=(const IOD &_val); // purposely not implemented
  //IOD(const IOD &_val); // purposely not implemented

  MapIODEntry IODInternal;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& _os, const IOD &_val)
{
  IOD::MapIODEntry::const_iterator it = _val.IODInternal.begin();
  for(;it != _val.IODInternal.end(); ++it)
    {
    _os << *it << '\n';
    }

  return _os;
}

} // end namespace gdcm

#endif //GDCMIOD_H
