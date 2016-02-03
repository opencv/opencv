/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSORTER_H
#define GDCMSORTER_H

#include "gdcmDirectory.h"
#include "gdcmTag.h"

#include <vector>
#include <string>
#include <map>

namespace gdcm
{
class DataSet;

/**
 * \brief Sorter
 * General class to do sorting using a custom function
 * You simply need to provide a function of type: Sorter::SortFunction
 *
 * \warning implementation details. For now there is no cache mechanism. Which means
 * that everytime you call Sort, all files specified as input paramater are *read*
 *
 * \see Scanner
 */
class GDCM_EXPORT Sorter
{
  friend std::ostream& operator<<(std::ostream &_os, const Sorter &s);
public:
  Sorter();
  virtual ~Sorter();

  /// Typically the output of Directory::GetFilenames()
  virtual bool Sort(std::vector<std::string> const & filenames);

  /// Return the list of filenames as sorted by the specific algorithm used.
  /// Empty by default (before Sort() is called)
  const std::vector<std::string> &GetFilenames() const { return Filenames; }

  /// Print
  void Print(std::ostream &os) const;

  /// UNSUPPORTED FOR NOW
  bool AddSelect( Tag const &tag, const char *value );

  /// Set the sort function which compares one dataset to the other
  typedef bool (*SortFunction)(DataSet const &, DataSet const &);
  void SetSortFunction( SortFunction f );

  virtual bool StableSort(std::vector<std::string> const & filenames);

protected:
  std::vector<std::string> Filenames;
  typedef std::map<Tag,std::string> SelectionMap;
  std::map<Tag,std::string> Selection;
  SortFunction SortFunc;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const Sorter &s)
{
  s.Print( os );
  return os;
}


} // end namespace gdcm

#endif //GDCMSORTER_H
