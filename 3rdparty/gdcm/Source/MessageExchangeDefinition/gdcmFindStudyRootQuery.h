/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFINDSTUDYROOTQUERY_H
#define GDCMFINDSTUDYROOTQUERY_H

#include "gdcmBaseRootQuery.h"

namespace gdcm
{
/**
 * \brief FindStudyRootQuery
 * contains: the class which will produce a dataset for C-FIND with study root
 */
class GDCM_EXPORT FindStudyRootQuery : public BaseRootQuery
{
  friend class QueryFactory;
public:
  FindStudyRootQuery();

  void InitializeDataSet(const EQueryLevel& inQueryLevel);

  std::vector<Tag> GetTagListByLevel(const EQueryLevel& inQueryLevel);

  /// have to be able to ensure that (0008,0052) is set
  /// that the level is appropriate (ie, not setting PATIENT for a study query
  /// that the tags in the query match the right level (either required, unique, optional)
  bool ValidateQuery(bool inStrict = true) const;

  UIDs::TSName GetAbstractSyntaxUID() const;
};

} // end namespace gdcm

#endif // GDCMFINDSTUDYROOTQUERY_H
