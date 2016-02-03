/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMMOVESTUDYROOTQUERY_H
#define GDCMMOVESTUDYROOTQUERY_H

#include "gdcmBaseRootQuery.h"

namespace gdcm
{
/**
 * \brief MoveStudyRootQuery
 * contains: the class which will produce a dataset for C-MOVE with study root
 */
class GDCM_EXPORT MoveStudyRootQuery : public BaseRootQuery
{
  friend class QueryFactory;
public:
  MoveStudyRootQuery();

  void InitializeDataSet(const EQueryLevel& inQueryLevel);

  std::vector<Tag> GetTagListByLevel(const EQueryLevel& inQueryLevel);

  bool ValidateQuery(bool inStrict = true) const;

  UIDs::TSName GetAbstractSyntaxUID() const;
};

} // end namespace gdcm

#endif // GDCMMOVESTUDYROOTQUERY_H
