/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMMOVEPATIENTROOTQUERY_H
#define GDCMMOVEPATIENTROOTQUERY_H

#include "gdcmFindPatientRootQuery.h"

namespace gdcm
{
/**
 * \brief MovePatientRootQuery
 * contains: the class which will produce a dataset for c-move with patient root
 */
class GDCM_EXPORT MovePatientRootQuery : public BaseRootQuery
{
  friend class QueryFactory;
public:
  MovePatientRootQuery();

  void InitializeDataSet(const EQueryLevel& inQueryLevel);

  std::vector<Tag> GetTagListByLevel(const EQueryLevel& inQueryLevel);

  bool ValidateQuery(bool inStrict = true) const;

  UIDs::TSName GetAbstractSyntaxUID() const;
};

} // end namespace gdcm

#endif // GDCMMOVEPATIENTROOTQUERY_H
