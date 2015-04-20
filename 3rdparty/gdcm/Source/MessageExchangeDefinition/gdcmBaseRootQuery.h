/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef GDCMBASEROOTQUERY_H
#define GDCMBASEROOTQUERY_H

#include "gdcmDataSet.h"
#include "gdcmUIDs.h"
#include "gdcmObject.h"
#include "gdcmQueryPatient.h"
#include "gdcmQueryStudy.h"
#include "gdcmQuerySeries.h"
#include "gdcmQueryImage.h"

namespace gdcm
{
  class QueryFactory;
  class DictEntry;

  enum EQueryLevel
    {
    // -1 is reserved do not use
    ePatient = 0,
    eStudy   = 1,
    eSeries  = 2,
    eImage   = 3
    };
  enum EQueryType
    {
    eFind = 0,
    eMove
    };

/**
 * \brief BaseRootQuery
 * contains: a baseclass which will produce a dataset for c-find and c-move
 * with patient/study root
 *
 * This class contains the functionality used in patient c-find and c-move
 * queries.  PatientRootQuery and StudyRootQuery derive from this class.
 *
 * Namely:
 * 1) list all tags associated with a particular query type
 * 2) produce a query dataset via tag association
 *
 * Eventually, it can be used to validate a particular dataset type.
 *
 * The dataset held by this object (or, really, one of its derivates) should be
 * passed to a c-find or c-move query.
 */
class GDCM_EXPORT BaseRootQuery : public Object
{
  //these four classes contain the required, unique, and optional tags from the standard.
  //used both to list the tags as well as to validate a dataset, if ever we were to do so.
protected:
  QueryPatient mPatient;
  QueryStudy mStudy;
  QuerySeries mSeries;
  QueryImage mImage;

  DataSet mDataSet;
  friend class QueryFactory;
  BaseRootQuery();

  ERootType mRootType; //set in construction, and it's something else in the study root type
  std::string mHelpDescription; //used when generating the help output

  void SetSearchParameter(const Tag& inTag, const DictEntry& inDictEntry, const std::string& inValue);
public:
  virtual ~BaseRootQuery();

  void SetSearchParameter(const Tag& inTag, const std::string& inValue);
  void SetSearchParameter(const std::string& inKeyword, const std::string& inValue);

  const std::ostream &WriteHelpFile(std::ostream &os);

  //this function allows writing of the query to disk for storing for future use
  //virtual in case it needs to be overiden
  //returns false if the operation failed
  bool WriteQuery(const std::string& inFileName);

  /// Set/Get the internal representation of the query as a DataSet
  DataSet const & GetQueryDataSet() const;
  DataSet & GetQueryDataSet();
  void AddQueryDataSet(const DataSet & ds);

  ///this function will return all tags at a given query level, so that
  ///they maybe selected for searching.  The boolean forFind is true
  ///if the query is a find query, or false for a move query.
  virtual std::vector<Tag> GetTagListByLevel(const EQueryLevel& inQueryLevel) = 0;

  /// this function sets tag 8,52 to the appropriate value based on query level
  /// also fills in the right unique tags, as per the standard's requirements
  /// should allow for connection with dcmtk
  virtual void InitializeDataSet(const EQueryLevel& inQueryLevel) = 0;

  ///have to be able to ensure that
  ///0x8,0x52 is set (which will be true if InitializeDataSet is called...)
  ///that the level is appropriate (ie, not setting PATIENT for a study query
  ///that the tags in the query match the right level (either required, unique, optional)
  ///by default, this function checks to see if the query is for finding, which is more
  ///permissive than for moving.  For moving, only the unique tags are allowed.
  ///10 Jan 2011: adding in the 'strict' mode.
  ///according to the standard (at least, how I've read it), only tags for a particular
  ///level should be allowed in a particular query (ie, just series level tags in a series
  ///level query).  However, it seems that dcm4chee doesn't share that interpretation.
  ///So, if 'inStrict' is false, then tags from the current level and all higher levels
  ///are now considered valid.  So, if you're doing a non-strict series-level query,
  ///tags from the patient and study level can be passed along as well.
  virtual bool ValidateQuery(bool inStrict = true) const = 0;

  virtual UIDs::TSName GetAbstractSyntaxUID() const = 0;

  void Print(std::ostream &os) const;

  static const char *GetQueryLevelString( EQueryLevel ql );
  static int GetQueryLevelFromString( const char * str );

  static QueryBase * Construct(ERootType inRootType, EQueryLevel qlevel);
  EQueryLevel GetQueryLevelFromQueryRoot( ERootType roottype );
};

} // end namespace gdcm

#endif //GDCMBASEROOTQUERY_H
