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
/*
contains: a baseclass which will produce a dataset for c-find and c-move with
patient root

This class contains the functionality used in patient c-find and c-move
queries.  StudyRootQuery derives from this class.

Namely:
1) list all tags associated with a particular query type
2) produce a query dataset via tag association

Eventually, it can be used to validate a particular dataset type.

The dataset held by this object (or, really, one of its derivates) should be
passed to a c-find or c-move query.

*/

#include "gdcmBaseRootQuery.h"
#include "gdcmDataElement.h"
#include "gdcmTag.h"
#include "gdcmDict.h"
#include "gdcmDictEntry.h"
#include "gdcmDicts.h"
#include "gdcmGlobal.h"
#include "gdcmAttribute.h"
#include "gdcmWriter.h"
#include "gdcmPrinter.h"
#include <limits>

namespace gdcm
{

BaseRootQuery::BaseRootQuery()
{
  //nothing to do, really
}
BaseRootQuery::~BaseRootQuery()
{
  //nothing to do, really
}

void BaseRootQuery::SetSearchParameter(const Tag& inTag, const DictEntry& inDictEntry, const std::string& inValue)
{
  //borrowed this code from anonymization; not sure if it's correct, though.
  DataElement de;
  de.SetTag( inTag );
  const VR &vr = inDictEntry.GetVR();
  if( vr.IsDual() )
    {
    if( vr == VR::US_SS )
      {
      de.SetVR( VR::US );
      }
    else if( vr == VR::US_SS_OW )
      {
      de.SetVR( VR::OW );
      }
    else if( vr == VR::OB_OW )
      {
      de.SetVR( VR::OB );
      }
    }
  else
    {
    de.SetVR( vr );
    }

  std::string thePaddedValue = inValue;
  if (thePaddedValue.length() % 2 ){
    thePaddedValue.push_back(' ');
  }

  assert(thePaddedValue.length() < std::numeric_limits<uint32_t>::max());
  de.SetByteValue(thePaddedValue.c_str(), (uint32_t)thePaddedValue.length());

  //Replace any existing values
  mDataSet.Replace(de);
}

void BaseRootQuery::SetSearchParameter(const Tag& inTag, const std::string& inValue){
  //IF WE WANTED, we could validate the incoming tag as belonging to our set of tags.
  //but we will not.

  static const Global &g = Global::GetInstance();
  static const Dicts &dicts = g.GetDicts();
  static const Dict &pubdict = dicts.GetPublicDict();

  const DictEntry &dictentry = pubdict.GetDictEntry(inTag);

  SetSearchParameter(inTag, dictentry, inValue);

}
void BaseRootQuery::SetSearchParameter(const std::string& inKeyword, const std::string& inValue){

  static const Global &g = Global::GetInstance();
  static const Dicts &dicts = g.GetDicts();
  static const Dict &pubdict = dicts.GetPublicDict();

  Tag theTag;
  const DictEntry &dictentry = pubdict.GetDictEntryByName(inKeyword.c_str(), theTag);
  SetSearchParameter(theTag, dictentry, inValue);
}

const std::ostream &BaseRootQuery::WriteHelpFile(std::ostream &os)
{
  //mash all the query types into a vector for ease-of-use
  std::vector<QueryBase*> theQueries;
  std::vector<QueryBase*>::const_iterator qtor;
  theQueries.push_back(&mPatient);
  theQueries.push_back(&mStudy);
  theQueries.push_back(&mSeries);
  theQueries.push_back(&mImage);


  std::vector<Tag> theTags;
  std::vector<Tag>::iterator ttor;


  static const Global &g = Global::GetInstance();
  static const Dicts &dicts = g.GetDicts();
  static const Dict &pubdict = dicts.GetPublicDict();

  os << "The following tags must be supported by a C-FIND/C-MOVE " << mHelpDescription << ": " << std::endl;
  for (qtor = theQueries.begin(); qtor < theQueries.end(); qtor++){
    os << "Level: " << (*qtor)->GetName() << std::endl;
    theTags = (*qtor)->GetRequiredTags(mRootType);
    for (ttor = theTags.begin(); ttor < theTags.end(); ttor++){
      const DictEntry &dictentry = pubdict.GetDictEntry(*ttor);
      os << "Keyword: " << dictentry.GetKeyword() << " Tag: " << *ttor << std::endl;
    }
    os << std::endl;
  }


  os << std::endl;
  os << "The following tags are unique at each level of a " << mHelpDescription << ": " << std::endl;
  for (qtor = theQueries.begin(); qtor < theQueries.end(); qtor++){
    os << "Level: " << (*qtor)->GetName() << std::endl;
    theTags = (*qtor)->GetUniqueTags(mRootType);
    for (ttor = theTags.begin(); ttor < theTags.end(); ttor++){
      const DictEntry &dictentry = pubdict.GetDictEntry(*ttor);
      os << "Keyword: " << dictentry.GetKeyword() << " Tag: " << *ttor << std::endl;
    }
    os << std::endl;
  }


  os << std::endl;
  os << "The following tags are optional at each level of a " << mHelpDescription << ": "  << std::endl;
  for (qtor = theQueries.begin(); qtor < theQueries.end(); qtor++){
    os << "Level: " << (*qtor)->GetName() << std::endl;
    theTags = (*qtor)->GetOptionalTags(mRootType);
    for (ttor = theTags.begin(); ttor < theTags.end(); ttor++){
      const DictEntry &dictentry = pubdict.GetDictEntry(*ttor);
      os << "Keyword: " << dictentry.GetKeyword() << " Tag: " << *ttor << std::endl;
    }
    os << std::endl;
  }

  os << std::endl;

  return os;
}


bool BaseRootQuery::WriteQuery(const std::string& inFileName)
{
  Writer writer;
  writer.SetCheckFileMetaInformation( false );
  writer.GetFile().GetHeader().SetDataSetTransferSyntax(
    TransferSyntax::ImplicitVRLittleEndian );
  writer.GetFile().SetDataSet( GetQueryDataSet() );
  writer.SetFileName( inFileName.c_str() );
  if( !writer.Write() )
    {
    gdcmWarningMacro( "Could not write: " << inFileName );
    return false;
    }
  return true;
}

DataSet const & BaseRootQuery::GetQueryDataSet() const
{
  return mDataSet;
}

DataSet & BaseRootQuery::GetQueryDataSet()
{
  return mDataSet;
}

void BaseRootQuery::AddQueryDataSet(const DataSet & ds)
{
  // Cannot use std::set::insert in case a unique key was added (eg. 10,20 with an empty
  // value). The user defined value for 10,20 in ds would not be taken into account
  DataSet::ConstIterator it = ds.Begin();
  for( ; it != ds.End(); ++it )
    {
    mDataSet.Replace( *it );
    }
}

void BaseRootQuery::Print(std::ostream &os) const
{
  UIDs::TSName asuid = GetAbstractSyntaxUID();
  const char *asname = UIDs::GetUIDName( asuid );
  os << "===================== OUTGOING DIMSE MESSAGE ====================" << std::endl;
  os << "Affected SOP Class UID        :" << asname << std::endl;
  os << "======================= END DIMSE MESSAGE =======================" << std::endl;

  os << "Find SCU Request Identifiers:" << std::endl;
  os << "# Dicom-Data-Set" << std::endl;
  os << "# Used TransferSyntax: Unknown Transfer Syntax" << std::endl;
  Printer p;
  p.PrintDataSet( mDataSet, os );
}

static const char *QueryLevelStrings[] = {
  "PATIENT ",
  "STUDY ",
  "SERIES",
  "IMAGE ",
};

const char *BaseRootQuery::GetQueryLevelString( EQueryLevel ql )
{
  return QueryLevelStrings[ ql ];
}

int BaseRootQuery::GetQueryLevelFromString( const char * str )
{
  if( str )
    {
    const std::string s = str;
    static const int n =
      sizeof( QueryLevelStrings ) / sizeof( *QueryLevelStrings );
    for( int i = 0; i < n; ++i )
      {
      if( s == QueryLevelStrings[ i ] )
        {
        return i;
        }
      }
    }
  return -1; // FIXME never use it as valid enum
}

QueryBase * BaseRootQuery::Construct( ERootType inRootType, EQueryLevel qlevel )
{
  QueryBase* qb = NULL;
  // Check no new extension:
  assert( inRootType == ePatientRootType || inRootType == eStudyRootType );

  if( qlevel == ePatient )
    {
    if( inRootType == ePatientRootType )
      {
      qb = new QueryPatient;
      }
    }
  else if( qlevel == eStudy )
    {
    qb = new QueryStudy;
    }
  else if( qlevel == eSeries )
    {
    qb = new QuerySeries;
    }
  else if( qlevel == eImage )
    {
    qb = new QueryImage;
    }
  return qb;
}

EQueryLevel BaseRootQuery::GetQueryLevelFromQueryRoot( ERootType roottype )
{
  EQueryLevel level = ePatient;
  switch( roottype )
    {
  case ePatientRootType:
    level = ePatient;
    break;
  case eStudyRootType:
    level = eStudy;
    break;
    }
  assert( level == eStudy || level == ePatient );
  return level;
}

} // end namespace gdcm
