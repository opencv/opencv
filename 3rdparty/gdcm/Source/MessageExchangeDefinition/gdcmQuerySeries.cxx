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

//note that at the series and image levels, there is no distinction between
//the root query types.

#include "gdcmQuerySeries.h"
#include "gdcmQueryPatient.h"
#include "gdcmQueryStudy.h"
#include "gdcmAttribute.h"

namespace gdcm
{

std::vector<Tag> QuerySeries::GetRequiredTags(const ERootType& ) const
{
  std::vector<Tag> theReturn;//see 3.4 C.6.1.1.4
  theReturn.push_back(Tag(0x0008, 0x0060));
  theReturn.push_back(Tag(0x0020, 0x0011));
  return theReturn;
}

std::vector<Tag> QuerySeries::GetUniqueTags(const ERootType& ) const
{
  std::vector<Tag> theReturn;//see 3.4 C.6.1.1.4
  theReturn.push_back(Tag(0x0020, 0x000E));

  return theReturn;
}

std::vector<Tag> QuerySeries::GetOptionalTags(const ERootType& ) const
{
  std::vector<Tag> theReturn;//see 3.4 C.6.1.1.4
  theReturn.push_back(Tag(0x0020, 0x1209));
  return theReturn;
}

std::vector<Tag> QuerySeries::GetHierachicalSearchTags(const ERootType& inRootType) const
{
  std::vector<Tag> tags;
  if( inRootType == ePatientRootType )
    {
    QueryPatient qp;
    tags = qp.GetUniqueTags(inRootType);
    }
  // add study level
  QueryStudy qs;
  std::vector<Tag> qstags = qs.GetUniqueTags(inRootType);
  tags.insert(tags.end(), qstags.begin(), qstags.end());
  // add series level
  std::vector<Tag> utags = GetUniqueTags(inRootType);
  tags.insert(tags.end(), utags.begin(), utags.end());
  return tags;
}

DataElement QuerySeries::GetQueryLevel() const
{
  const Attribute<0x0008, 0x0052> level = { "SERIES" };
  return level.GetAsDataElement();
}

static const char QuerySeriesString[] = "Series";

const char * QuerySeries::GetName() const
{
  return QuerySeriesString;
}

}
