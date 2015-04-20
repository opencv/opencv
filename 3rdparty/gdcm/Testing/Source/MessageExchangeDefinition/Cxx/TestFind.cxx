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

#include "gdcmCompositeNetworkFunctions.h"
#include "gdcmTag.h"
#include "gdcmQueryFactory.h"
#include "gdcmMovePatientRootQuery.h"


int TestFind(int , char *[])
{
  std::string hostname = "www.dicomserver.co.uk";
  uint16_t port = 11112;
  std::string callaetitle = "GDCM_ROCKS";
  std::string callingaetitle = "ACME1";

  
  gdcm::Tag theTag(0x0010, 0x0010);
  std::string theName = "F*";
  std::pair<gdcm::Tag, std::string> theTagPair =
    std::make_pair(theTag, theName);

  std::vector<std::pair<gdcm::Tag, std::string> > theTags;
  theTags.push_back(theTagPair);

  gdcm::BaseRootQuery* theQuery = gdcm::CompositeNetworkFunctions::ConstructQuery(
    gdcm::ePatientRootType, gdcm::ePatient, theTags);

  if (!theQuery) {
    std::cerr << "Query construction failed!" << std::endl; 
    return 1;
  }    
  
  if (!theQuery->ValidateQuery(false))
    {
    std::cerr << "Find query is not valid.  Please try again." << std::endl;
    delete theQuery;
    return 1;
    }

  std::vector<gdcm::DataSet> theDataSet ;
  bool b =
    gdcm::CompositeNetworkFunctions::CFind(hostname.c_str(), port,
      theQuery, theDataSet , callingaetitle.c_str(), callaetitle.c_str());
  if( !b ) return 1;

  //need to put some kind of validation of theDataSet here

  return (theDataSet.empty() ? 1:0);//shouldn't be a zero-sized dataset
}
