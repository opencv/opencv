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
this file defines the messages for the cfind action
5 oct 2010 mmr
*/

#include "gdcmCFindMessages.h"
#include "gdcmUIDs.h"
#include "gdcmAttribute.h"
#include "gdcmImplicitDataElement.h"
#include "gdcmPresentationContextRQ.h"
#include "gdcmCommandDataSet.h"
#include "gdcmULConnection.h"

namespace gdcm{
namespace network{

std::vector<PresentationDataValue> CFindRQ::ConstructPDV(
 const ULConnection &inConnection, const BaseRootQuery* inRootQuery)
{
  std::vector<PresentationDataValue> thePDVs;
  PresentationDataValue thePDV;
#if 0
  int contextID = ePatientRootQueryRetrieveInformationModelFIND;
  const char *uid = UIDs::GetUIDString(
    UIDs::PatientRootQueryRetrieveInformationModelFIND );
  std::string suid = uid;
  if (dynamic_cast<const StudyRootQuery*>(inRootQuery)!=NULL)
    {
    contextID = eStudyRootQueryRetrieveInformationModelFIND;
    const char *uid2 = UIDs::GetUIDString(
      UIDs::StudyRootQueryRetrieveInformationModelFIND );
    suid = uid2;
    }
  thePDV.SetPresentationContextID(contextID);//could it be 5, if the server does study?
#else
  PresentationContextRQ pc( inRootQuery->GetAbstractSyntaxUID() );
  uint8_t presidx = inConnection.GetPresentationContextIDFromPresentationContext(pc);
  if( !presidx )
    {
    // try harder:
    PresentationContextRQ pc2( inRootQuery->GetAbstractSyntaxUID(), UIDs::ExplicitVRLittleEndian);
    presidx = inConnection.GetPresentationContextIDFromPresentationContext(pc2);
    if( !presidx )
      {
      gdcmErrorMacro( "Could not find Pres Cont ID" );
      return thePDVs;
      }
    }
  thePDV.SetPresentationContextID( presidx );
#endif

  thePDV.SetCommand(true);
  thePDV.SetLastFragment(true);
  //ignore incoming data set, make your own

  CommandDataSet ds;
  ds.Insert( pc.GetAbstractSyntax().GetAsDataElement() );
  {
  Attribute<0x0,0x100> at = { 32 };
  ds.Insert( at.GetAsDataElement() );
  }
  {
  Attribute<0x0,0x110> at = { 1 };
  ds.Insert( at.GetAsDataElement() );
  }
  {
  Attribute<0x0,0x700> at = { 2 };
  ds.Insert( at.GetAsDataElement() );
  }
  {
  Attribute<0x0,0x800> at = { 1 };
  ds.Insert( at.GetAsDataElement() );
  }
  {
  Attribute<0x0,0x0> at = { 0 };
  unsigned int glen = ds.GetLength<ImplicitDataElement>();
  assert( (glen % 2) == 0 );
  at.SetValue( glen );
  ds.Insert( at.GetAsDataElement() );
  }

  thePDV.SetDataSet(ds);
  thePDVs.push_back(thePDV);
  thePDV.SetDataSet(inRootQuery->GetQueryDataSet());
  thePDV.SetMessageHeader( 2 );
  thePDVs.push_back(thePDV);
  return thePDVs;
}

std::vector<PresentationDataValue>  CFindRSP::ConstructPDVByDataSet(const DataSet* inDataSet){
  std::vector<PresentationDataValue> thePDV;
  (void)inDataSet;
  assert( 0 && "TODO" );
  return thePDV;
}
std::vector<PresentationDataValue>  CFindCancelRQ::ConstructPDVByDataSet(const DataSet* inDataSet){
  std::vector<PresentationDataValue> thePDV;
  (void)inDataSet;
  assert( 0 && "TODO" );
  return thePDV;
}


}//namespace network
}//namespace gdcm
