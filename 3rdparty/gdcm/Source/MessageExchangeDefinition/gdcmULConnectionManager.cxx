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
#include "gdcmULConnectionManager.h"

#include "gdcmUserInformation.h"
#include "gdcmULEvent.h"
#include "gdcmPDUFactory.h"
#include "gdcmReader.h"
#include "gdcmAAssociateRQPDU.h"
#include "gdcmAttribute.h"
#include "gdcmBaseRootQuery.h"
#include "gdcmDataSetEvent.h"

#include "gdcmAReleaseRPPDU.h"

#include "gdcmULBasicCallback.h"

#include <vector>
#include <socket++/echo.h>//for setting up the local socket
#include "gdcmTrace.h"
#include "gdcmPrinter.h"

namespace gdcm
{
namespace network
{


ULConnectionManager::ULConnectionManager()
{
  mConnection = NULL;
  mSecondaryConnection = NULL;
}

ULConnectionManager::~ULConnectionManager()
{
  if (mConnection != NULL)
    {
    delete mConnection;
    mConnection = NULL;
    }
  if (mSecondaryConnection != NULL)
    {
    delete mSecondaryConnection;
    mSecondaryConnection = NULL;
    }
}

bool ULConnectionManager::EstablishConnection(const std::string& inAETitle,
  const std::string& inConnectAETitle,
  const std::string& inComputerName, long inIPAddress,
  unsigned short inConnectPort, double inTimeout,
  std::vector<PresentationContext> const & pcVector)
{

  //generate a ULConnectionInfo object
  UserInformation userInfo;
  ULConnectionInfo connectInfo;
  if (inConnectAETitle.size() > 16)
    return false;//too long an AETitle, probably need better failure message
  if (inAETitle.size() > 16) return false; //as above
  if (!connectInfo.Initialize(userInfo, inConnectAETitle.c_str(),
      inAETitle.c_str(), inIPAddress, inConnectPort, inComputerName))
    {
    return false;
    }

  if (mConnection != NULL)
    {
    delete mConnection;
    }
  mConnection = new ULConnection(connectInfo);

  mConnection->GetTimer().SetTimeout(inTimeout);

  // Warning PresentationContextID is important
  // this is a sort of uniq key used by the recevier. Eg.
  // if one push_pack
  //  (1, Secondary)
  //  (1, Verification)
  // Then the last one is prefered (DCMTK 3.5.5)

  // The following only works for C-STORE / C-ECHO
  // however it does not make much sense to add a lot of abstract syntax
  // when doing only C-ECHO.
  // FIXME is there a way to know here if we are in C-ECHO ?
  //there is now!
  //the presentation context will now be part of the connection, so that this
  //initialization for the association-rq will use parameters from the connection


#if 0
  AbstractSyntax as;
  std::vector<PresentationContextRQ> pcVector;
  PresentationContextRQ pc;
  TransferSyntaxSub ts;
  ts.SetNameFromUID( UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM );
  pc.AddTransferSyntax( ts );
  ts.SetNameFromUID( UIDs::ExplicitVRLittleEndian );
  //ts.SetNameFromUID( UIDs::JPEGLosslessNonHierarchicalFirstOrderPredictionProcess14SelectionValue1DefaultTransferSyntaxforLosslessJPEGImageCompression);
  //pc.AddTransferSyntax( ts ); // we do not support explicit (mm)
  switch (inConnectionType){
    case eEcho:
        pc.SetPresentationContextID( eVerificationSOPClass );
        as.SetNameFromUID( UIDs::VerificationSOPClass );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
      break;
    case eFind:
        pc.SetPresentationContextID( ePatientRootQueryRetrieveInformationModelFIND );
        as.SetNameFromUID( UIDs::PatientRootQueryRetrieveInformationModelFIND );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
        pc.SetPresentationContextID(eStudyRootQueryRetrieveInformationModelFIND );
        as.SetNameFromUID( UIDs::StudyRootQueryRetrieveInformationModelFIND );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
        pc.SetPresentationContextID( ePatientStudyOnlyQueryRetrieveInformationModelFINDRetired );
        as.SetNameFromUID( UIDs::PatientStudyOnlyQueryRetrieveInformationModelFINDRetired );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
        pc.SetPresentationContextID( eModalityWorklistInformationModelFIND );
        as.SetNameFromUID( UIDs::ModalityWorklistInformationModelFIND );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
        pc.SetPresentationContextID( eGeneralPurposeWorklistInformationModelFIND );
        as.SetNameFromUID( UIDs::GeneralPurposeWorklistInformationModelFIND );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
      break;
      //our spec does not require C-GET support
//    case eGet:
//      break;
/*    case eMove:
        // should we also send stuff from FIND ?
        // E: Move PresCtx but no Find (accepting for now)
        pc.SetPresentationContextID( ePatientRootQueryRetrieveInformationModelFIND );
        as.SetNameFromUID( UIDs::PatientRootQueryRetrieveInformationModelFIND );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
        // move
        pc.SetPresentationContextID( ePatientRootQueryRetrieveInformationModelMOVE );
        as.SetNameFromUID( UIDs::PatientRootQueryRetrieveInformationModelMOVE );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
        pc.SetPresentationContextID( eStudyRootQueryRetrieveInformationModelFIND );
        as.SetNameFromUID( UIDs::StudyRootQueryRetrieveInformationModelFIND );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
        pc.SetPresentationContextID( eStudyRootQueryRetrieveInformationModelMOVE );
        as.SetNameFromUID( UIDs::StudyRootQueryRetrieveInformationModelMOVE );
        pc.SetAbstractSyntax( as );
        pcVector.push_back(pc);
      break;*/
    case eStore:
      std::string uidName;
        pc.SetPresentationContextID( PresentationContextRQ::AssignPresentationContextID(inDS, uidName) );
        if (pc.GetPresentationContextID() != eVerificationSOPClass){
          as.SetNameFromUIDString( uidName );
          pc.SetAbstractSyntax( as );
          pcVector.push_back(pc);
        }
      break;
  }
  if (pcVector.empty()){
    gdcmWarningMacro("Unable to establish presentation context; ensure that dataset has tags 0x8,0x16 and 0x8,0x18 defined." <<std::endl);
    return false;
  }
#endif
  mConnection->SetPresentationContexts(pcVector);


  //now, try to establish a connection by starting the transition table and the event loop.
  //here's the thing
  //if there's nothing on the event loop, assume that it's done & the function can exit.
  //otherwise, keep rolling the event loop
  ULEvent theEvent(eAASSOCIATERequestLocalUser, NULL);
  //no callback, assume that no data is transferred back, because there shouldn't be any
  EStateID theState = RunEventLoop(theEvent, mConnection, NULL, false);

  if(theState != eSta6TransferReady)
    {
    std::vector<BasePDU*> const & thePDUs = theEvent.GetPDUs();
    for( std::vector<BasePDU*>::const_iterator itor
      = thePDUs.begin(); itor != thePDUs.end(); itor++)
      {
      //assert(*itor);
      if (*itor == NULL) continue; //can have a nulled pdu, apparently
      (*itor)->Print(Trace::GetErrorStream());
      }
    }
  else if (Trace::GetDebugFlag())
    {
    std::vector<BasePDU*> const & thePDUs = theEvent.GetPDUs();
    for( std::vector<BasePDU*>::const_iterator itor
      = thePDUs.begin(); itor != thePDUs.end(); itor++)
      {
      assert(*itor);
      if (*itor == NULL) continue; //can have a nulled pdu, apparently
      (*itor)->Print(Trace::GetDebugStream());
      }
    }

  return (theState == eSta6TransferReady);//ie, finished the transitions
}



/// returns true for above reasons, but contains the special 'move' port
bool ULConnectionManager::EstablishConnectionMove(const std::string& inAETitle,
  const std::string& inConnectAETitle,
  const std::string& inComputerName, long inIPAddress,
  uint16_t inConnectPort, double inTimeout,
  uint16_t inReturnPort,
  std::vector<PresentationContext> const & pcVector)
{
  gdcmDebugMacro( "Start EstablishConnectionMove" );
  //generate a ULConnectionInfo object
  UserInformation userInfo;
  ULConnectionInfo connectInfo;
  if (inConnectAETitle.size() > 16) return false;//too long an AETitle, probably need better failure message
  if (inAETitle.size() > 16) return false; //as above
  if (!connectInfo.Initialize(userInfo,inAETitle.c_str(), inConnectAETitle.c_str(),
      inIPAddress, inReturnPort, inComputerName))
    {
    gdcmDebugMacro( "Could not Initialize connectInfo" );
    return false;
    }
  gdcmDebugMacro( "SCP: First connection established on port " << inReturnPort );

  if (mSecondaryConnection != NULL)
    {
    gdcmDebugMacro( "delete mSecondaryConnection" );
    delete mSecondaryConnection;
    }
  mSecondaryConnection = new ULConnection(connectInfo);
  mSecondaryConnection->GetTimer().SetTimeout(inTimeout);

  //generate a ULConnectionInfo object
  UserInformation userInfo2;
  ULConnectionInfo connectInfo2;
  if (inConnectAETitle.size() > 16) return false;//too long an AETitle, probably need better failure message
  if (inAETitle.size() > 16) return false; //as above
  if (!connectInfo2.Initialize(userInfo2, inConnectAETitle.c_str(),
      inAETitle.c_str(), inIPAddress, inConnectPort, inComputerName))
    {
    gdcmDebugMacro( "Could not Initialize connectInfo2" );
    return false;
    }
  gdcmDebugMacro( "SCU: Second connection established on port " << inConnectPort );

  if (mConnection!= NULL)
    {
    gdcmDebugMacro( "delete mConnection" );
    delete mConnection;
    }
  mConnection = new ULConnection(connectInfo2);
  mConnection->GetTimer().SetTimeout(inTimeout);


  // Warning PresentationContextID is important
  // this is a sort of uniq key used by the recevier. Eg.
  // if one push_pack
  //  (1, Secondary)
  //  (1, Verification)
  // Then the last one is prefered (DCMTK 3.5.5)

  // The following only works for C-STORE / C-ECHO
  // however it does not make much sense to add a lot of abstract syntax
  // when doing only C-ECHO.
  // FIXME is there a way to know here if we are in C-ECHO ?
  //there is now!
  //the presentation context will now be part of the connection, so that this
  //initialization for the association-rq will use parameters from the connection

  AbstractSyntax as;

#if 0
  std::vector<PresentationContextRQ> pcVector;
  PresentationContextRQ pc;
  TransferSyntaxSub ts;
  ts.SetNameFromUID( UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM );
  pc.AddTransferSyntax( ts );
  ts.SetNameFromUID( UIDs::ExplicitVRLittleEndian );
  //pc.AddTransferSyntax( ts ); // we do not support explicit (mm)
  // should we also send stuff from FIND ?
  // E: Move PresCtx but no Find (accepting for now)
  pc.SetPresentationContextID( ePatientRootQueryRetrieveInformationModelFIND );
  as.SetNameFromUID( UIDs::PatientRootQueryRetrieveInformationModelFIND );
  pc.SetAbstractSyntax( as );
  pcVector.push_back(pc);
  // move
  pc.SetPresentationContextID( ePatientRootQueryRetrieveInformationModelMOVE );
  as.SetNameFromUID( UIDs::PatientRootQueryRetrieveInformationModelMOVE );
  pc.SetAbstractSyntax( as );
  pcVector.push_back(pc);
  pc.SetPresentationContextID( eStudyRootQueryRetrieveInformationModelFIND );
  as.SetNameFromUID( UIDs::StudyRootQueryRetrieveInformationModelFIND );
  pc.SetAbstractSyntax( as );
  pcVector.push_back(pc);
  pc.SetPresentationContextID( eStudyRootQueryRetrieveInformationModelMOVE );
  as.SetNameFromUID( UIDs::StudyRootQueryRetrieveInformationModelMOVE );
  pc.SetAbstractSyntax( as );
  pcVector.push_back(pc);
#endif
  mConnection->SetPresentationContexts(pcVector);


  //now, try to establish a connection by starting the transition table and the event loop.
  //here's the thing
  //if there's nothing on the event loop, assume that it's done & the function can exit.
  //otherwise, keep rolling the event loop
  ULEvent theEvent(eAASSOCIATERequestLocalUser, NULL);
  std::vector<DataSet> empty;
  //No data should be returned when connections are established
  EStateID theState = RunEventLoop(theEvent, mConnection, NULL, false);

  if (Trace::GetDebugFlag())
    {
      std::vector<BasePDU*> thePDUs = theEvent.GetPDUs();
      std::vector<BasePDU*>::iterator itor;
      for (itor = thePDUs.begin(); itor != thePDUs.end(); itor++)
        {
        if (*itor == NULL) continue; //can have a nulled pdu, apparently
        (*itor)->Print(Trace::GetStream());
        }
    }

  return (theState == eSta6TransferReady);//ie, finished the transitions
}

//send the Data PDU associated with Echo (ie, a default DataPDU)
//this lets the user confirm that the connection is alive.
//the user should look to cout to see the response of the echo command
std::vector<PresentationDataValue> ULConnectionManager::SendEcho(){

  std::vector<BasePDU*> theDataPDU = PDUFactory::CreateCEchoPDU(*mConnection);//pass NULL for C-Echo
  ULEvent theEvent(ePDATArequest, theDataPDU);

  EStateID theState = RunEventLoop(theEvent, mConnection, NULL, false);
  //theEvent should contain the PDU for the echo!

  if (theState == eSta6TransferReady){//ie, finished the transitions
    return PDUFactory::GetPDVs(theEvent.GetPDUs());
  } else {
    std::vector<PresentationDataValue> empty;
    return empty;
  }
}

std::vector<DataSet> ULConnectionManager::SendMove(const BaseRootQuery* inRootQuery)
{
  ULBasicCallback theCallback;
  SendMove(inRootQuery, &theCallback);
  return theCallback.GetDataSets();
}

bool ULConnectionManager::SendMove(const BaseRootQuery* inRootQuery,
  ULConnectionCallback* inCallback)
{
  if (mConnection == NULL)
    {
    gdcmErrorMacro( "mConnection is NULL" );
    return false;
    }
  std::vector<BasePDU*> theDataPDU = PDUFactory::CreateCMovePDU( *mConnection, inRootQuery );
  ULEvent theEvent(ePDATArequest, theDataPDU);
  EStateID stateid = RunMoveEventLoop(theEvent, inCallback);
  gdcmDebugMacro( "Final StateID: " << (int) stateid );
  return stateid == gdcm::network::eSta6TransferReady;
}

std::vector<DataSet> ULConnectionManager::SendFind(const BaseRootQuery* inRootQuery)
{
  ULBasicCallback theCallback;
  SendFind(inRootQuery, &theCallback);
  return theCallback.GetDataSets();
}

void ULConnectionManager::SendFind(const BaseRootQuery* inRootQuery, ULConnectionCallback* inCallback)
{
  if (mConnection == NULL)
    {
    return;
    }
  std::vector<BasePDU*> theDataPDU = PDUFactory::CreateCFindPDU( *mConnection, inRootQuery );
  ULEvent theEvent(ePDATArequest, theDataPDU);
  RunEventLoop(theEvent, mConnection, inCallback, false);
}

std::vector<DataSet> ULConnectionManager::SendStore(const File &file)
{
  ULBasicCallback theCallback;
  SendStore(file, &theCallback);
  return theCallback.GetResponses();
}

void ULConnectionManager::SendStore(const File & file, ULConnectionCallback* inCallback)
{
  if (mConnection == NULL)
    {
    return;
    }
  std::vector<BasePDU*> theDataPDU = PDUFactory::CreateCStoreRQPDU(*mConnection, file);
  const DataSet* inDataSet = &file.GetDataSet();
  DataSetEvent dse( inDataSet );
  this->InvokeEvent( dse );

  ULEvent theEvent(ePDATArequest, theDataPDU);
  EStateID theState = RunEventLoop(theEvent, mConnection, inCallback, false);
  assert( theState == eSta6TransferReady || theState == eStaDoesNotExist ); (void)theState;
}

bool ULConnectionManager::BreakConnection(const double& inTimeOut){
  std::vector<DataSet> theResult;
  if (mConnection == NULL){
    return false;
  }
  BasePDU* thePDU = PDUFactory::ConstructReleasePDU();
  ULEvent theEvent(eARELEASERequest, thePDU);
  mConnection->GetTimer().SetTimeout(inTimeOut);

  //assume no data coming back when dying, no need for callback
  EStateID theState = RunEventLoop(theEvent, mConnection, NULL, false);

  return (theState == eSta1Idle);//ie, finished the transitions
}

void ULConnectionManager::BreakConnectionNow(){
  BasePDU* thePDU = PDUFactory::ConstructAbortPDU();
  ULEvent theEvent(eAABORTRequest, thePDU);

  //assume no data coming back when dying, no need for callback
  EStateID theState = RunEventLoop(theEvent, mConnection, NULL, false);
  (void)theState;
}
  
//event handler loop for move-- will interweave the two event loops,
//one for storescp and the other for movescu.  Perhaps complicated, but
//avoids starting a second process.
EStateID ULConnectionManager::RunMoveEventLoop(ULEvent& currentEvent, ULConnectionCallback* inCallback){
  gdcmDebugMacro( "Start RunMoveEventLoop" );
  EStateID theState = eStaDoesNotExist;
  bool waitingForEvent;
  EEventID raisedEvent;

  bool receivingData = false;
  bool justWaiting = false;
  //when receiving data from a find, etc, then justWaiting is true and only receiving is done
  //eventually, could add cancel into the mix... but that would be through a callback or something similar
  do {
    gdcmDebugMacro( "Before mTransitions.HandleEvent" );
    if (!justWaiting){
      mTransitions.HandleEvent(this,currentEvent, *mConnection, waitingForEvent, raisedEvent);
    }

    theState = mConnection->GetState();
    std::istream &is = *mConnection->GetProtocol();
    //std::ostream &os = *mConnection->GetProtocol();


    //When doing a C-MOVE we receive the Requested DataSet over
    //another channel (technically this is send to an SCP)
    //in our case we use another port to receive it.
    EStateID theCStoreStateID = eSta6TransferReady;
    bool secondConnectionEstablished = false;
    gdcmDebugMacro( "Before mSecondaryConnection.GetProtocol" );
    if (mSecondaryConnection->GetProtocol() == NULL){
      //establish the connection
      //can fail if is_readready doesn't return true, ie, the connection
      //wasn't opened on the other side because the other side isn't sending data yet
      //for whatever reason (maybe there's nothing to get?)
      gdcmDebugMacro( "Before mSecondaryConnection.InitializeIncomingConnection" );
      secondConnectionEstablished =
        mSecondaryConnection->InitializeIncomingConnection();
    }
    gdcmDebugMacro( "After mSecondaryConnection.InitializeIncomingConnection: " <<
      "secondConnectionEstablished=" << secondConnectionEstablished <<
      " GetState() =" << (int)mSecondaryConnection->GetState()
    );
    if (!secondConnectionEstablished )
      {
      gdcmErrorMacro( "Could not establish 2nd connection" );
      //return eStaDoesNotExist;
      }
    if (secondConnectionEstablished &&
      (mSecondaryConnection->GetState()== eSta1Idle ||
      mSecondaryConnection->GetState() == eSta2Open)){
      ULEvent theCStoreEvent(eEventDoesNotExist, NULL);//have to fill this in, we're in passive mode now
      theCStoreStateID = RunEventLoop(theCStoreEvent, mSecondaryConnection, inCallback, true);
    }
    gdcmDebugMacro( "After mSecondaryConnection / RunEventLoop: " << (int)theCStoreStateID );

    //just as for the regular event loop, but we have to alternate between the connections.
    //it may be that nothing comes back over the is connection, but lots over the
    //isSCP connection.  So, if is fails, meh.  But if isSCP fails, that's not so meh.
    //we care only about the datasets coming back from isSCP, ultimately, though the datasets
    //from is will contain progress info.
    std::vector<BasePDU*> incomingPDUs;
    if (waitingForEvent){
      while (waitingForEvent)
        {//loop for reading in the events that come down the wire
        uint8_t itemtype = 0x0;
        gdcmDebugMacro( "Waiting for ItemType (#2)" );
        is.read( (char*)&itemtype, 1 );

        BasePDU* thePDU = PDUFactory::ConstructPDU(itemtype);
        if (thePDU != NULL)
          {
          incomingPDUs.push_back(thePDU);
          thePDU->Read(is);
          gdcmDebugMacro("PDU code: " << static_cast<int>(itemtype) << std::endl);
          if (Trace::GetDebugFlag())
            {
            thePDU->Print(Trace::GetStream());
            }
          if (thePDU->IsLastFragment()) waitingForEvent = false;
          }
        else
          {
          waitingForEvent = false; //because no PDU means not waiting anymore
          }
        }
      //now, we have to figure out the event that just happened based on the PDU that was received.
      if (!incomingPDUs.empty())
        {
        currentEvent.SetEvent(PDUFactory::DetermineEventByPDU(incomingPDUs[0]));
        currentEvent.SetPDU(incomingPDUs);
        if (mConnection->GetTimer().GetHasExpired())
          {
          currentEvent.SetEvent(eARTIMTimerExpired);
          }
        if (theState == eSta6TransferReady){//ie, finished the transitions
          //with find, the results now come down the wire.
          //the pdu we already have from the event will tell us how many to expect.
          uint32_t pendingDE1, pendingDE2, success, theVal;
          pendingDE1 = 0xff01;
          pendingDE2 = 0xff00;
          success = 0x0000;
          theVal = pendingDE1;
          uint32_t theNumLeft = 0; // the number of pending sub operations left.
          //so here's the thing: dcmtk responds with 'success' as it first cmove rsp
          //which is retarded and, I think, wrong.  However, dcm4chee responds with 'pending'
          //so, we look either for pending, or for the number of operations left
          // (tag 0000, 1020) if the value is success, and that number should be 0.
          DataSet theRSP = PresentationDataValue::ConcatenatePDVBlobs(PDUFactory::GetPDVs(currentEvent.GetPDUs()));
          if (Trace::GetDebugFlag())
            {
            Printer thePrinter;
            Trace::GetStream() << "Response: " << std::endl;
            thePrinter.PrintDataSet(theRSP, Trace::GetStream());
            Trace::GetStream() << std::endl;
            }
          if (theRSP.FindDataElement(Tag(0x0, 0x0800))){
            DataElement const & de = theRSP.GetDataElement(Tag(0x0,0x0800));
            Attribute<0x0,0x0800> at;
            at.SetFromDataElement( de );
            unsigned short datasettype = at.GetValue();
            assert( datasettype == 0x0101 || datasettype == 0x1 ); (void)datasettype;
          }
          if (theRSP.FindDataElement(Tag(0x0, 0x0900))){
            DataElement const & de = theRSP.GetDataElement(Tag(0x0,0x0900));
            Attribute<0x0,0x0900> at;
            at.SetFromDataElement( de );
            theVal = at.GetValues()[0];
            //if theVal is Pending or Success, then we need to enter the loop below,
            //because we need the data PDUs.
            //so, the loop below is a do/while loop; there should be at least a second packet
            //with the dataset, even if the status is 'success'
            //success == 0000H
          }
          uint32_t theCommandCode = 0;
          if (theRSP.FindDataElement(Tag(0x0,0x0100))){
            DataElement const & de = theRSP.GetDataElement(Tag(0x0,0x0100));
            Attribute<0x0,0x0100> at;
            at.SetFromDataElement( de );
            theCommandCode = at.GetValues()[0];
          }
          if (theRSP.FindDataElement(Tag(0x0, 0x1020))){
            DataElement de = theRSP.GetDataElement(Tag(0x0,0x1020));
            Attribute<0x0,0x1020> at;
            at.SetFromDataElement( de );
            theNumLeft = at.GetValues()[0];
            //if theVal is Pending or Success, then we need to enter the loop below,
            //because we need the data PDUs.
            //so, the loop below is a do/while loop; there should be at least a second packet
            //with the dataset, even if the status is 'success'
            //success == 0000H
          }
          if (theVal != pendingDE1 && theVal != pendingDE2 && theVal != success){
            //check for other error fields
            const ByteValue *err1 = NULL, *err2 = NULL;
            gdcmErrorMacro( "Transfer failed with code " << theVal << std::endl);
            switch (theVal){
              case 0xA701:
                gdcmErrorMacro( "Refused: Out of Resources Unable to calculate number of matches" << std::endl);
                break;
              case 0xA702:
                gdcmErrorMacro( "Refused: Out of Resources Unable to perform sub-operations" << std::endl);
                break;
              case 0xA801:
                gdcmErrorMacro( "Refused: Move Destination unknown" << std::endl);
                break;
              case 0xA900:
                gdcmErrorMacro( "Identifier does not match SOP Class" << std::endl);
                break;
              case 0xAA00:
                gdcmErrorMacro( "None of the frames requested were found in the SOP Instance" << std::endl);
                break;
              case 0xAA01:
                gdcmErrorMacro( "Unable to create new object for this SOP class" << std::endl);
                break;
              case 0xAA02:
                gdcmErrorMacro( "Unable to extract frames" << std::endl);
                break;
              case 0xAA03:
                gdcmErrorMacro( "Time-based request received for a non-time-based original SOP Instance. " << std::endl);
                break;
              case 0xAA04:
                gdcmErrorMacro( "Invalid Request" << std::endl);
                break;
              case 0xFE00:
                gdcmErrorMacro( "Sub-operations terminated due to Cancel Indication" << std::endl);
                break;
              case 0xB000:
                gdcmErrorMacro( "Sub-operations Complete One or more Failures or Warnings" << std::endl);
                break;
              default:
                gdcmErrorMacro( "Unable to process" << std::endl);
                break;
            }
            if (theRSP.FindDataElement(Tag(0x0,0x0901))){
              DataElement de = theRSP.GetDataElement(Tag(0x0,0x0901));
              err1 = de.GetByteValue();
              gdcmErrorMacro( " Tag 0x0,0x901 reported as " << *err1 << std::endl); (void)err1;
            }
            if (theRSP.FindDataElement(Tag(0x0,0x0902))){
              DataElement de = theRSP.GetDataElement(Tag(0x0,0x0902));
              err2 = de.GetByteValue();
              gdcmErrorMacro( " Tag 0x0,0x902 reported as " << *err2 << std::endl); (void)err2;
            }
          }
          receivingData = false;
          justWaiting = false;
          if (theVal == pendingDE1 || theVal == pendingDE2 || (theVal == success && theNumLeft != 0)) {
            receivingData = true; //wait for more data as more PDUs (findrsps, for instance)
            justWaiting = true;
            waitingForEvent = true;

            //ok, if we're pending, then let's open the cstorescp connection here
            //(if it's not already open), and then from here start a storescp event loop.
            //just don't listen to the cmove event loop until this is done.
            //could cause a pileup on the main connection, I suppose.
            //could also report the progress here, if we liked.
            if (theCommandCode == 0x8021){//cmove response, so prep the retrieval loop on the back connection

              bool dataSetCountIncremented = true;//false once the number of incoming datasets doesn't change.
              if (mSecondaryConnection->GetProtocol() == NULL){
                //establish the connection
                //can fail if is_readready doesn't return true, ie, the connection
                //wasn't opened on the other side because the other side isn't sending data yet
                //for whatever reason (maybe there's nothing to get?)
                secondConnectionEstablished =
                  mSecondaryConnection->InitializeIncomingConnection();
                if (secondConnectionEstablished &&
                  (mSecondaryConnection->GetState()== eSta1Idle ||
                  mSecondaryConnection->GetState() == eSta2Open)){
                  ULEvent theCStoreEvent(eEventDoesNotExist, NULL);//have to fill this in, we're in passive mode now
                  theCStoreStateID = RunEventLoop(theCStoreEvent, mSecondaryConnection, inCallback, true);
                } else {//something broke, can't establish secondary move connection here
                  gdcmErrorMacro( "Unable to establish secondary connection with server, aborting." << std::endl);
                  return eStaDoesNotExist;
                }
              }
              if (secondConnectionEstablished){
                while (theCStoreStateID == eSta6TransferReady && dataSetCountIncremented){
                  ULEvent theCStoreEvent(eEventDoesNotExist, NULL);//have to fill this in, we're in passive mode now
                  //now, get data from across the network
                  theCStoreStateID = RunEventLoop(theCStoreEvent, mSecondaryConnection, inCallback, true);
                  if (inCallback){
                    dataSetCountIncremented = true;
                    inCallback->ResetHandledDataSet();
                  } else {
                    dataSetCountIncremented = false;
                  }
                }
              }
              //force the abort from our side
            //  ULEvent theCStoreEvent(eAABORTRequest, NULL);//have to fill this in, we're in passive mode now
            //  theCStoreStateID = RunEventLoop(theCStoreEvent, outDataSet, mSecondaryConnection, true);
            } else {//not dealing with cmove progress updates, apparently
              //keep looping if we haven't succeeded or failed; these are the values for 'pending'
              //first, dynamically cast that pdu in the event
              //should be a data pdu
              //then, look for tag 0x0,0x900

              //only add datasets that are _not_ part of the network response
              std::vector<DataSet> final;
              std::vector<BasePDU*> theData;
              BasePDU* thePDU;//outside the loop for the do/while stopping condition
              bool interrupted = false;
              do {
                uint8_t itemtype = 0x0;
                is.read( (char*)&itemtype, 1 );
                //what happens if nothing's read?
                thePDU = PDUFactory::ConstructPDU(itemtype);
                if (itemtype != 0x4 && thePDU != NULL){ //ie, not a pdatapdu
                  std::vector<BasePDU*> interruptingPDUs;
                  currentEvent.SetEvent(PDUFactory::DetermineEventByPDU(interruptingPDUs[0]));
                  currentEvent.SetPDU(interruptingPDUs);
                  interrupted= true;
                  break;
                }
                if (thePDU != NULL){
                  thePDU->Read(is);
                  theData.push_back(thePDU);
                } else{
                  break;
                }
                //!!!need to handle incoming PDUs that are not data, ie, an abort
              } while(/*!is.eof() &&*/ !thePDU->IsLastFragment());
              if (!interrupted){//ie, if the remote server didn't hang up
                DataSet theCompleteFindResponse =
                  PresentationDataValue::ConcatenatePDVBlobs(PDUFactory::GetPDVs(theData));
                //note that it's the responsibility of the event to delete the PDU in theFindRSP
                for (size_t i = 0; i < theData.size(); i++){
                  delete theData[i];
                }
                //outDataSet.push_back(theCompleteFindResponse);
                if (inCallback){
                  inCallback->HandleDataSet(theCompleteFindResponse);
                }
              }
            }
          }
        }
      } else {
        raisedEvent = eEventDoesNotExist;
        waitingForEvent = false;
      }
    }
    else {
      currentEvent.SetEvent(raisedEvent);//actions that cause transitions in the state table
      //locally just raise local events that will therefore cause the trigger to be pulled.
    }
  } while (currentEvent.GetEvent() != eEventDoesNotExist &&
    theState != eStaDoesNotExist && theState != eSta13AwaitingClose && theState != eSta1Idle &&
    (theState != eSta6TransferReady || (theState == eSta6TransferReady && receivingData )));
  //stop when the AE is done, or when ready to transfer data (ie, the next PDU should be sent in),
  //or when the connection is idle after a disconnection.
  //or, if in state 6 and receiving data, until all data is received.

  return theState;
}


//event handler loop.
//will just keep running until the current event is nonexistent.
//at which point, it will return the current state of the connection
//to do this, execute an event, and then see if there's a response on the
//incoming connection (with a reasonable amount of timeout).
//if no response, assume that the connection is broken.
//if there's a response, then yay.
//note that this is the ARTIM timeout event
EStateID ULConnectionManager::RunEventLoop(ULEvent& currentEvent, ULConnection* inWhichConnection,
                                           ULConnectionCallback* inCallback, const bool& startWaiting = false){
  gdcmDebugMacro( "Start RunEventLoop" );
  EStateID theState = eStaDoesNotExist;
  bool waitingForEvent = startWaiting;//overwritten if not starting waiting, but if waiting, then wait
  EEventID raisedEvent;

  bool receivingData = false;
  //bool justWaiting = startWaiting;
  //not sure justwaiting is useful; for now, go back to waiting for event

  //when receiving data from a find, etc, then justWaiting is true and only receiving is done
  //eventually, could add cancel into the mix... but that would be through a callback or something similar
  do {
    gdcmDebugMacro( "Before mTransitions.HandleEvent #2" );
    raisedEvent = eEventDoesNotExist;
    if (!waitingForEvent){//justWaiting){
      mTransitions.HandleEvent(this, currentEvent, *inWhichConnection, waitingForEvent, raisedEvent);
      //this gathering of the state is for scus that have just sent out a request
      theState = inWhichConnection->GetState();
    }
    std::istream &is = *inWhichConnection->GetProtocol();
    //std::ostream &os = *inWhichConnection->GetProtocol();

    BasePDU* theFirstPDU = NULL;// the first pdu read in during this event loop,
    //used to make sure the presentation context ID is correct

    //read the connection, as that's an event as well.
    //waiting for an object to come back across the connection, so that it can get handled.
    //ie, accept, reject, timeout, etc.
    //of course, if the connection is down, just leave the loop.
    //also leave the loop if nothing's waiting.
    //use the PDUFactory to create the appropriate pdu, which has its own
    //internal mechanisms for handling itself (but will, of course, be put inside the event object).
    //but, and here's the important thing, only read on the socket when we should.
    std::vector<BasePDU*> incomingPDUs;
    if (waitingForEvent){
      while (waitingForEvent){//loop for reading in the events that come down the wire
        uint8_t itemtype = 0x0;
        try {
          gdcmDebugMacro( "Waiting for ItemType" );
          is.read( (char*)&itemtype, 1 );
          gdcmDebugMacro( "Received ItemType #" << (int)itemtype );
          //what happens if nothing's read?
          theFirstPDU = PDUFactory::ConstructPDU(itemtype);
          if (theFirstPDU != NULL){
            incomingPDUs.push_back(theFirstPDU);
            theFirstPDU->Read(is);
            gdcmDebugMacro("PDU code: " << static_cast<int>(itemtype) << std::endl);
            if (Trace::GetDebugFlag())
              {
              theFirstPDU->Print(Trace::GetStream());
              }

            if (theFirstPDU->IsLastFragment()) waitingForEvent = false;
          } else {
            gdcmDebugMacro( "NULL theFirstPDU for ItemType" << (int)itemtype );
            waitingForEvent = false; //because no PDU means not waiting anymore
            return eStaDoesNotExist;
          }
        }
        catch (...)
          {
          //handle the exception, which is basically that nothing came in over the pipe.
          gdcmAssertAlwaysMacro( 0 );
          }
      }
      //now, we have to figure out the event that just happened based on the PDU that was received.
      //this state gathering is for scps, especially the cstore for cmove.
      theState = inWhichConnection->GetState();
      if (!incomingPDUs.empty()){
        currentEvent.SetEvent(PDUFactory::DetermineEventByPDU(incomingPDUs[0]));
        currentEvent.SetPDU(incomingPDUs);
        //here's the scp handling code
        if (mConnection->GetTimer().GetHasExpired()){
          currentEvent.SetEvent(eARTIMTimerExpired);
        }
        switch(currentEvent.GetEvent()){
          case ePDATATFPDU:
            {
            //if (theState == eSta6TransferReady){//ie, finished the transitions
              //with find, the results now come down the wire.
              //the pdu we already have from the event will tell us how many to expect.
              uint32_t pendingDE1, pendingDE2, success, theVal;
              pendingDE1 = 0xff01;
              pendingDE2 = 0xff00;
              success = 0x0000;
              theVal = pendingDE1;
              uint32_t theCommandCode = 0;//for now, a nothing value
              DataSet theRSP =
                PresentationDataValue::ConcatenatePDVBlobs(
                  PDUFactory::GetPDVs(currentEvent.GetPDUs()));
              if (inCallback)
                {
                inCallback->HandleResponse(theRSP);
                }

              if (theRSP.FindDataElement(Tag(0x0, 0x0900))){
                DataElement de = theRSP.GetDataElement(Tag(0x0,0x0900));
                Attribute<0x0,0x0900> at;
                at.SetFromDataElement( de );
                theVal = at.GetValues()[0];
                //if theVal is Pending or Success, then we need to enter the loop below,
                //because we need the data PDUs.
                //so, the loop below is a do/while loop; there should be at least a second packet
                //with the dataset, even if the status is 'success'
                //success == 0000H
              }
              if (Trace::GetDebugFlag())
                {
                Printer thePrinter;
                thePrinter.PrintDataSet(theRSP, Trace::GetStream());
              }

              //check to see if this is a cstorerq
              if (theRSP.FindDataElement(Tag(0x0, 0x0100)))
                {
                DataElement de2 = theRSP.GetDataElement(Tag(0x0,0x0100));
                Attribute<0x0,0x0100> at2;
                at2.SetFromDataElement( de2 );
                theCommandCode = at2.GetValues()[0];
                }

              if (theVal != pendingDE1 && theVal != pendingDE2 && theVal != success)
                {
                //check for other error fields
                const ByteValue *err1 = NULL, *err2 = NULL;
                gdcmErrorMacro( "Transfer failed with code " << theVal << std::endl);
                switch (theVal){
                  case 0xA701:
                    gdcmErrorMacro( "Refused: Out of Resources Unable to calculate number of matches" << std::endl);
                    break;
                  case 0xA702:
                    gdcmErrorMacro( "Refused: Out of Resources Unable to perform sub-operations" << std::endl);
                    break;
                  case 0xA801:
                    gdcmErrorMacro( "Refused: Move Destination unknown" << std::endl);
                    break;
                  case 0xA900:
                    gdcmErrorMacro( "Identifier does not match SOP Class" << std::endl);
                    break;
                  case 0xAA00:
                    gdcmErrorMacro( "None of the frames requested were found in the SOP Instance" << std::endl);
                    break;
                  case 0xAA01:
                    gdcmErrorMacro( "Unable to create new object for this SOP class" << std::endl);
                    break;
                  case 0xAA02:
                    gdcmErrorMacro( "Unable to extract frames" << std::endl);
                    break;
                  case 0xAA03:
                    gdcmErrorMacro( "Time-based request received for a non-time-based original SOP Instance. " << std::endl);
                    break;
                  case 0xAA04:
                    gdcmErrorMacro( "Invalid Request" << std::endl);
                    break;
                  case 0xFE00:
                    gdcmErrorMacro( "Sub-operations terminated due to Cancel Indication" << std::endl);
                    break;
                  case 0xB000:
                    gdcmErrorMacro( "Sub-operations Complete One or more Failures or Warnings" << std::endl);
                    break;
                  default:
                    gdcmErrorMacro( "Unable to process" << std::endl);
                    break;
                }
                if (theRSP.FindDataElement(Tag(0x0,0x0901))){
                  DataElement de = theRSP.GetDataElement(Tag(0x0,0x0901));
                  err1 = de.GetByteValue();
                  gdcmErrorMacro( " Tag 0x0,0x901 reported as " << *err1 << std::endl); (void)err1;
                }
                if (theRSP.FindDataElement(Tag(0x0,0x0902))){
                  DataElement de = theRSP.GetDataElement(Tag(0x0,0x0902));
                  err2 = de.GetByteValue();
                  gdcmErrorMacro( " Tag 0x0,0x902 reported as " << *err2 << std::endl); (void)err2;
                }
              }

              receivingData = false;
              //justWaiting = false;
              if (theVal == pendingDE1 || theVal == pendingDE2) {
                receivingData = true; //wait for more data as more PDUs (findrsps, for instance)
                //justWaiting = true;
                waitingForEvent = true;
              }
              if (theVal == pendingDE1 || theVal == pendingDE2 /*|| theVal == success*/){//keep looping if we haven't succeeded or failed; these are the values for 'pending'
                //first, dynamically cast that pdu in the event
                //should be a data pdu
                //then, look for tag 0x0,0x900

                //only add datasets that are _not_ part of the network response
                std::vector<DataSet> final;
                std::vector<BasePDU*> theData;
                BasePDU* thePDU;//outside the loop for the do/while stopping condition
                bool interrupted = false;
                do {
                  uint8_t itemtype = 0x0;
                  is.read( (char*)&itemtype, 1 );
                  //what happens if nothing's read?
                  thePDU = PDUFactory::ConstructPDU(itemtype);
                  if (itemtype != 0x4 && thePDU != NULL){ //ie, not a pdatapdu
                    std::vector<BasePDU*> interruptingPDUs;
                    interruptingPDUs.push_back(thePDU);
                    currentEvent.SetEvent(PDUFactory::DetermineEventByPDU(interruptingPDUs[0]));
                    currentEvent.SetPDU(interruptingPDUs);
                    interrupted= true;
                    break;
                  }
                  if (thePDU != NULL){
                    thePDU->Read(is);
                    theData.push_back(thePDU);
                  } else{
                    break;
                  }
                  //!!!need to handle incoming PDUs that are not data, ie, an abort
                } while(!thePDU->IsLastFragment());
                if (!interrupted){//ie, if the remote server didn't hang up
                  bool useimplicit = true;
                  TransferSyntaxSub ts1;
                  ts1.SetNameFromUID( UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM );
                  if( mSecondaryConnection )
                    {
                    const TransferSyntaxSub & ts_ = mSecondaryConnection->GetCStoreTransferSyntax();
                    if( strcmp(ts_.GetName(), ts1.GetName()) != 0)
                      {
                      useimplicit = false;
                      }
                    }
                  DataSet theCompleteFindResponse;
                  if( useimplicit )
                    {
                    inCallback->SetImplicitFlag(true);
                    theCompleteFindResponse =
                      PresentationDataValue::ConcatenatePDVBlobs(PDUFactory::GetPDVs(theData));
                    }
                  else
                    {
                    inCallback->SetImplicitFlag(false);
                    theCompleteFindResponse =
                      PresentationDataValue::ConcatenatePDVBlobsAsExplicit(PDUFactory::GetPDVs(theData));
                    }
                  //note that it's the responsibility of the event to delete the PDU in theFindRSP
                  for (size_t i = 0; i < theData.size(); i++)
                    {
                    delete theData[i];
                    }
                  //outDataSet.push_back(theCompleteFindResponse);
                  if (inCallback)
                    {
                    inCallback->HandleDataSet(theCompleteFindResponse);
                    }
                  //  DataSetEvent dse( &theCompleteFindResponse );
                  //  this->InvokeEvent( dse );


                  if (theCommandCode == 1){//if we're doing cstore scp stuff, send information back along the connection.
                    std::vector<BasePDU*> theCStoreRSPPDU = PDUFactory::CreateCStoreRSPPDU(&theRSP, theFirstPDU);//pass NULL for C-Echo
                    //send them directly back over the connection
                    //ideall, should go through the transition table, but we know this should work
                    //and it won't change the state (unless something breaks?, but then an exception should throw)
                    std::vector<BasePDU*>::iterator itor;
                    for (itor = theCStoreRSPPDU.begin(); itor < theCStoreRSPPDU.end(); itor++){
                      (*itor)->Write(*inWhichConnection->GetProtocol());
                    }

                    inWhichConnection->GetProtocol()->flush();

                    // FIXME added MM / Oct 30 2010
                    //AReleaseRPPDU rel;
                    //rel.Write( *inWhichConnection->GetProtocol() );
                    //inWhichConnection->GetProtocol()->flush();

                    receivingData = false; //gotta get data on the other connection for a cmove

                    // cleanup
                    for (itor = theCStoreRSPPDU.begin(); itor < theCStoreRSPPDU.end(); itor++){
                      delete *itor;
                    }
                  }
                }
              }
            }
            break;
            case eARELEASERequest://process this via the transition table
              waitingForEvent = false;
              break;
            case eARELEASE_RQPDUReceivedOpen://process this via the transition table
              waitingForEvent = false;
              receivingData = true; //to continue the loop to process the release
              break;
            case eAABORTPDUReceivedOpen:
              {
              raisedEvent = eEventDoesNotExist;
              theState = eStaDoesNotExist;
              } // explicitely declare fall-through for some picky compiler
            case eAABORTRequest:
              waitingForEvent = false;
              inWhichConnection->StopProtocol();
              break;
            case eASSOCIATE_ACPDUreceived:
            default:
              waitingForEvent = false;
              break;
          }
        }
      //} else {
      //  raisedEvent = eEventDoesNotExist;
      //  waitingForEvent = false;
      //}
    }
    else {
      currentEvent.SetEvent(raisedEvent);//actions that cause transitions in the state table
      //locally just raise local events that will therefore cause the trigger to be pulled.
    }
  } while (currentEvent.GetEvent() != eEventDoesNotExist &&
    theState != eStaDoesNotExist && theState != eSta13AwaitingClose && theState != eSta1Idle &&
    (theState != eSta6TransferReady || (theState == eSta6TransferReady && receivingData )));
  //stop when the AE is done, or when ready to transfer data (ie, the next PDU should be sent in),
  //or when the connection is idle after a disconnection.
  //or, if in state 6 and receiving data, until all data is received.

  return theState;
}

} // end namespace network
} // end namespace gdcm
