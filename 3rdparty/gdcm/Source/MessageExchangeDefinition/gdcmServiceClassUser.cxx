/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmServiceClassUser.h"

#include "gdcmULTransitionTable.h"
#include "gdcmULConnection.h"
#include "gdcmULConnectionInfo.h"
#include "gdcmPresentationContextAC.h"
#include "gdcmPresentationDataValue.h"
#include "gdcmULConnectionCallback.h"
#include "gdcmULBasicCallback.h"
#include "gdcmPDUFactory.h"
#include "gdcmAttribute.h"
#include "gdcmULWritingCallback.h"

#include "gdcmPrinter.h" // FIXME
#include "gdcmReader.h" // FIXME

namespace gdcm
{
static const char GDCM_AETITLE[] = "GDCMSCU";

using namespace network;
class ServiceClassUserInternals
{
public:
  ULConnection* mConnection;
  ULConnection* mSecondaryConnection;
  ULTransitionTable mTransitions;

  std::string hostname;
  int port;
  int portscp;
  std::string aetitle;
  std::string calledaetitle;
  double timeout;

  ServiceClassUserInternals():mConnection(NULL),mSecondaryConnection(NULL){}
  ~ServiceClassUserInternals(){
    delete mConnection;
    delete mSecondaryConnection;
  }
};

ServiceClassUser::ServiceClassUser()
{
  Internals = new ServiceClassUserInternals;
  Internals->hostname = "localhost";
  Internals->port = 104;
  Internals->portscp = 104;
  Internals->aetitle = GDCM_AETITLE;
  Internals->calledaetitle = "ANY-SCP";
  Internals->timeout = 10;
}

ServiceClassUser::~ServiceClassUser()
{
  delete Internals;
}

void ServiceClassUser::SetPresentationContexts(std::vector<PresentationContext> const & pcs)
{
  if( Internals->mConnection )
    {
    Internals->mConnection->SetPresentationContexts(pcs);
    }
}

bool ServiceClassUser::IsPresentationContextAccepted(const PresentationContext& pc) const
{
  bool found = false;
  const std::vector<PresentationContextAC> &acceptedContexts =
    Internals->mConnection->GetAcceptedPresentationContexts();
  std::vector<PresentationContextAC>::const_iterator itor;
  uint8_t contextID = pc.GetPresentationContextID();

  for (itor = acceptedContexts.begin();
    itor != acceptedContexts.end() && !found;
    itor++)
    {
    if (contextID == itor->GetPresentationContextID())
      found = true;
    }

  return found;
}

bool ServiceClassUser::InitializeConnection()
{
  UserInformation userInfo;
  ULConnectionInfo connectInfo;
  if (Internals->aetitle.size() > 16)
    {
    return false;
    }
  if (Internals->calledaetitle.size() > 16)
    {
    return false;
    }
  if (!Internals->port)
    {
    return false;
    }
  if (Internals->hostname.empty())
    {
    return false;
    }
  if (!connectInfo.Initialize(userInfo, Internals->calledaetitle.c_str(),
      Internals->aetitle.c_str(), 0, Internals->port, Internals->hostname))
    {
    return false;
    }

  ULConnection* mConnection = Internals->mConnection;
  if (mConnection)
    {
    delete mConnection;
    }
  Internals->mConnection = new ULConnection(connectInfo);
  Internals->mConnection->GetTimer().SetTimeout(Internals->timeout);

  return true;
}

bool ServiceClassUser::StartAssociation()
{
  // need to make sure no cached PresContAC are around:
  Internals->mConnection->GetAcceptedPresentationContexts().clear();
  if( Internals->mConnection->GetPresentationContexts().empty() )
    {
    return false;
    }

  ULEvent theEvent(eAASSOCIATERequestLocalUser, NULL);
  network::EStateID theState = RunEventLoop(theEvent, Internals->mConnection, NULL, false);
  if(theState != eSta6TransferReady)
    {
    std::vector<BasePDU*> const & thePDUs = theEvent.GetPDUs();
    for( std::vector<BasePDU*>::const_iterator itor
      = thePDUs.begin(); itor != thePDUs.end(); itor++)
      {
      assert(*itor);
      if (*itor == NULL) continue; //can have a nulled pdu, apparently
      (*itor)->Print(Trace::GetErrorStream());
      }
    }

  return theState == eSta6TransferReady;
}

bool ServiceClassUser::StopAssociation()
{
  ULConnection* mConnection = Internals->mConnection;

  BasePDU* thePDU = PDUFactory::ConstructReleasePDU();
  ULEvent theEvent(eARELEASERequest, thePDU);
  EStateID theState = RunEventLoop(theEvent, mConnection, NULL, false);

  return theState == eSta1Idle;
}

void ServiceClassUser::SetTimeout(double t)
{
  Internals->timeout = t;
}

double ServiceClassUser::GetTimeout() const
{
  return Internals->timeout;
}

void ServiceClassUser::SetCalledAETitle(const char *aetitle)
{
  if( aetitle )
    Internals->calledaetitle = aetitle;
}

void ServiceClassUser::SetAETitle(const char *aetitle)
{
  if( aetitle )
    Internals->aetitle = aetitle;
}

const char *ServiceClassUser::GetCalledAETitle() const
{
  return Internals->calledaetitle.c_str();
}

const char *ServiceClassUser::GetAETitle() const
{
  return Internals->aetitle.c_str();
}

void ServiceClassUser::SetHostname( const char *hostname )
{
  if( hostname )
    Internals->hostname = hostname;
}

void ServiceClassUser::SetPort( uint16_t port )
{
  Internals->port = port;
}

void ServiceClassUser::SetPortSCP( uint16_t portscp )
{
  Internals->portscp = portscp;
}

bool ServiceClassUser::SendEcho()
{
  ULConnection* mConnection = Internals->mConnection;
  std::vector<BasePDU*> theDataPDU = PDUFactory::CreateCEchoPDU(*mConnection);
  ULEvent theEvent(ePDATArequest, theDataPDU);

  EStateID theState = RunEventLoop(theEvent, mConnection, NULL, false);

  return theState == eSta6TransferReady;
}

bool ServiceClassUser::SendStore(const char *filename)
{
  if( !filename ) return false;
  Reader reader;
  reader.SetFileName( filename );
  bool b = reader.Read();
  if( !b )
    {
    gdcmDebugMacro( "Could not read: " << filename );
    return false;
    }
  const File & file = reader.GetFile();
  return SendStore( file );
}

bool ServiceClassUser::SendStore(DataSet const &ds)
{
  SmartPointer<File> file = new File;
  file->SetDataSet( ds );
  file->GetHeader().SetDataSetTransferSyntax( TransferSyntax::ImplicitVRLittleEndian );
  file->GetHeader().FillFromDataSet( ds );
  return SendStore(*file);
}

bool ServiceClassUser::SendStore(File const &file)
{
  ULConnection* mConnection = Internals->mConnection;

  std::vector<BasePDU*> theDataPDU;
  try
    {
    theDataPDU = PDUFactory::CreateCStoreRQPDU(*mConnection, file);
    }
  catch ( std::exception &ex )
    {
    (void)ex;  //to avoid unreferenced variable warning on release
    gdcmErrorMacro( "Could not C-STORE: " << ex.what() );
    return false;
    }

  network::ULBasicCallback theCallback;
  network::ULConnectionCallback* inCallback = &theCallback;

  ULEvent theEvent(ePDATArequest, theDataPDU);
  EStateID stateid = RunEventLoop(theEvent, mConnection, inCallback, false);
  assert( stateid == eSta6TransferReady ); (void)stateid;
  std::vector<DataSet> const &theDataSets = theCallback.GetResponses();

  bool ret = true;
  assert( theDataSets.size() == 1 );
  const DataSet &ds = theDataSets[0];
  assert ( ds.FindDataElement(Tag(0x0, 0x0900)) );
  DataElement const & de = ds.GetDataElement(Tag(0x0,0x0900));
  Attribute<0x0,0x0900> at;
  at.SetFromDataElement( de );
  // PS 3.4 - 2011
  // Table W.4-1 C-STORE RESPONSE STATUS VALUES
  const uint16_t theVal = at.GetValue();
  switch( theVal )
    {
  case 0x0:
    gdcmDebugMacro( "C-Store of file was successful." );
    break;
  case 0xA700:
  case 0xA900:
  case 0xC000:
      {
      // TODO: value from 0901 ?
      gdcmErrorMacro( "C-Store of file was a failure." );
      Attribute<0x0,0x0902> errormsg;
      errormsg.SetFromDataSet( ds );
      const char *themsg = errormsg.GetValue();
      assert( themsg ); (void)themsg;
      gdcmErrorMacro( "Response Status: " << themsg );
      ret = false; // at least one file was not sent correctly
      }
    break;
  default:
    gdcmErrorMacro( "Unhandle error code: " << theVal );
    gdcmAssertAlwaysMacro( 0 );
    }

  return ret;
}

bool ServiceClassUser::SendFind(const BaseRootQuery* query, std::vector<DataSet> &retDataSets)
{
  ULConnection* mConnection = Internals->mConnection;
  network::ULBasicCallback theCallback;
  network::ULConnectionCallback* inCallback = &theCallback;

  std::vector<BasePDU*> theDataPDU = PDUFactory::CreateCFindPDU( *mConnection, query);
  ULEvent theEvent(ePDATArequest, theDataPDU);
  RunEventLoop(theEvent, mConnection, inCallback, false);

  std::vector<DataSet> const & theDataSets = theCallback.GetDataSets();
  std::vector<DataSet> const & theResponses = theCallback.GetResponses();

  bool ret = false; // by default an error
  assert( theResponses.size() >= 1 );
  // take the last one:
  const DataSet &ds = theResponses[ theResponses.size() - 1 ]; // FIXME
  assert ( ds.FindDataElement(Tag(0x0, 0x0900)) );
  Attribute<0x0,0x0900> at;
  at.SetFromDataSet( ds );

  //          Table CC.2.8-2
  //C-FIND RESPONSE STATUS VALUES
  const uint16_t theVal = at.GetValue();
  switch( theVal )
    {
  case 0x0: // Matching is complete - No final Identifier is supplied.
    gdcmDebugMacro( "C-Find was successful." );
    // Append the new DataSet to the ret one:
    retDataSets.insert( retDataSets.end(), theDataSets.begin(), theDataSets.end() );
    ret = true;
    break;
  case 0xA900: // Identifier Does Not Match SOP Class
      {
      Attribute<0x0,0x0901> errormsg;
      if( ds.FindDataElement( errormsg.GetTag() ) )
        {
        errormsg.SetFromDataSet( ds );
        gdcm::Tag const & t = errormsg.GetValue();
        gdcmErrorMacro( "Offending Element: " << t ); (void)t;
        }
      else
        {
        gdcmErrorMacro( "Offending Element ??" );
        }
      }
    break;
  case 0xA700: // Refused: Out of Resources
      {
      Attribute<0x0,0x0902> errormsg;
      errormsg.SetFromDataSet( ds );
      const char *themsg = errormsg.GetValue();
      assert( themsg ); (void)themsg;
      gdcmErrorMacro( "Response Status: [" << themsg << "]" );
      }
    break;
  case 0x0122: // SOP Class not Supported
    gdcmErrorMacro( "SOP Class not Supported" );
    break;
  case 0xfe00: // Matching terminated due to Cancel request
    gdcmErrorMacro( "Matching terminated due to Cancel request" );
    break;
  default:
      {
      if( theVal >= 0xC000 && theVal <= 0xCFFF ) // Unable to process
        {
        Attribute<0x0,0x0902> errormsg;
        errormsg.SetFromDataSet( ds );
        const char *themsg = errormsg.GetValue();
        assert( themsg ); (void)themsg;
        gdcmErrorMacro( "Response Status: " << themsg );
        }
      }
    }

  return ret;
}

bool ServiceClassUser::SendMove(const BaseRootQuery* query, const char *outputdir)
{
  UserInformation userInfo2;
  ULConnectionInfo connectInfo2;
  if (!connectInfo2.Initialize(userInfo2, Internals->aetitle.c_str(),
      Internals->calledaetitle.c_str(), 0, Internals->portscp, Internals->hostname))
    {
    return false;
    }

  // let's start the secondary connection
  ULConnection* mSecondaryConnection = Internals->mSecondaryConnection;
  if (mSecondaryConnection)
    {
    delete mSecondaryConnection;
    }
  Internals->mSecondaryConnection = new ULConnection(connectInfo2);
  Internals->mSecondaryConnection->GetTimer().SetTimeout(Internals->timeout);

  ULConnection* mConnection = Internals->mConnection;
  network::ULWritingCallback theCallback;
  theCallback.SetDirectory(outputdir);
  network::ULConnectionCallback* inCallback = &theCallback;

  std::vector<BasePDU*> theDataPDU = PDUFactory::CreateCMovePDU( *mConnection, query );
  ULEvent theEvent(ePDATArequest, theDataPDU);
  EStateID stateid = RunMoveEventLoop(theEvent, inCallback);
  if( stateid != gdcm::network::eSta6TransferReady )
    {
    return false;
    }

  return true;
}

bool ServiceClassUser::SendMove(const BaseRootQuery* query, std::vector<DataSet> &retDataSets)
{
  UserInformation userInfo2;
  ULConnectionInfo connectInfo2;
  if (!connectInfo2.Initialize(userInfo2, Internals->aetitle.c_str(),
      Internals->calledaetitle.c_str(), 0, Internals->portscp, Internals->hostname))
    {
    return false;
    }

  // let's start the secondary connection
  ULConnection* mSecondaryConnection = Internals->mSecondaryConnection;
  if (mSecondaryConnection)
    {
    delete mSecondaryConnection;
    }
  Internals->mSecondaryConnection = new ULConnection(connectInfo2);
  Internals->mSecondaryConnection->GetTimer().SetTimeout(Internals->timeout);

  ULConnection* mConnection = Internals->mConnection;
  network::ULBasicCallback theCallback;
  network::ULConnectionCallback* inCallback = &theCallback;

  std::vector<BasePDU*> theDataPDU = PDUFactory::CreateCMovePDU( *mConnection, query );
  ULEvent theEvent(ePDATArequest, theDataPDU);
  EStateID stateid = RunMoveEventLoop(theEvent, inCallback);
  if( stateid != gdcm::network::eSta6TransferReady )
    {
    return false;
    }

  std::vector<DataSet> const & theDataSets = theCallback.GetDataSets();
  retDataSets.insert( retDataSets.end(), theDataSets.begin(), theDataSets.end() );

  return true;
}

bool ServiceClassUser::SendMove(const BaseRootQuery* query, std::vector<File> &retFiles)
{
  (void)query;
  (void)retFiles;
  assert( 0 && "unimplemented do not use" );
  return false;
}

//event handler loop.
//will just keep running until the current event is nonexistent.
//at which point, it will return the current state of the connection
//to do this, execute an event, and then see if there's a response on the
//incoming connection (with a reasonable amount of timeout).
//if no response, assume that the connection is broken.
//if there's a response, then yay.
//note that this is the ARTIM timeout event
EStateID ServiceClassUser::RunEventLoop(network::ULEvent& currentEvent,
  network::ULConnection* inWhichConnection,
  network::ULConnectionCallback* inCallback,
  const bool& startWaiting = false)
{
  EStateID theState = eStaDoesNotExist;
  bool waitingForEvent = startWaiting;//overwritten if not starting waiting, but if waiting, then wait
  EEventID raisedEvent;

  bool receivingData = false;
  //bool justWaiting = startWaiting;
  //not sure justwaiting is useful; for now, go back to waiting for event

  //when receiving data from a find, etc, then justWaiting is true and only receiving is done
  //eventually, could add cancel into the mix... but that would be through a callback or something similar
  do {
    raisedEvent = eEventDoesNotExist;
    if (!waitingForEvent){//justWaiting){
      Internals->mTransitions.HandleEvent(this, currentEvent, *inWhichConnection, waitingForEvent, raisedEvent);
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
          is.read( (char*)&itemtype, 1 );
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
            waitingForEvent = false; //because no PDU means not waiting anymore
          }
        }
        catch (...)
          {
          //handle the exception, which is basically that nothing came in over the pipe.
          assert( 0 );
          }
      }
      //now, we have to figure out the event that just happened based on the PDU that was received.
      //this state gathering is for scps, especially the cstore for cmove.
      theState = inWhichConnection->GetState();
      if (!incomingPDUs.empty()){
        currentEvent.SetEvent(PDUFactory::DetermineEventByPDU(incomingPDUs[0]));
        currentEvent.SetPDU(incomingPDUs);
        //here's the scp handling code
        if (Internals->mConnection->GetTimer().GetHasExpired()){
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
                    // FIXME: How do I find out how many PData we are receiving ?
                    // This is needed for proper progress report
                    thePDU->Read(is);
                    theData.push_back(thePDU);
                  } else{
                    break;
                  }
                  //!!!need to handle incoming PDUs that are not data, ie, an abort
                } while(!thePDU->IsLastFragment());
                if (!interrupted){//ie, if the remote server didn't hang up
                  DataSet theCompleteFindResponse =
                    PresentationDataValue::ConcatenatePDVBlobs(PDUFactory::GetPDVs(theData));
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

EStateID ServiceClassUser::RunMoveEventLoop(ULEvent& currentEvent, ULConnectionCallback* inCallback){
  EStateID theState = eStaDoesNotExist;
  bool waitingForEvent;
  EEventID raisedEvent;

  bool receivingData = false;
  bool justWaiting = false;
  //when receiving data from a find, etc, then justWaiting is true and only receiving is done
  //eventually, could add cancel into the mix... but that would be through a callback or something similar
  do {
    if (!justWaiting){
      Internals->mTransitions.HandleEvent(this,currentEvent, *Internals->mConnection, waitingForEvent, raisedEvent);
    }

    theState = Internals->mConnection->GetState();
    std::istream &is = *Internals->mConnection->GetProtocol();
    //std::ostream &os = *mConnection->GetProtocol();


    //When doing a C-MOVE we receive the Requested DataSet over
    //another channel (technically this is send to an SCP)
    //in our case we use another port to receive it.
    EStateID theCStoreStateID = eSta6TransferReady;
    bool secondConnectionEstablished = false;
    if (Internals->mSecondaryConnection->GetProtocol() == NULL){
      //establish the connection
      //can fail if is_readready doesn't return true, ie, the connection
      //wasn't opened on the other side because the other side isn't sending data yet
      //for whatever reason (maybe there's nothing to get?)
      try
        {
        secondConnectionEstablished =
          Internals->mSecondaryConnection->InitializeIncomingConnection();
        }
      catch ( std::exception & e )
        {
        gdcmErrorMacro( "Error 2nd connection:" << e.what() );
        }
      catch ( ... )
        {
        assert( 0 );
        }
    }
    if (secondConnectionEstablished &&
      (Internals->mSecondaryConnection->GetState()== eSta1Idle ||
      Internals->mSecondaryConnection->GetState() == eSta2Open)){
      ULEvent theCStoreEvent(eEventDoesNotExist, NULL);//have to fill this in, we're in passive mode now
      theCStoreStateID = RunEventLoop(theCStoreEvent, Internals->mSecondaryConnection, inCallback, true);
    }

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
        if (Internals->mConnection->GetTimer().GetHasExpired())
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
          if (Trace::GetDebugFlag()){
            Printer thePrinter;
            thePrinter.PrintDataSet(theRSP, Trace::GetStream());
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
          uint32_t theCommandCode = 0;
          if (theRSP.FindDataElement(Tag(0x0,0x0100))){
            DataElement de = theRSP.GetDataElement(Tag(0x0,0x0100));
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
              if (Internals->mSecondaryConnection->GetProtocol() == NULL){
                //establish the connection
                //can fail if is_readready doesn't return true, ie, the connection
                //wasn't opened on the other side because the other side isn't sending data yet
                //for whatever reason (maybe there's nothing to get?)
                secondConnectionEstablished =
                  Internals->mSecondaryConnection->InitializeIncomingConnection();
                if (secondConnectionEstablished &&
                  (Internals->mSecondaryConnection->GetState()== eSta1Idle ||
                  Internals->mSecondaryConnection->GetState() == eSta2Open)){
                  ULEvent theCStoreEvent(eEventDoesNotExist, NULL);//have to fill this in, we're in passive mode now
                  theCStoreStateID = RunEventLoop(theCStoreEvent, Internals->mSecondaryConnection, inCallback, true);
                } else {//something broke, can't establish secondary move connection here
                  gdcmErrorMacro( "Unable to establish secondary connection with server, aborting." << std::endl);
                  return eStaDoesNotExist;
                }
              }
              if (secondConnectionEstablished){
                while (theCStoreStateID == eSta6TransferReady && dataSetCountIncremented){
                  ULEvent theCStoreEvent(eEventDoesNotExist, NULL);//have to fill this in, we're in passive mode now
                  //now, get data from across the network
                  theCStoreStateID = RunEventLoop(theCStoreEvent, Internals->mSecondaryConnection, inCallback, true);
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


} // end namespace gdcm
