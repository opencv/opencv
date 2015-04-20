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

This file contains the implementation for the classes for the AE Actions,
Association Establishment Related Actions (Table 9-6 of ps 3.8-2009).

Since each class is essentially a placeholder for a function pointer, I'm breaking with having
each class have its own file for the sake of brevity of the number of files.

*/

#include "gdcmULActionAE.h"
#include "gdcmARTIMTimer.h"
#include "gdcmAAssociateRQPDU.h"
#include "gdcmAAssociateACPDU.h"
#include "gdcmAAssociateRJPDU.h"

#include <socket++/echo.h>//for setting up the local socket

namespace gdcm
{
namespace network
{

//Issue TRANSPORT CONNECT request primitive to local transport service.
EStateID ULActionAE1::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){

  //opening a local socket
  outWaitingForEvent = false;
  if (!inConnection.InitializeConnection())
    {
    outRaisedEvent = eEventDoesNotExist;
    return eSta1Idle;
    }
  else
    {
    outRaisedEvent = eTransportConnConfirmLocal;
    }
  return eSta4LocalAssocDone;
}

//Send A-ASSOCIATE-RQ-PDU
EStateID ULActionAE2::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
  bool& outWaitingForEvent, EEventID& outRaisedEvent)
{
  AAssociateRQPDU thePDU;

  thePDU.SetCallingAETitle( inConnection.GetConnectionInfo().GetCallingAETitle() );
  thePDU.SetCalledAETitle( inConnection.GetConnectionInfo().GetCalledAETitle() );

  //the presentation context is now defined when the connection is first
  //desired to be established. The connection proposes these different
  //presentation contexts. ideally, we could refine it further to a particular
  //presentation context, but if the server supports many and we support many,
  //then an arbitrary decision can be made.
  std::vector<PresentationContextRQ> const & thePCS =
    inConnection.GetPresentationContexts();

  std::vector<PresentationContextRQ>::const_iterator itor;
  for (itor = thePCS.begin(); itor < thePCS.end(); itor++)
    {
    thePDU.AddPresentationContext(*itor);
    }

  thePDU.Write(*inConnection.GetProtocol());
  inConnection.GetProtocol()->flush();

  outWaitingForEvent = true;
  outRaisedEvent = eEventDoesNotExist;

  return eSta5WaitRemoteAssoc;
}

//Issue A-ASSOCIATE confirmation (accept) primitive
// NOTE: A-ASSOCIATE is NOT A-ASSOCIATE-AC
// PS 3.7 / Annex D for A-ASSOCIATE definition
EStateID ULActionAE3::PerformAction(Subject *, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){


  // Mark please check this junk:
  assert(!inEvent.GetPDUs().empty());
  AAssociateACPDU* acpdu;
    acpdu = dynamic_cast<AAssociateACPDU*>(inEvent.GetPDUs()[0]);
  assert( acpdu );
  uint32_t maxpdu = acpdu->GetUserInformation().GetMaximumLengthSub().GetMaximumLength();
  inConnection.SetMaxPDUSize(maxpdu);

  // once again duplicate AAssociateACPDU vs ULConnection
  for( unsigned int index = 0; index < acpdu->GetNumberOfPresentationContextAC(); index++ ){
    PresentationContextAC const &pc = acpdu->GetPresentationContextAC(index);
    inConnection.AddAcceptedPresentationContext(pc);
  }

  outWaitingForEvent = false;
  outRaisedEvent = eEventDoesNotExist;//no event is raised,
  //wait for the user to try to send some data.
  return eSta6TransferReady;
}

//Issue A-ASSOCIATE confirmation (reject) primitive and close transport connection
EStateID ULActionAE4::PerformAction(Subject *, ULEvent& , ULConnection& ,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){

  outWaitingForEvent = false;
  outRaisedEvent = eASSOCIATE_RJPDUreceived;
  return eSta1Idle;
}

//Issue Transport connection response primitive, start ARTIM timer
EStateID ULActionAE5::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){

  //issue response primitive; have to set that up
  inConnection.GetTimer().Start();

  outWaitingForEvent = false;
  outRaisedEvent = eTransportConnConfirmLocal;
  return eSta2Open;
}

//Stop ARTIM timer and if A-ASSOCIATE-RQ acceptable by service-provider:
//- issue A-ASSOCIATE indication primitive
//Next state: eSta3WaitLocalAssoc
//otherwise:
//- issue A-ASSOCIATE-RJ-PDU and start ARTIM timer
//Next state: eSta13AwaitingClose
EStateID ULActionAE6::PerformAction(Subject *, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){

 // we are in a C-MOVE

  inConnection.GetTimer().Stop();

  //have to determine 'acceptability'
  //this is more server side than client, so it's a bit empty now
  //we have one server type, a store scp started on a cmove
  //so, it's defined as acceptable.
  bool acceptable = true;//for now, always accept
  if (inEvent.GetPDUs().empty()){
    acceptable = false; //can't accept an empty set of pdus.
    //also, requrie little endian, not sure how to set that, but it should be here.
  }
  AAssociateRQPDU* rqpdu;
  if (acceptable){
    rqpdu = dynamic_cast<AAssociateRQPDU*>(inEvent.GetPDUs()[0]);
    if (rqpdu == NULL){
      acceptable = false;
    }
  }
  if (acceptable){
    outWaitingForEvent = false;//not waiting, now want to get the
    //sending of data underway.  Have to get info now
    outRaisedEvent = eAASSOCIATEresponseAccept;

    TransferSyntaxSub ts1;
    ts1.SetNameFromUID( UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM );

    AAssociateACPDU acpdu;

    assert( rqpdu->GetNumberOfPresentationContext() );
    for( unsigned int index = 0; index < rqpdu->GetNumberOfPresentationContext(); index++ )
      {
      // FIXME / HARDCODED We only ever accept Little Endian
      // FIXME we should check :
      // rqpdu.GetAbstractSyntax() contains LittleEndian
      PresentationContextAC pcac1;
      PresentationContextRQ const &pc = rqpdu->GetPresentationContext(index);
      //add the presentation context back into the connection,
      //so later functions will know what's allowed on this connection
      // BOGUS (MM):
      //inConnection.AddAcceptedPresentationContext(pc);

      const uint8_t id = pc.GetPresentationContextID();

      std::vector<TransferSyntaxSub> const & tsSet = pc.GetTransferSyntaxes();
      std::vector<TransferSyntaxSub>::const_iterator tsitor;
      // PS 3.8 Table 9-18 PRESENTATION CONTEXT ITEM FIELDS
      uint8_t result = 4; // transfer-syntaxes-not-supported (provider rejection)
      for (tsitor = tsSet.begin(); tsitor < tsSet.end(); tsitor++)
        {
        //gdcmDebugMacro( "Checking: [" << tsitor->GetName() << "] vs [" << ts1.GetName() << "]" << std::endl );
        if (strcmp(tsitor->GetName(), ts1.GetName()) == 0 )
          {
          result = 0; // 0 - acceptance
          inConnection.SetCStoreTransferSyntax( ts1 );
          pcac1.SetTransferSyntax( ts1 );
          }
        }
      if( result )
        {
        gdcmWarningMacro( "Could not find Implicit or Explicit Little Endian in Response. Giving another try" );
        // Okay little endian implicit was not found, this happen sometimes, for eg with DVTk, let's be nice and accept also Explicit
        TransferSyntaxSub ts2;
        ts2.SetNameFromUID( UIDs::ExplicitVRLittleEndian );
        for (tsitor = tsSet.begin(); tsitor < tsSet.end(); tsitor++)
          {
          //gdcmDebugMacro( "Checking: [" << tsitor->GetName() << "] vs [" << ts1.GetName() << "]" << std::endl );
          if (strcmp(tsitor->GetName(), ts2.GetName()) == 0 )
            {
            result = 0; // 0 - acceptance
            inConnection.SetCStoreTransferSyntax( ts2 );
            pcac1.SetTransferSyntax( ts2 );
            }
          }
        }
      if( result )
        {
        gdcmErrorMacro( "Could not find Implicit or Explicit Little Endian in Response. Giving up" );
        }
      pcac1.SetPresentationContextID( id );
      pcac1.SetReason( result );
      acpdu.AddPresentationContextAC( pcac1 );
    }
    assert( acpdu.GetNumberOfPresentationContextAC() );

    // Init AE-Titles:
    acpdu.InitFromRQ( *rqpdu );

    acpdu.Write( *inConnection.GetProtocol() );
    inConnection.GetProtocol()->flush();

    return eSta3WaitLocalAssoc;
  } else {

    outWaitingForEvent = false;
    outRaisedEvent = eAASSOCIATEresponseReject;
    AAssociateRJPDU thePDU;
    thePDU.Write(*inConnection.GetProtocol());
    inConnection.GetProtocol()->flush();
    inConnection.GetTimer().Stop();
    return eSta13AwaitingClose;
  }

}

//Send A-ASSOCIATE-AC PDU
EStateID ULActionAE7::PerformAction(Subject *, ULEvent& , ULConnection& ,
        bool& outWaitingForEvent, EEventID& outRaisedEvent)
{
  outWaitingForEvent = true;
  outRaisedEvent = eEventDoesNotExist;
  return eSta6TransferReady;
}

//Send A-ASSOCIATE-RJ PDU and start ARTIM timer
EStateID ULActionAE8::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent)
{
  AAssociateRJPDU thePDU;
  thePDU.Write(*inConnection.GetProtocol());
  inConnection.GetTimer().Start();
  outWaitingForEvent = false;
  outRaisedEvent = eAASSOCIATEresponseReject;

  return eSta13AwaitingClose;
}

} // end namespace network
} // end namespace gdcm
