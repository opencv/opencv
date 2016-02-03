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
This file contains the implementation for the classes for the AR Actions,
Association Release Related Actions (Table 9-8 of ps 3.8-2009).

Since each class is essentially a placeholder for a function pointer, I'm breaking with having
each class have its own file for the sake of brevity of the number of files.
*/

#include "gdcmULActionAR.h"
#include "gdcmARTIMTimer.h"
#include "gdcmAReleaseRQPDU.h"
#include "gdcmAReleaseRPPDU.h"
#include "gdcmPDataTFPDU.h"

namespace gdcm
{
namespace network
{

//Send A-RELEASE-RQ-PDU
EStateID ULActionAR1::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& ){

  AReleaseRQPDU thePDU;//for now, use Matheiu's default values
  thePDU.Write(*inConnection.GetProtocol());
  inConnection.GetProtocol()->flush(); // important


  outWaitingForEvent = true;

  return eSta7WaitRelease;
}

//Issue A-RELEASE indication primitive
EStateID ULActionAR2::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){

  outWaitingForEvent = false;
  outRaisedEvent = eARELEASERequest;//here's the primitive being sent

  //this is very stupid.
  //the A-release indication primitive is given to determine whether or not the
  //server wants to send an a-release, an a-abort, or a pdata pdu.
  //we just want to send an a-release.
  //therfore, this function will directly send the release.
  //ar8 does the same thing, but so far, we have not had that collision yet.

  AReleaseRPPDU thePDU;//for now, use Matheiu's default values
  thePDU.Write(*inConnection.GetProtocol());
  inConnection.GetProtocol()->flush();

  //if we hadn't actually just performed the primitive right here, we sould be in sta8
  //as it is, we should be in sta13
//  return eSta8WaitLocalRelease;
  return eSta13AwaitingClose;
}

//Issue A-RELEASE confirmation primitive, and close transport connection
EStateID ULActionAR3::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){

  outWaitingForEvent = false;
  outRaisedEvent = eARELEASERequest;
  inConnection.StopProtocol();
  return eSta1Idle;
}

//Issue A-RELEASE-RP PDU and start ARTIM timer
EStateID ULActionAR4::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){

  AReleaseRPPDU thePDU;//for now, use Matheiu's default values
  thePDU.Write(*inConnection.GetProtocol());
  inConnection.GetProtocol()->flush();
  inConnection.GetTimer().Start();
  outWaitingForEvent = false;
  outRaisedEvent = eARELEASERequest;

  return eSta13AwaitingClose;
}

//Stop ARTIM timer
EStateID ULActionAR5::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& , EEventID& ){

  inConnection.GetTimer().Stop();
  return eSta1Idle;
}

//Issue P-DATA indication
EStateID ULActionAR6::PerformAction(Subject *, ULEvent& , ULConnection& ,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){

  outWaitingForEvent = true;
  outRaisedEvent = eEventDoesNotExist;
  return eSta7WaitRelease;
}

//Issue P-DATA-TF PDU
EStateID ULActionAR7::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& , EEventID& ){

assert(0);
  PDataTFPDU thePDU;//for now, use Matheiu's default values
  thePDU.Write(*inConnection.GetProtocol());
  inConnection.GetProtocol()->flush();
  return eSta8WaitLocalRelease;
}

//Issue A-RELEASE indication (release collision):
//- If association-requestor, next state is eSta9ReleaseCollisionRqLocal
//- if not, next state is eSta10ReleaseCollisionAc
EStateID ULActionAR8::PerformAction(Subject *, ULEvent& , ULConnection& ,
        bool& , EEventID& ){

assert(0);
  return eSta10ReleaseCollisionAc;
}

//Send A-RELEASE-RP PDU
EStateID ULActionAR9::PerformAction(Subject *, ULEvent& , ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& ){

  AReleaseRPPDU thePDU;//for now, use Matheiu's default values
  thePDU.Write(*inConnection.GetProtocol());
  inConnection.GetProtocol()->flush();

  outWaitingForEvent = true;
  return eSta11ReleaseCollisionRq;
}

//Issue A-RELEASE confirmation primitive
EStateID ULActionAR10::PerformAction(Subject *, ULEvent& , ULConnection& ,
        bool& outWaitingForEvent, EEventID& outRaisedEvent){

  outWaitingForEvent = false;
  outRaisedEvent = eARELEASEResponse;

  return eSta12ReleaseCollisionAcLocal;
}

} // end namespace network
} // end namespace gdcm
