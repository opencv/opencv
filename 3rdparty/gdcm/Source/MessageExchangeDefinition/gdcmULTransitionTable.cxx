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
 * This file is the implementation of the ULTransitionTable class, including
 * the actual event handling as well as the construction of the table itself.
 */

#include "gdcmULTransitionTable.h"
#include "gdcmULActionAA.h"
#include "gdcmULActionAE.h"
#include "gdcmULActionAR.h"
#include "gdcmULActionDT.h"

namespace gdcm
{
namespace network
{
//construct the table
ULTransitionTable::ULTransitionTable()
{
//row 1
// A-ASSOCIATE Request (local user)
  mTable[eAASSOCIATERequestLocalUser].transitions[GetStateIndex(eSta1Idle)] =
    Transition::MakeNew(eSta4LocalAssocDone, new ULActionAE1());
//row 2
// Transport Conn. Confirmn (local transport service)
  mTable[eTransportConnConfirmLocal].transitions[GetStateIndex(eSta4LocalAssocDone)] =
    Transition::MakeNew(eSta5WaitRemoteAssoc, new ULActionAE2());
//row 3
// A-ASSOCIATE-AC PDU (received on transport connection)
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta2Open)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta6TransferReady, new ULActionAE3());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_ACPDUreceived].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA6());

//row 4
// A-ASSOCIATE-RJ PDU (received on transport connection)
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta2Open)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta1Idle, new ULActionAE4());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eASSOCIATE_RJPDUreceived].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA6());
//row 5
// Transport Connection Indication (local transport service)
  mTable[eTransportConnIndicLocal].transitions[GetStateIndex(eSta1Idle)] =
    Transition::MakeNew(eSta2Open, new ULActionAE5());
//row 6
// A-ASSOCIATE-RQ PDU (received on transport connection)
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta2Open)] =
    Transition::MakeNew(eSta3WaitLocalAssoc | eSta13AwaitingClose, new ULActionAE6());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eAASSOCIATE_RQPDUreceived].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA7());
//row 7
// A-ASSOCIATE response primitive (accept)
  mTable[eAASSOCIATEresponseAccept].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta7WaitRelease, new ULActionAE7());
//row 8
// A-ASSOCIATE response primitive (reject)
  mTable[eAASSOCIATEresponseReject].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta7WaitRelease, new ULActionAE7());
//Row 9
// P-DATA request primitive
  mTable[ePDATArequest].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta6TransferReady, new ULActionDT1());
  mTable[ePDATArequest].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta8WaitLocalRelease, new ULActionAR7());
//row 10
// P-DATA-TF PDU
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta2Open)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta6TransferReady, new ULActionDT2());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta7WaitRelease, new ULActionAR6());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[ePDATATFPDU].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA6());
//row 11
// A-RELEASE Request primitive
  mTable[eARELEASERequest].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta7WaitRelease, new ULActionAR1());
//row 12
// A-RELEASE-RQ PDU (received on open transport connection)
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta2Open)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta8WaitLocalRelease, new ULActionAR2());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta9ReleaseCollisionRqLocal | eSta10ReleaseCollisionAc, new ULActionAR8());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RQPDUReceivedOpen].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA6());
//row 13
// A-RELEASE-RP PDU (received on transport connection)
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta2Open)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta1Idle, new ULActionAR3());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eARELEASE_RPPDUReceived].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA6());
// Row 14
// A-RELEASE Response primitive
  mTable[eARELEASEResponse].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAR4());
  mTable[eARELEASEResponse].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta11ReleaseCollisionRq, new ULActionAR9);
  mTable[eARELEASEResponse].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAR4());
// row 15
// A-ABORT Request primitive
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta4LocalAssocDone)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA2());
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eAABORTRequest].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
// row 16
// A-ABORT PDU (received on open transport connection)
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta2Open)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA2());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA3());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA3());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA3());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA3());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA3());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA3());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA3());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA3());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA3());
  mTable[eAABORTPDUReceivedOpen].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA2());
//row 17
// Transport connection closed indication (local transport service),
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta2Open)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA5());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta4LocalAssocDone)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA4());
  mTable[eTransportConnectionClosed].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA5());
//row 18
// ARTIM timer expired (Association reject/release timer),
  mTable[eARTIMTimerExpired].transitions[GetStateIndex(eSta2Open)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA2());
  mTable[eARTIMTimerExpired].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA2());
//row 19
// Unrecognized or invalid PDU received
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta3WaitLocalAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA1());
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta13AwaitingClose)] =
    Transition::MakeNew(eSta1Idle, new ULActionAA8());
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta5WaitRemoteAssoc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta6TransferReady)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta7WaitRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta8WaitLocalRelease)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta9ReleaseCollisionRqLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta10ReleaseCollisionAc)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta11ReleaseCollisionRq)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA8());
  mTable[eUnrecognizedPDUReceived].transitions[GetStateIndex(eSta12ReleaseCollisionAcLocal)] =
    Transition::MakeNew(eSta13AwaitingClose, new ULActionAA7());
}

//given the event and the state of the connection, call the appropriate action
void ULTransitionTable::HandleEvent(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
                                    bool& outWaitingForEvent, EEventID& outRaisedEvent) const{
  //first, find the Event
  EEventID eventID = inEvent.GetEvent();
  if (eventID >= 0 && eventID < eEventDoesNotExist)
    { //make sure that the event exists
    //have to convert the state ID into an index
    int stateIndex = GetStateIndex(inConnection.GetState());
    if (stateIndex >= 0 && stateIndex < cMaxStateID)
      {
      if ( mTable[eventID].transitions[stateIndex] )
        {
        if (mTable[eventID].transitions[stateIndex]->mAction != NULL)
          {
          gdcmDebugMacro( "Process: Event:" << (int)eventID << ", State:" << stateIndex );
          inConnection.SetState(mTable[eventID].transitions[stateIndex]->mAction->
            PerformAction(s,inEvent, inConnection, outWaitingForEvent, outRaisedEvent));
          }
        }
      else
        {
        gdcmDebugMacro( "Transition failed (NULL) for event:" << (int)eventID << ", State:" << stateIndex );
        }
      }
    }
}

} // end namespace network
} // end namespace gdcm
