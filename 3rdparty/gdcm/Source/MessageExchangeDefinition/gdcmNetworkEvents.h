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
The NetworkEvents enumeration defines the inputs into the state of the network connection.

These inputs can come either from user input or input from other things on the socket,
ie, responses from the peer or ARTIM timeouts.

Note that this enumeration is not 'power of two', like the states, because you can't have
multiple simultaneous events.  Multiple state outputs in transition tables, however, is possible.

*/
#ifndef GDCMNETWORKEVENTS_H
#define GDCMNETWORKEVENTS_H

namespace gdcm {
  namespace network{
    typedef enum {
      eAASSOCIATERequestLocalUser = 0,
      eTransportConnConfirmLocal,
      eASSOCIATE_ACPDUreceived,
      eASSOCIATE_RJPDUreceived,
      eTransportConnIndicLocal,
      eAASSOCIATE_RQPDUreceived,
      eAASSOCIATEresponseAccept,
      eAASSOCIATEresponseReject,
      ePDATArequest,
      ePDATATFPDU,
      eARELEASERequest,
      eARELEASE_RQPDUReceivedOpen,
      eARELEASE_RPPDUReceived,
      eARELEASEResponse,
      eAABORTRequest,
      eAABORTPDUReceivedOpen,
      eTransportConnectionClosed,
      eARTIMTimerExpired,
      eUnrecognizedPDUReceived,
      eEventDoesNotExist
    } EEventID;

    const int cMaxEventID = eEventDoesNotExist;
  }
}

#endif //NETWORKEVENTS_H
