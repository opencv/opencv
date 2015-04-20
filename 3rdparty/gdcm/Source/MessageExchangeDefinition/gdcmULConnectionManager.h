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
#ifndef GDCMULCONNECTIONMANAGER_H
#define GDCMULCONNECTIONMANAGER_H

#include "gdcmULTransitionTable.h"
#include "gdcmULConnection.h"
#include "gdcmULConnectionInfo.h"
#include "gdcmPresentationDataValue.h"
#include "gdcmULConnectionCallback.h"
#include "gdcmSubject.h"
#include "gdcmPresentationContext.h"

namespace gdcm {
  class File;
  class BaseRootQuery;

  namespace network {

/**
 * \brief ULConnectionManager
 * The ULConnectionManager performs actions on the ULConnection given inputs
 * from the user and from the state of what's going on around the connection
 * (ie, timeouts of the ARTIM timer, responses from the peer across the
 * connection, etc).
 *
 * Its inputs are ULEvents, and it performs ULActions.
 */
  class GDCM_EXPORT ULConnectionManager : public Subject
  {
    private:
      ULConnection* mConnection;
      ULConnection* mSecondaryConnection;
      ULTransitionTable mTransitions;

      //no copying
      ULConnectionManager(const ULConnectionManager& inCM);

      //event handler loop.
      //will just keep running until the current event is nonexistent.
      //at which point, it will return the current state of the connection
      //this starts by initiating an action, but can be put into a passive mode
      //for a cmove/cstore combination by setting startWaiting to true
      EStateID RunEventLoop(ULEvent& inEvent, ULConnection* inWhichConnection, 
        ULConnectionCallback* inCallback, const bool& startWaiting);

      //like the above, but will manage the event loop for a move event (which
      //is basically two simultaneous connections interwoven, one inbound and
      //the other outbound.  Note, for instance, that cmoversp's can be sent back
      //during the other connection's operation.
      EStateID RunMoveEventLoop(ULEvent& inEvent, ULConnectionCallback* inCallback);

    public:
      ULConnectionManager();
      ~ULConnectionManager();

      // NOTE: (MM) The following two functions are difficults to use, therefore marking
      // them as internal for now.

      // \internal
      /// returns true if a connection of the given AETitle (ie, 'this' program)
      /// is able to connect to the given AETitle and Port in a certain amount of
      /// time providing the connection type will establish the proper exchange
      /// syntax with a server; if a different functionality is required, a
      /// different connection should be established.
      /// returns false if the connection type is 'move'-- have to give a return
      /// port for move to work as specified.
      bool EstablishConnection(const std::string& inAETitle,
        const std::string& inConnectAETitle,
        const std::string& inComputerName, long inIPAddress,
        uint16_t inConnectPort, double inTimeout,
        std::vector<PresentationContext> const & pcVector );

      /// returns true for above reasons, but contains the special 'move' port
      /// \internal
      bool EstablishConnectionMove(const std::string& inAETitle,
        const std::string& inConnectAETitle,
        const std::string& inComputerName, long inIPAddress,
        uint16_t inConnectPort, double inTimeout,
        uint16_t inReturnPort,
        std::vector<PresentationContext> const & pcVector);
      // \endinternal


      //bool ReestablishConnection(const EConnectionType& inConnectionType,
      //  const DataSet& inDS);

      //allows for a connection to be broken, but waits for an acknowledgement
      //of the breaking for a certain amount of time.  Returns true of the
      //other side acknowledges the break
      bool BreakConnection(const double& inTimeout);

      //severs the connection, if it's open, without waiting for any kind of response.
      //typically done if the program is going down.
      void BreakConnectionNow();

      //This function will send a given piece of data
      //across the network connection.  It will return true if the
      //sending worked, false otherwise.
      //note that sending is asynchronous; as such, there's
      //also a 'receive' option, but that requires a callback function.
      //bool SendData();

      //send the Data PDU associated with Echo (ie, a default DataPDU)
      //this lets the user confirm that the connection is alive.
      //the user should look to cout to see the response of the echo command
      //returns the PresentationDataValue that was returned by the remote
      //host.  Note that the PDV can be uninitialized, which would indicate failure.
      //Echo does not use a callback for results.
      std::vector<PresentationDataValue> SendEcho();

      // \internal
      // API will change...
      std::vector<DataSet> SendStore(const File &file);
      std::vector<DataSet> SendFind(const BaseRootQuery* inRootQuery);
      std::vector<DataSet> SendMove(const BaseRootQuery* inRootQuery);
      // \endinternal

      ///callback based API
      void SendStore(const File & file, ULConnectionCallback* inCallback);
      void SendFind(const BaseRootQuery* inRootQuery, ULConnectionCallback* inCallback);
      /// return false upon error
      bool SendMove(const BaseRootQuery* inRootQuery, ULConnectionCallback* inCallback);

    };
  }
}

#endif // GDCMULCONNECTIONMANAGER_H
