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
#ifndef GDCMULACTION_H
#define GDCMULACTION_H

#include "gdcmNetworkStateID.h"
#include "gdcmULEvent.h"
#include "gdcmULConnection.h"

namespace gdcm {
class Subject;
  namespace network {

/**
 * \brief ULAction
 * A ULConnection in a given ULState can perform certain ULActions.  This base class
 * provides the interface for running those ULActions on a given ULConnection.
 *
 * Essentially, the ULConnectionManager will take this object, determined from the current
 * ULState of the ULConnection, and pass the ULConnection object to the ULAction.  The ULAction
 * will then invoke whatever necessary commands are required by a given action.
 *
 * The result of a ULAction is a ULEvent (ie, what happened as a result of the action).
 *
 * This ULEvent is passed to the ULState, so that the transition to the next state can occur.
 *
 * Actions are associated with Payloads-- be thos filestreams, AETitles to establish connections,
 * whatever.  The actual parameters that the user will pass via an action will come through
 * a Payload object, which should, in itself, be some gdcm-based object (but not all objects can
 * be payloads; sending a single dataelement as a payload isn't meaningful).  As such, each action
 * has its own particular payload.
 *
 * For the sake of keeping files together, both the particular payload class and the action class
 * will be defined in the same header file.  Payloads should JUST be data (or streams), NO METHODS.
 *
 * Some actions perform changes that should raise events on the local system, and some
 * actions perform changes that will require waiting for events from the remote system.
 *
 * Therefore, this base action has been modified so that those events are set by each action.
 * When the event loop runs an action, it will then test to see if a local event was raised by the
 * action, and if so, perform the appropriate subsequent action.  If the action requires waiting
 * for a response from the remote system, then the event loop will sit there (presumably with the
 * ARTIM timer running) and wait for a response from the remote system.  Once a response is
 * obtained, then the the rest of the state transitions can happen.
 *
 */
class ULAction {
    private:
      //cannot copy a ULAction
      ULAction(const ULAction& inAction);

    protected:


    public:
      ULAction() {};
      //make sure destructors are virtual to avoid memory leaks
      virtual ~ULAction() {};

      virtual EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent) = 0;
    };
  }
}

#endif // GDCMULACTION_H
