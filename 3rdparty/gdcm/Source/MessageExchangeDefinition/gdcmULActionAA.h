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
#ifndef GDCMULACTIONAA_H
#define GDCMULACTIONAA_H

#include "gdcmULAction.h"

/**
This header defines the classes for the AA Actions,
Association Abort Related Actions (Table 9-9 of ps 3.8-2009).

Since each class is essentially a placeholder for a function pointer, I'm breaking with having
each class have its own file for the sake of brevity of the number of files.
*/

namespace gdcm {
  namespace network {

    //Send A-ABORT PDU (service-user source) and start (or restart if already started) ARTIM timer
    //Next State: eSta13AwaitingClose
    class ULActionAA1 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Stop ARTIM timer if running.  Close transport connection.
    //Next State: eSta1Idle
    class ULActionAA2 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //If (service-user initiated abort)
    //- issue A-ABORT indication and close transport connection
    //otherwise (service-provider initiated abort):
    //- issue A-P-ABORT indication and close transport connection
    //Next State: eSta1Idle
    class ULActionAA3 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue A-P-ABORT indication primitive
    //Next State: eSta1Idle
    class ULActionAA4 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Stop ARTIM timer
    //Next State: eSta1Idle
    class ULActionAA5 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Ignore PDU
    //Next State: eSta13AwaitingClose
    class ULActionAA6 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Send A-ABORT PDU
    //Next State: eSta13AwaitingClose
    class ULActionAA7 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Send A-ABORT PDU (service-provider source), issue an A-P-ABORT indication, and start ARTIM timer
    //Next State: eSta13AwaitingClose
    class ULActionAA8 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };
  }
}

#endif // GDCMULACTIONAA_H
