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
#ifndef GDCMULACTIONAR_H
#define GDCMULACTIONAR_H

#include "gdcmULAction.h"

/**
This header defines the classes for the AR Actions,
Association Release Related Actions (Table 9-8 of ps 3.8-2009).

Since each class is essentially a placeholder for a function pointer, I'm breaking with having
each class have its own file for the sake of brevity of the number of files.
*/

namespace gdcm {
  namespace network {

    //Send A-RELEASE-RQ-PDU
    //Next State: eSta7WaitRelease
    class ULActionAR1 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue A-RELEASE indication primitive
    //Next State: eSta8WaitLocalRelease
    class ULActionAR2 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue A-RELEASE confirmation primitive, and close transport connection
    //Next State: eSta1Idle
    class ULActionAR3 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue A-RELEASE-RP PDU and start ARTIM timer
    //Next State: eSta13AwaitingClose
    class ULActionAR4 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Stop ARTIM timer
    //Next State: eSta1Idle
    class ULActionAR5 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue P-Data indication
    //Next State: eSta7WaitRelease
    class ULActionAR6 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue P-DATA-TF PDU
    //Next State: eSta8WaitLocalRelease
    class ULActionAR7 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue A-RELEASE indication (release collision):
    //- If association-requestor, next state is eSta9ReleaseCollisionRqLocal
    //- if not, next state is eSta10ReleaseCollisionAc
    class ULActionAR8 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Send A-RELEASE-RP PDU
    //Next State: eSta11ReleaseCollisionRq
    class ULActionAR9 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue A-RELEASE confirmation primitive
    //Next State: eSta12ReleaseCollisionAcLocal
    class ULActionAR10 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };
  }
}
#endif // GDCMULACTIONAR_H
