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
#ifndef GDCMULACTIONAE_H
#define GDCMULACTIONAE_H

#include "gdcmULAction.h"

/**
This header defines the classes for the AE Actions,
Association Establishment Related Actions (Table 9-6 of ps 3.8-2009).

Since each class is essentially a placeholder for a function pointer, I'm breaking with having
each class have its own file for the sake of brevity of the number of files.

*/

namespace gdcm {
  namespace network {

    //Issue TRANSPORT CONNECT request primitive to local transport service.
    class ULActionAE1 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Send A-ASSOCIATE-RQ-PDU
    //Next State: eSta5WaitRemoteAssoc
    class ULActionAE2 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue A-ASSOCIATE confirmation (accept) primitive
    //Next State: eSta6TransferReady
    class ULActionAE3 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue A-ASSOCIATE confirmation (reject) primitive and close transport connection
    //Next State: eSta1Idle
    class ULActionAE4 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Issue Transport connection response primitive, start ARTIM timer
    //Next State: eSta2Open
    class ULActionAE5 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Stop ARTIM timer and if A-ASSOCIATE-RQ acceptable by service-provider:
    //- issue A-ASSOCIATE indication primitive
    //Next state: eSta3WaitLocalAssoc
    //otherwise:
    //- issue A-ASSOCIATE-RJ-PDU and start ARTIM timer
    //Next state: eSta13AwaitingClose
    class ULActionAE6 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Send A-ASSOCIATE-AC PDU
    //Next State: eSta6TransferReady
    class ULActionAE7 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };

    //Send A-ASSOCIATE-RJ PDU and start ARTIM timer
    //Next State: eSta13AwaitingClose
    class ULActionAE8 : public ULAction {
    public:
      EStateID PerformAction(Subject *s, ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent);
    };
  }
}
#endif // GDCMULACTIONAE_H
