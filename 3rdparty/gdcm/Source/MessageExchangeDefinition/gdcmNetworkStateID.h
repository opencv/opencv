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
#ifndef GDCMNETWORKSTATEID_H
#define GDCMNETWORKSTATEID_H

namespace gdcm {
  namespace network {

/**
 * Each network connection will be in a particular state at any given time.
 * Those states have IDs as described in the standard ps3.8-2009, roughly 1-13.
 * This enumeration lists those states. The actual ULState class will contain more information
 * about transitions to other states.
 *
 * name and date: 16 sept 2010 mmr
 */
    enum EStateID {
      eStaDoesNotExist = 0,
      eSta1Idle = 1,
      eSta2Open = 2,
      eSta3WaitLocalAssoc = 4,
      eSta4LocalAssocDone = 8,
      eSta5WaitRemoteAssoc = 16,
      eSta6TransferReady = 32,
      eSta7WaitRelease = 64,
      eSta8WaitLocalRelease = 128,
      eSta9ReleaseCollisionRqLocal = 256,
      eSta10ReleaseCollisionAc = 512,
      eSta11ReleaseCollisionRq = 1024,
      eSta12ReleaseCollisionAcLocal = 2048,
      eSta13AwaitingClose = 4096
    };

    const int cMaxStateID = 13;

    //the transition table is built on state indeces
    //this function will produce the index from the power-of-two EStateID
    inline int GetStateIndex(EStateID inState){
      switch (inState){
        case eStaDoesNotExist:
        default:
          return -1;
        case eSta1Idle:
          return 0;
        case eSta2Open:
          return 1;
        case eSta3WaitLocalAssoc:
          return 2;
        case eSta4LocalAssocDone:
          return 3;
        case eSta5WaitRemoteAssoc:
          return 4;
        case eSta6TransferReady:
          return 5;
        case eSta7WaitRelease:
          return 6;
        case eSta8WaitLocalRelease:
          return 7;
        case eSta9ReleaseCollisionRqLocal:
          return 8;
        case eSta10ReleaseCollisionAc:
          return 9;
        case eSta11ReleaseCollisionRq:
          return 10;
        case eSta12ReleaseCollisionAcLocal:
          return 11;
        case eSta13AwaitingClose:
          return 12;
      }
    }
  }
}

#endif //GDCMNETWORKSTATEID_H
