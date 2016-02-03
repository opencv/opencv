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
#ifndef GDCMULTRANSITIONTABLE_H
#define GDCMULTRANSITIONTABLE_H

#include "gdcmNetworkStateID.h"
#include "gdcmNetworkEvents.h"
#include "gdcmULAction.h"

#include <cstdlib>  // NULL

namespace gdcm {
class Subject;
  namespace network{
class ULConnection;
class ULAction;
class ULEvent;

    //The transition dictates the action that should be taken from the start state to the end state
    struct Transition {
      int mEnd;
      ULAction* mAction;
      Transition(){
        mEnd = eStaDoesNotExist;
        mAction = NULL;
      }
      ~Transition(){
        if (mAction != NULL){
          delete mAction;
          mAction = NULL;
        }
      }
      Transition(int inEndState, ULAction* inAction){
        mEnd = inEndState;
        mAction = inAction;
      }
      static Transition* MakeNew(int inEndState, ULAction* inAction){
        return new Transition(inEndState, inAction);
      }
    };

    //used to define a row in table 9-10 of 3.8 2009
    //the transition table is events, then state,
    //then the transition itself (which has the event
    //and start state implied by their starting locations)
    //don't need to store the event; that's implicitly defined in the Table itself by location
    class TableRow{
    public:
      TableRow() {
        for(int stateIndex = 0; stateIndex < cMaxStateID; ++stateIndex)
          {
          transitions[stateIndex] = NULL;
          }
      }
      ~TableRow() {
        for(int stateIndex = 0; stateIndex < cMaxStateID; ++stateIndex)
          {
          Transition *t = transitions[stateIndex];
          delete t;
          }
       }
      Transition *transitions[cMaxStateID];

      //copy constructor for stl additions into the transition table below.
    };

/**
 * \brief ULTransitionTable
 * The transition table of all the ULEvents, new ULActions, and ULStates.
 *
 * Based roughly on the solutions in player2.cpp in the boost examples and this
 * so question:
 * http://stackoverflow.com/questions/1647631/c-state-machine-design
 *
 * The transition table is constructed of TableRows.  Each row is based on an
 * event, and an event handler in the TransitionTable object takes a given
 * event, and then finds the given row.
 *
 * Then, given the current state of the connection, determines the appropriate
 * action to take and then the state to transition to next.
 *
 */
class ULTransitionTable
{
    private:
      TableRow mTable[cMaxEventID];
    public:
      ULTransitionTable();

      void HandleEvent(Subject*s,ULEvent& inEvent, ULConnection& inConnection,
        bool& outWaitingForEvent, EEventID& outRaisedEvent) const;

      void PrintTable() const; //so that the table can be printed and verified against the DICOM standard
    };
  }
}
#endif // GDCMULTRANSITIONTABLE_H
