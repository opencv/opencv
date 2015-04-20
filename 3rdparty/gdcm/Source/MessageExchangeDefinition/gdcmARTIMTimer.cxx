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

This file contains the code for the ARTIM timer.

Basically, the ARTIM timer will just get the wall time when it's started,
and then can be queried for the current time, and then can be stopped (ie,
the start time reset).

Because we're trying to do this without threading, we should be able to 'start' the
ARTIM timer by this mechanism, and then when waiting for a particular response, tight
loop that with sleep calls and determinations of when the ARTIM timer has reached its
peak.  As such, this isn't a strict 'timer' in the traditional sense of the word,
but more of a time keeper.

*/

#include "gdcmARTIMTimer.h"
#include "gdcmSystem.h"

namespace gdcm
{
namespace network
{

//initiates the start and timeout at -1;
ARTIMTimer::ARTIMTimer(){
  mStartTime = 0;
  mTimeOut = 0;
}
double ARTIMTimer::GetCurrentTime() const{
  return 0; //platform-specific timing functions go here...
}

void ARTIMTimer::Start(){
  mStartTime = GetCurrentTime();
}
void ARTIMTimer::SetTimeout(double inTimeOut){
  mTimeOut = inTimeOut;
}

double ARTIMTimer::GetTimeout() const{
  return mTimeOut;
}
double ARTIMTimer::GetElapsedTime() const{
  if (mStartTime > 0){
    return GetCurrentTime() - mStartTime;
  } else {
    return -1; //not started yet
  }
}

bool ARTIMTimer::GetHasExpired() const{
  double theElapsed = GetElapsedTime();
  if (theElapsed > 0){
    return theElapsed > mTimeOut;
  } else {
    return false; //not started yet
  }
}

void ARTIMTimer::Stop() {
  mStartTime = -1;//stop the timer by resetting it.
}

} // end namespace network
} // end namespace gdcm
