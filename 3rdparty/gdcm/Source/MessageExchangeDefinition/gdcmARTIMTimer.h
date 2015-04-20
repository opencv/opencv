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
#ifndef GDCMARTIMTIMER_H
#define GDCMARTIMTIMER_H

namespace gdcm {
  namespace network{
/** \brief ARTIMTimer
 * This file contains the code for the ARTIM timer.
 *
 * Basically, the ARTIM timer will just get the wall time when it's started,
 * and then can be queried for the current time, and then can be stopped (ie,
 * the start time reset).
 *
 * Because we're trying to do this without threading, we should be able to 'start' the
 * ARTIM timer by this mechanism, and then when waiting for a particular response, tight
 * loop that with sleep calls and determinations of when the ARTIM timer has reached its
 * peak.  As such, this isn't a strict 'timer' in the traditional sense of the word,
 * but more of a time keeper.
 *
 * There can be only one ARTIM timer per connection.
 */
class ARTIMTimer
{
    private:
      double mStartTime; //ms timing should be good enough, but there are also
      //high-resolution timing options.  Those return doubles.  For now,
      //go with integer timing solutions based on milliseconds (DWORD on windows),
      //but leave as doubles to ease transitions to other timing methods.

      double mTimeOut;
      //once GetCurrentTime() -mStartTime > mTimeout, GetHasExpired returns true.

      double GetCurrentTime() const;//a platform-specific implementation of getting the
      //current time.

    public:
      ARTIMTimer(); //initiates the start and timeout at -1;
      void Start(); //'start' the timer by getting the current wall time
      void Stop();//'stop' the timer by resetting the 'start' to -1;
      void SetTimeout(double inTimeout);
      double GetTimeout() const;

      double GetElapsedTime() const;

      bool GetHasExpired() const;

    };
  }
}

#endif //GDCMARTIMTIMER_H
