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
#ifndef GDCMULCONNECTIONCALLBACK_H
#define GDCMULCONNECTIONCALLBACK_H

#include "gdcmTypes.h" //to be able to export the class

namespace gdcm 
{
  class DataSet;
  namespace network
  {
    ///When a dataset comes back from a query/move/etc, the result can either be
    ///stored entirely in memory, or could be stored on disk.  This class provides
    ///a mechanism to indicate what the ULConnectionManager should do with datasets
    ///that are produced through query results.
    ///The ULConnectionManager will call the HandleDataSet function during the course
    ///of receiving datasets.  Particular implementations should fill in what that
    ///function does, including updating progress, etc.
    ///NOTE: since cmove requires that multiple event loops be employed,
    ///the callback function MUST set mHandledDataSet to true.
    ///otherwise, the cmove event loop handler will not know data was received, and
    ///proceed to end the loop prematurely.
    class GDCM_EXPORT ULConnectionCallback {
      bool mHandledDataSet;
    protected:
      bool mImplicit;
      //inherited callbacks MUST call this function for the cmove loop to work properly
      void DataSetHandled() { mHandledDataSet = true; }
    public:
      ULConnectionCallback():mHandledDataSet(false),mImplicit(true){}
      virtual ~ULConnectionCallback() {}; //placeholder for inherited objects
      virtual void HandleDataSet(const DataSet& inDataSet) = 0;
      virtual void HandleResponse(const DataSet& inDataSet) = 0;

      bool DataSetHandles() const { return mHandledDataSet; }
      void ResetHandledDataSet() { mHandledDataSet = false; }

      void SetImplicitFlag( const bool imp ) { mImplicit = imp; }
    };
  }
}
#endif //GDCMULCONNECTIONCALLBACK_H
