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
#ifndef GDCMULCONNECTIONBASICCALLBACK_H
#define GDCMULCONNECTIONBASICCALLBACK_H

#include "gdcmULConnectionCallback.h"
#include "gdcmDataSet.h"
#include <vector>

namespace gdcm 
{
  namespace network
    {
    /**
     * \brief ULBasicCallback
     * This is the most basic of callbacks for how the ULConnectionManager handles
     * incoming datasets.  DataSets are just concatenated to the mDataSets vector,
     * and the result can be pulled out of the vector by later code.
     * Alternatives to this method include progress updates, saving to disk, etc.
     * This class is NOT THREAD SAFE.  Access the dataset vector after the
     * entire set of datasets has been returned by the ULConnectionManager.
     */
    class GDCM_EXPORT ULBasicCallback : public ULConnectionCallback
    {
    std::vector<DataSet> mDataSets;
    std::vector<DataSet> mResponses;
  public:
    ULBasicCallback() {};
    virtual ~ULBasicCallback() {} //empty, for later inheritance

    virtual void HandleDataSet(const DataSet& inDataSet);
    virtual void HandleResponse(const DataSet& inDataSet);

    std::vector<DataSet> const & GetDataSets() const;
    std::vector<DataSet> const & GetResponses() const;
    };
    } // end namespace network
} // end namespace gdcm

#endif // GDCMULCONNECTIONBASICCALLBACK_H
