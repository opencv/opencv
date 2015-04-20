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

#include "gdcmULBasicCallback.h"

namespace gdcm {
namespace network {

void ULBasicCallback::HandleDataSet(const DataSet& inDataSet)
{
  mDataSets.push_back(inDataSet);
  DataSetHandled();
}

std::vector<DataSet> const & ULBasicCallback::GetDataSets() const
{
  return mDataSets;
}

void ULBasicCallback::HandleResponse(const DataSet& inDataSet)
{
  mResponses.push_back(inDataSet);
}

std::vector<DataSet> const & ULBasicCallback::GetResponses() const
{
  return mResponses;
}

} // end namespace network
} // end namespace gdcm
