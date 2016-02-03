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
#ifndef GDCMCFINDMESSAGES_H
#define GDCMCFINDMESSAGES_H

#include "gdcmBaseCompositeMessage.h"
#include "gdcmBaseRootQuery.h"

namespace gdcm
{
namespace network
{

/**
 * \brief CFindRQ
 * this file defines the messages for the cfind action
 */
class CFindRQ : public BaseCompositeMessage
{
public:
  std::vector<PresentationDataValue> ConstructPDV(const ULConnection &inConnection,
    const BaseRootQuery* inRootQuery);
};

/**
 * \brief CFindRSP
 * this file defines the messages for the cfind action
 */
class CFindRSP : public BaseCompositeMessage {
public:
  std::vector<PresentationDataValue> ConstructPDVByDataSet(const DataSet* inDataSet);
};

/**
 * \brief CFindCancelRQ
 * this file defines the messages for the cfind action
 */
  class CFindCancelRQ : public BaseCompositeMessage {
  public:
    std::vector<PresentationDataValue> ConstructPDVByDataSet(const DataSet* inDataSet);
  };
}
}
#endif
