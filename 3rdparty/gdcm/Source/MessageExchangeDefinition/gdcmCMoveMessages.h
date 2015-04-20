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
#ifndef GDCMCMOVEMESSAGES_H
#define GDCMCMOVEMESSAGES_H

#include "gdcmBaseCompositeMessage.h"
#include "gdcmBaseRootQuery.h"

namespace gdcm{
  namespace network{
  class ULConnection;
/**
 * \brief CMoveRQ
 * this file defines the messages for the cmove action
 */
class CMoveRQ : public BaseCompositeMessage {
      //this class will fulfill the inheritance,
      //but additional information is needed by cmovd
      //namely, the root type or the calling AE-TITLE
      std::vector<PresentationDataValue> ConstructPDVByDataSet(const DataSet* inDataSet);
    public:
      std::vector<PresentationDataValue> ConstructPDV(
        const ULConnection &inConnection,
        const BaseRootQuery* inRootQuery);
    };

/**
 * \brief CMoveRSP
 * this file defines the messages for the cmove action
 */
class CMoveRSP : public BaseCompositeMessage {
    public:
      std::vector<PresentationDataValue> ConstructPDVByDataSet(const DataSet* inDataSet);
    };

    class CMoveCancelRq : public BaseCompositeMessage {
    public:
      std::vector<PresentationDataValue> ConstructPDVByDataSet(const DataSet* inDataSet);
    };
  }
}
#endif
