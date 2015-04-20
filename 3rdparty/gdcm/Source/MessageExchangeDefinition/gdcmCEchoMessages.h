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
#ifndef GDCMCECHOMESSAGES_H
#define GDCMCECHOMESSAGES_H

#include "gdcmBaseCompositeMessage.h"

namespace gdcm{
  namespace network{

class ULConnection;

/**
 * \brief CEchoRQ
 * this file defines the messages for the cecho action
 */
class CEchoRQ : public BaseCompositeMessage {
    public:
      std::vector<PresentationDataValue> ConstructPDV(const ULConnection &inConnection,
        const BaseRootQuery* inRootQuery);
    };

/**
 * \brief CEchoRSP
 * this file defines the messages for the cecho action
 */
    class CEchoRSP : public BaseCompositeMessage {
    public:
      std::vector<PresentationDataValue> ConstructPDVByDataSet(const DataSet* inDataSet);
    };
  }
}
#endif // GDCMCECHOMESSAGES_H
