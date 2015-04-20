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
#ifndef GDCMCSTOREMESSAGES_H
#define GDCMCSTOREMESSAGES_H

#include "gdcmBaseCompositeMessage.h"

namespace gdcm{
class File;
  namespace network{
    class BasePDU;
/**
 * \brief CStoreRQ
 * this file defines the messages for the cecho action
 */
class CStoreRQ : public BaseCompositeMessage {
      std::vector<PresentationDataValue> ConstructPDV(const ULConnection &inConnection, const BaseRootQuery* inRootQuery);//to fulfill the virtual contract
    public:
      std::vector<PresentationDataValue> ConstructPDV(const ULConnection &inConnection,
        const File& file);
    };

/**
 * \brief CStoreRSP
 * this file defines the messages for the cecho action
 */
    class CStoreRSP : public BaseCompositeMessage {
      std::vector<PresentationDataValue> ConstructPDV(const ULConnection &inConnection, const BaseRootQuery* inRootQuery);//to fulfill the virtual contract
    public:
      std::vector<PresentationDataValue> ConstructPDV(const DataSet* inDataSet, const BasePDU* inPC);
    };
  }
}
#endif // GDCMCSTOREMESSAGES_H
