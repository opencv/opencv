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
#ifndef GDCMPDUFACTORY_H
#define GDCMPDUFACTORY_H

#include "gdcmTypes.h"
#include "gdcmNetworkEvents.h"
#include "gdcmULConnection.h"
#include "gdcmPresentationDataValue.h"

namespace gdcm{
  class BaseRootQuery;
  class File;
  namespace network{
    class BasePDU;

/**
 * \brief PDUFactory basically, given an initial byte, construct the
 * appropriate PDU. This way, the event loop doesn't have to know about all
 * the different PDU types.
 */
  class PDUFactory {
      public:
      static BasePDU* ConstructPDU(uint8_t itemtype);//eventually needs to be smartpointer'd
      static EEventID DetermineEventByPDU(const BasePDU* inPDU);
      static BasePDU* ConstructReleasePDU();
      static BasePDU* ConstructAbortPDU();

      //these are the composite PDU construction methods for the PDataPDUs.
      //basically, builds a pdatapdu, and then puts the appropriate information in
      //for the appropriate composite service (c-echo, c-find, c-store, c-get, c-move)
      //the connection is necessary to construct the stream of PDVs that will
      //be then placed into the vector of PDUs
      static std::vector<BasePDU*> CreateCEchoPDU(const ULConnection& inConnection);
      static std::vector<BasePDU*> CreateCStoreRQPDU(const ULConnection& inConnection, const File &file);
      static std::vector<BasePDU*> CreateCStoreRSPPDU(const DataSet *inDataSet, const BasePDU* inPC);
      static std::vector<BasePDU*> CreateCFindPDU(const ULConnection& inConnection, const BaseRootQuery* inRootQuery);
      static std::vector<BasePDU*> CreateCMovePDU(const ULConnection& inConnection, const BaseRootQuery* inRootQuery);

      //given data pdus, produce the presentation data values stored within.
      //all operations have these as the payload of the data sending operation
      //however, echo does not have a dataset in the pdv.
      static std::vector<PresentationDataValue> GetPDVs(const std::vector<BasePDU*> & inDataPDUs);
    };
  }
}
#endif //GDCMPDUFACTORY_H
