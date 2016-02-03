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
#ifndef GDCMCOMPOSITEMESSAGEFACTORY_H
#define GDCMCOMPOSITEMESSAGEFACTORY_H

#include "gdcmPresentationDataValue.h"
#include "gdcmULConnection.h"

namespace gdcm {
  class BaseRootQuery;
  class File;
  namespace network {
    class BasePDU;
/**
 * \brief CompositeMessageFactory
 * This class constructs PDataPDUs, but that have been specifically constructed for the
 * composite DICOM services (C-Echo, C-Find, C-Get, C-Move, and C-Store).  It will also handle
 * parsing the incoming data to determine which of the CompositePDUs the incoming data is, and
 * so therefore allowing the scu to determine what to do with incoming data (if acting as
 * a storescp server, for instance).
 */
class CompositeMessageFactory
{
    public:
      //the echo request only needs a properly constructed PDV.
      //find, move, etc, may need something more robust, but since those are
      //easily placed into the appropriate pdatapdu in the pdufactory,
      //this approach without a base class (but done internally) is useful.
      static std::vector<PresentationDataValue> ConstructCEchoRQ(const ULConnection& inConnection);

      static std::vector<PresentationDataValue> ConstructCStoreRQ(const ULConnection& inConnection,const File &file);
      static std::vector<PresentationDataValue> ConstructCStoreRSP(const DataSet *inDataSet, const BasePDU* inPC);

      static  std::vector<PresentationDataValue> ConstructCFindRQ(const ULConnection& inConnection, const BaseRootQuery* inRootQuery);

      static  std::vector<PresentationDataValue> ConstructCMoveRQ(const ULConnection& inConnection, const BaseRootQuery* inRootQuery);


    };
  }
}

#endif // GDCMCOMPOSITEMESSAGEFACTORY_H
