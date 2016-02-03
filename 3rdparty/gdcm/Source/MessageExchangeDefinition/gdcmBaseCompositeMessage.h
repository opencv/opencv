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
#ifndef GDCMBASECOMPOSITEMESSSAGE_H
#define GDCMBASECOMPOSITEMESSSAGE_H

#include "gdcmPresentationDataValue.h"
#include "gdcmBaseRootQuery.h"

#include <vector>

namespace gdcm
{
  namespace network
    {
class ULConnection;
/**
 * \brief BaseCompositeMessage
 * The Composite events described in section 3.7-2009 of the DICOM standard all
 * use their own messages.  These messages are constructed using Presentation
 * Data Values, from section 3.8-2009 of the standard, and then fill in
 * appropriate values in their datasets.
 *
 * So, for the five composites:
 * \li C-ECHO
 * \li C-FIND
 * \li C-MOVE
 * \li C-GET
 * \li C-STORE
 * there are a series of messages.  However, all of these messages are obtained
 * as part of a PDataPDU, and all have to be placed there.  Therefore, since
 * they all have shared functionality and construction tropes, that will be put
 * into a base class.  Further, the base class will be then returned by the
 * factory class, gdcmCompositePDUFactory.
 *
 * This is an abstract class.  It cannot be instantiated on its own.
 */
class BaseCompositeMessage
{
    public:
      virtual ~BaseCompositeMessage() {}
      //construct the appropriate pdv and dataset for this message
      //for instance, setting tag 0x0,0x100 to the appropriate value
      //the pdv, as described in Annex E of 3.8-2009, is the first byte
      //of the message (the MessageHeader), and then the subsequent dataset
      //that describes the operation.
      virtual std::vector<PresentationDataValue> ConstructPDV(const ULConnection &inConnection,
        const BaseRootQuery * inRootQuery) = 0;
    };
  }
}
#endif //BASECOMPOSITEMESSSAGE_H
