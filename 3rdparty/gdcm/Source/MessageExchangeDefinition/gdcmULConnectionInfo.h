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
#ifndef GDCMULCONNECTIONINFO_H
#define GDCMULCONNECTIONINFO_H

#include "gdcmUserInformation.h"
#include <string>

namespace gdcm{
  namespace network {
/**
 * \brief ULConnectionInfo
 * this class contains all the information about a particular connection
 * as established by the user.  That is, it's:
 *  User Information
 *  Calling AE Title
 *  Called AE Title
 *  IP address/computer name
 *  IP Port
 * A connection must be established with this information, that's subsequently
 * placed into various primitives for actual communication.
 */
class ULConnectionInfo {
      UserInformation mUserInformation;

      std::string mCalledAETitle;
      std::string mCallingAETitle;

      unsigned long mCalledIPAddress;
      int mCalledIPPort;
      std::string mCalledComputerName; //either the IP or the name has to be filled in
  
      unsigned long mMaxPDULength;
    public:
      ULConnectionInfo();

      //it is possible to misinitialize this object, so
      //have it return false if something breaks (ie, given AEs are bigger than 16 characters,
      //no name or IP address).
      bool Initialize(UserInformation const &inUserInformation,
        const char *inCalledAETitle, const char *inCallingAETitle,
        unsigned long inCalledIPAddress, int inCalledIPPort,
        std::string inCalledComputerName);

      //UserInformation GetUserInformation() const;
      const char* GetCalledAETitle() const;
      const char* GetCallingAETitle() const;

      unsigned long GetCalledIPAddress() const;
      int GetCalledIPPort() const;
      std::string GetCalledComputerName() const;

      //CStore needs to know the max pdu length, so the value gets initialized
      //when a cstore connection is established (but not for the others).
      void SetMaxPDULength(unsigned long inMaxPDULength);
      unsigned long GetMaxPDULength() const;
    };
  }
}

#endif //GDCMULCONNECTIONINFO_H
