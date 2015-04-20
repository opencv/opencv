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

//this class contains all the information about a particular connection
//as established by the user.  That is, it's:
// User Information
// Calling AE Title
// Called AE Title
// IP address/computer name
// IP Port
//A connection must be established with this information, that's subsequently
//placed into various primitives for actual communication.
#include "gdcmULConnectionInfo.h"
#include "gdcmAAssociateRQPDU.h"

#include <socket++/sockinet.h>//for setting up the local socket

#if defined(_WIN32)
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

namespace gdcm
{
namespace network
{

ULConnectionInfo::ULConnectionInfo()
{
}

      //it is possible to misinitialize this object, so
      //have it return false if something breaks (ie, given AEs are bigger than 16 characters,
      //no name or IP address).
bool ULConnectionInfo::Initialize(UserInformation const & inUserInformation,
        const char *inCalledAETitle, const char *inCallingAETitle,
        unsigned long inCalledIPAddress, int inCalledIPPort,
        std::string inCalledComputerName)
{
  if (inCalledIPAddress == 0 && inCalledComputerName.empty()){
    return false;
  }
  assert( inCalledAETitle );
  assert( inCallingAETitle );
  assert( AAssociateRQPDU::IsAETitleValid( inCalledAETitle ) );
  assert( AAssociateRQPDU::IsAETitleValid( inCallingAETitle ) );
  const size_t lcalled  = strlen( inCalledAETitle );
  const size_t lcalling = strlen( inCallingAETitle );
  mCalledAETitle = std::string(inCalledAETitle, lcalled > 16 ? 16 : lcalled );
  mCallingAETitle = std::string(inCallingAETitle, lcalling > 16 ? 16 : lcalling );
  mCalledComputerName = inCalledComputerName;
  mCalledIPPort = inCalledIPPort;
  mCalledIPAddress = inCalledIPAddress;

  //test to see if the given computer name is actually an IP address
  if (mCalledIPAddress == 0 && !mCalledComputerName.empty()){
    mCalledIPAddress = inet_addr(mCalledComputerName.c_str());
  //  if (mCalledIPAddress != 0)
  //    mCalledComputerName = "";
  }

  //mUserInformation = inUserInformation;
  (void)inUserInformation;
  return true;
}

//UserInformation ULConnectionInfo::GetUserInformation() const{
//  return mUserInformation;
//}
const char* ULConnectionInfo::GetCalledAETitle() const{
  return mCalledAETitle.c_str();
}
const char* ULConnectionInfo::GetCallingAETitle() const{
  return mCallingAETitle.c_str();
}

unsigned long ULConnectionInfo::GetCalledIPAddress() const{
  return mCalledIPAddress;
}
int ULConnectionInfo::GetCalledIPPort() const{
  return mCalledIPPort;
}
std::string ULConnectionInfo::GetCalledComputerName() const{
  return mCalledComputerName;
}

void ULConnectionInfo::SetMaxPDULength(unsigned long inMaxPDULength){
  mMaxPDULength = inMaxPDULength;
}
unsigned long ULConnectionInfo::GetMaxPDULength() const{
  return mMaxPDULength;
}

} // end namespace network
} // end namespace gdcm
