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

#include "gdcmCompositeNetworkFunctions.h"
#include "gdcmTag.h"
#include "gdcmQueryFactory.h"

int TestEcho(int , char *[])
{
  std::string hostname = "www.dicomserver.co.uk";
  uint16_t port = 11112;
  std::string callaetitle = "GDCM_ROCKS";
  std::string callingaetitle = "ACME1";

  bool didItWork = gdcm::CompositeNetworkFunctions::CEcho( hostname.c_str(), port,
    callingaetitle.c_str(), callaetitle.c_str() );

  return (didItWork ? 0:1);
}
