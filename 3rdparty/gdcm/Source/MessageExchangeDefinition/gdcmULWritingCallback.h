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
#ifndef GDCMULCONNECTIONWRITINGCALLBACK_H
#define GDCMULCONNECTIONWRITINGCALLBACK_H

#include "gdcmULConnectionCallback.h"

namespace gdcm 
{
class DataSet;
namespace network
{
/* \brief ULWritingCallback
 * This is the most basic of callbacks for how the ULConnectionManager handles
 * incoming datasets.  DataSets are immediately written to disk as soon as they
 * are received.  NOTE that if the incoming connection is faster than the disk
 * writing speed, this callback could cause some pileups!
 */
class GDCM_EXPORT ULWritingCallback : public ULConnectionCallback
{
  std::string mDirectoryName;
public:
  ULWritingCallback() {};
  virtual ~ULWritingCallback() {} //empty, for later inheritance

  ///provide the directory into which all files are written.
  void SetDirectory(const std::string& inDirectoryName) { mDirectoryName = inDirectoryName; }

  virtual void HandleDataSet(const DataSet& inDataSet);
  virtual void HandleResponse(const DataSet& inDataSet);
};
} // end namespace network
} // end namespace gdcm

#endif //GDCMULCONNECTIONWRITINGCALLBACK_H
