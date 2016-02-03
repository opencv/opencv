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
#ifndef GDCMBASEPDU_H
#define GDCMBASEPDU_H

#include "gdcmTypes.h"

namespace gdcm
{
namespace network
{

/**
 * \brief BasePDU
 * base class for PDUs
 *
 * all PDUs start with the first ten bytes as specified:
 * 01 PDU type
 * 02 reserved
 * 3-6 PDU Length (unsigned)
 * 7-10 variable
 *
 * on some, 7-10 are split (7-8 as protocol version in Associate-RQ, for instance,
 * while associate-rj splits those four bytes differently).
 *
 * Also common to all the PDUs is their ability to read and write to a stream.
 *
 * So, let's just get them all bunched together into one (abstract) class, shall we?
 *
 *  Why?
 *  1) so that the ULEvent can have the PDU stored in it, since the event takes PDUs and not
 *  other class structures (other class structures get converted into PDUs)
 *  2) to make reading PDUs in the event loop cleaner
 */
class BasePDU
{
public:
  virtual ~BasePDU() {}

  virtual std::istream &Read(std::istream &is) = 0;
  virtual const std::ostream &Write(std::ostream &os) const = 0;

  virtual size_t Size() const = 0;
  virtual void Print(std::ostream &os) const = 0;

  virtual bool IsLastFragment() const = 0;
};

} // end namespace network
} // end namespace gdcm

#endif // GDCMBASEPDU_H
