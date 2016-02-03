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
#ifndef GDCMULCONNECTION_H
#define GDCMULCONNECTION_H

#include "gdcmNetworkStateID.h"
#include "gdcmARTIMTimer.h"
#include "gdcmULConnectionInfo.h"
#include "gdcmPresentationContextRQ.h"
#include "gdcmDataElement.h"
#include "gdcmPresentationContextAC.h"
#include "gdcmPresentationContext.h"

class iosockinet;
class echo;
namespace gdcm{
  namespace network{

/**
 * \brief ULConnection
 * This is the class that contains the socket to another machine, and passes
 * data through itself, as well as maintaining a sense of state.
 *
 * The ULConnectionManager tells the ULConnection what data can actually be
 * sent.
 *
 * This class is done this way so that it can be eventually be replaced with a
 * ULSecureConnection, if such a protocol is warranted, so that all data that
 * passes through can be managed through a secure connection.  For now, this
 * class provides a simple pass-through mechanism to the socket itself.
 *
 * So, for instance, a gdcm object will be passes to this object, and it will
 * then get passed along the connection, if that connection is in the proper
 * state to do so.
 *
 * For right now, this class is not directly intended to be inherited from, but
 * the potential for future ULSecureConnection warrants the addition, rather
 * than having everything be managed from within the ULConnectionManager (or
 * this class) without a wrapper.
 *
 */
class ULConnection
{
      ULConnectionInfo mInfo;
      //this is a dirty dirty hack
      //but to establish an outgoing connection (scu), we need the echo service
      //to establish incoming, we just need a port and localhost, so an iosockinet works while an
      //echo would fail (probably because one already exists)
      echo* mEcho;
      iosockinet* mSocket;//of the three protocols offered by socket++-- echo, smtp, and ftp--
      //echo most closely matches what the DICOM standard describes as a network connection
      ARTIMTimer mTimer;

      EStateID mCurrentState;

      std::vector<PresentationContextRQ> mPresentationContexts;
      //this is our list of presentation contexts of what we can send
      uint32_t mMaxPDUSize;

      std::vector<PresentationContextAC> mAcceptedPresentationContexts;//these come back from the server
      //and tell us what can be sent over this connection

      TransferSyntaxSub cstorets;

      friend class ULActionAE6;
      void SetCStoreTransferSyntax( TransferSyntaxSub const & ts );
      friend class ULConnectionManager;
      TransferSyntaxSub const & GetCStoreTransferSyntax( ) const;
    public:

      ULConnection(const ULConnectionInfo& inUserInformation);
      //destructors are virtual to prevent memory leaks by inherited classes
      virtual ~ULConnection();

      EStateID GetState() const;
      void SetState(const EStateID& inState);//must be able to update state...

      //echo* GetProtocol();
      std::iostream* GetProtocol();
      void StopProtocol();

      ARTIMTimer& GetTimer();

      const ULConnectionInfo &GetConnectionInfo() const;

      //when the connection is first associated, the connection is told
      //the max packet/PDU size and the way in which to present data
      //(presentation contexts, etc). Store that here.
      void SetMaxPDUSize(uint32_t inSize);
      uint32_t GetMaxPDUSize() const;

      const PresentationContextAC *GetPresentationContextACByID(uint8_t id) const;
      const PresentationContextRQ *GetPresentationContextRQByID(uint8_t id) const;

      /// return 0 upon error
      uint8_t GetPresentationContextIDFromPresentationContext(PresentationContextRQ const & pc) const;

      std::vector<PresentationContextRQ> const & GetPresentationContexts() const;
      void SetPresentationContexts(const std::vector<PresentationContextRQ>& inContexts);

      void SetPresentationContexts(const std::vector<PresentationContext>& inContexts);

      //given a particular data element, presumably the SOP class,
      //find the presentation context for that SOP
      //NOT YET IMPLEMENTED
      PresentationContextRQ FindContext(const DataElement& de) const;

      std::vector<PresentationContextAC> const & GetAcceptedPresentationContexts() const;
      std::vector<PresentationContextAC> & GetAcceptedPresentationContexts();
      void AddAcceptedPresentationContext(const PresentationContextAC& inPC);

      /// used to establish scu connections
      bool InitializeConnection();

      /// used to establish scp connections
      bool InitializeIncomingConnection();
private:
  ULConnection(const ULConnection&);  // Not implemented.
  void operator=(const ULConnection&);  // Not implemented.

    };
  }
}

#endif // ULCONNECTION_H
