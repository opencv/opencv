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
#include "gdcmULConnection.h"

#include <algorithm> // std::find
#include <socket++/echo.h>

namespace gdcm
{
namespace network
{
ULConnection::ULConnection(const ULConnectionInfo& inConnectInfo)
{
  mCurrentState = eSta1Idle;
  mSocket = NULL;
  mEcho = NULL;
  mInfo = inConnectInfo;

  TransferSyntaxSub ts1;
  ts1.SetNameFromUID( UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM );
  SetCStoreTransferSyntax( ts1 );
}

ULConnection::~ULConnection()
{
  if (mEcho != NULL)
    {
    delete mEcho;
    mEcho = NULL;
    }
  if (mSocket != NULL)
    {
    delete mSocket;
    mSocket = NULL;
    }
}

EStateID ULConnection::GetState() const
{
  return mCurrentState;
}

void ULConnection::SetState(const EStateID& inState)
{
  mCurrentState = inState;
}

//echo* ULConnection::GetProtocol(){
std::iostream* ULConnection::GetProtocol()
{
  if (mEcho)
    {
    return mEcho;
    }
  //be careful-- the socket doesn't get removed when the protocol is stopped,
  //because then subsequent cstores will break.
  if (mSocket)
    {
    return mSocket;
    }
  return NULL;
}

ARTIMTimer& ULConnection::GetTimer()
{
  return mTimer;
}

const ULConnectionInfo &ULConnection::GetConnectionInfo() const
{
  return mInfo;
}

void ULConnection::SetMaxPDUSize(uint32_t inSize)
{
  mMaxPDUSize = inSize;
}

uint32_t ULConnection::GetMaxPDUSize() const
{
  return mMaxPDUSize;
}

std::vector<PresentationContextRQ> const &
ULConnection::GetPresentationContexts() const
{
  return mPresentationContexts;
}

void ULConnection::SetPresentationContexts(
  const std::vector<PresentationContext>& inContexts)
{
  mPresentationContexts.clear();
  for( size_t i = 0; i < inContexts.size(); ++i )
    {
    PresentationContext const &in = inContexts[i];
    mPresentationContexts.push_back( in );
    }
}

void ULConnection::SetPresentationContexts(
  const std::vector<PresentationContextRQ>& inContexts)
{
  mPresentationContexts = inContexts;
}

std::vector<PresentationContextAC> & ULConnection::GetAcceptedPresentationContexts()
{
  return mAcceptedPresentationContexts;
}

std::vector<PresentationContextAC> const & ULConnection::GetAcceptedPresentationContexts() const
{
  return mAcceptedPresentationContexts;
}
void ULConnection::AddAcceptedPresentationContext(const PresentationContextAC& inPC)
{
  mAcceptedPresentationContexts.push_back(inPC);
}

//given a particular data element, presumably the SOP class,
//find the presentation context for that SOP
//NOT YET IMPLEMENTED
PresentationContextRQ ULConnection::FindContext(const DataElement& ) const
{
  PresentationContextRQ empty;
  assert( 0 && "TODO" );
  return empty;
}

bool ULConnection::InitializeConnection()
{
  try
    {
    echo* p = new echo(protocol::tcp);
    if (GetConnectionInfo().GetCalledIPPort() == 0)
      {
      if (!GetConnectionInfo().GetCalledComputerName().empty())
        (*p)->connect(GetConnectionInfo().GetCalledComputerName().c_str());
      else
        (*p)->connect(GetConnectionInfo().GetCalledIPAddress());
      }
    else
      {
      if (!GetConnectionInfo().GetCalledComputerName().empty())
        (*p)->connect(GetConnectionInfo().GetCalledComputerName().c_str(),
          GetConnectionInfo().GetCalledIPPort());
      }
    //make sure to convert timeouts to platform appropriate values.
    (*p)->recvtimeout((int)GetTimer().GetTimeout());
    (*p)->sendtimeout((int)GetTimer().GetTimeout());
    if (mEcho != NULL)
      {
      delete mEcho;
      mEcho = NULL;
      }
    if (mSocket != NULL)
      {
      delete mSocket;
      mSocket = NULL;
      }
    mEcho = p;
    }
  catch ( sockerr &err )
    {
    (void)err;  //to avoid unreferenced variable warning on release
    gdcmWarningMacro( "Unable to open connection with exception " << err.what()
      << std::endl );
    return false;
    }
  catch (std::exception& ex)
    {
    (void)ex;  //to avoid unreferenced variable warning on release
    //unable to establish connection, so break off.
    gdcmWarningMacro( "Unable to open connection with exception " << ex.what()
      << std::endl );
    return false;
    }
  return true;
}

bool ULConnection::InitializeIncomingConnection()
{
  try
    {
    if (mEcho != NULL)
      {
      delete mEcho;
      mEcho = NULL;
      }
    if (mSocket != NULL)
      {
      delete mSocket;
      mSocket = NULL;
      }
    sockinetbuf sin (sockbuf::sock_stream);
    // http://hea-www.harvard.edu/~fine/Tech/addrinuse.html
    int val = 1;
    sin.setopt( SO_REUSEADDR, &val, sizeof(val) );
    sin.bind( mInfo.GetCalledIPPort() );
    //int theRecvTimeout = 
    sin.recvtimeout(60);//(int)GetTimer().GetTimeout());
    //int theSendTimeout = 
    sin.sendtimeout(60);//(int)GetTimer().GetTimeout());
    sin.listen();
    //sin.debug( true );
    if (sin.is_readready(60, 0))
      {
      mSocket = new iosockinet(sin.accept());
      }
    else
      {
      gdcmDebugMacro( "Call to is_readready failed" );
      SetState(eStaDoesNotExist);
      return false; //no connection here, so have to initialize later.
      }
    SetState(eSta2Open);

    /*
    if (mSocket != NULL){
    delete mSocket;
    }
    mSocket = new protocol();
    sockinetaddr theAddy(GetConnectionInfo().GetCalledComputerName().c_str(),
    GetConnectionInfo().GetCalledIPPort());
    mSocket->rdbuf()->connect(theAddy);
    mSocket->rdbuf()->recvtimeout((int)GetTimer().GetTimeout());
    mSocket->rdbuf()->sendtimeout((int)GetTimer().GetTimeout());
     */
  }
  catch (sockerr& ex)
    {
    //unable to establish connection, so break off.
    (void)ex;  //to avoid unreferenced variable warning on release
    gdcmErrorMacro("Unable to open connection with exception " << ex.what() <<
      " and " << ex.operation() << " on port: " << mInfo.GetCalledIPPort() <<
      std::endl);
    return false;
    }
  catch (std::exception& ex)
    {
    //unable to establish connection, so break off.
    (void)ex;  //to avoid unreferenced variable warning on release
    gdcmErrorMacro("Unable to open connection with exception " << ex.what() << std::endl);
    return false;
    }
  return true;
}

void ULConnection::StopProtocol()
{
  if (mEcho != NULL)
    {
    delete mEcho;
    mEcho = NULL;
    SetState(eSta1Idle);
    }
  else
    {
    //don't actually kill the connection, just kill the association.
    //this is just for a cstorescp initialized by a cmove
    SetState(eSta2Open);
    }
}

const PresentationContextRQ *ULConnection::GetPresentationContextRQByID(uint8_t id) const
{
  // one day ULConnection will actually use a AAssociateRQPDU as internal implementation
  // for now duplicate code from AAssociateRQPDU::GetPresentationContextFromAbstractSyntax
  std::vector<PresentationContextRQ>::const_iterator it = mPresentationContexts.begin();
  for( ; it != mPresentationContexts.end(); ++it)
    {
    if( it->GetPresentationContextID() == id )
      {
      return &*it;
      }
    }

  return NULL;
}

const PresentationContextAC *ULConnection::GetPresentationContextACByID(uint8_t id) const
{
  // one day ULConnection will actually use a AAssociateRQPDU as internal implementation
  // for now duplicate code from AAssociateRQPDU::GetPresentationContextFromAbstractSyntax
  std::vector<PresentationContextAC>::const_iterator it = mAcceptedPresentationContexts.begin();
  for( ; it != mAcceptedPresentationContexts.end(); ++it)
    {
    if( it->GetPresentationContextID() == id )
      {
      return &*it;
      }
    }

  return NULL;
}

uint8_t ULConnection::GetPresentationContextIDFromPresentationContext(PresentationContextRQ const & pc) const
{
  // one day ULConnection will actually use a AAssociateRQPDU as internal implementation
  // for now duplicate code from AAssociateRQPDU::GetPresentationContextIDFromAbstractSyntax
  uint8_t ret = 0;

  std::vector<PresentationContextRQ>::const_iterator it =
    std::find( mPresentationContexts.begin(), mPresentationContexts.end(), pc );
  if( it != mPresentationContexts.end() )
    {
    ret = it->GetPresentationContextID();
    }

  assert( ret );
  return ret;
}

void ULConnection::SetCStoreTransferSyntax( TransferSyntaxSub const & ts )
{
  cstorets = ts;
}

TransferSyntaxSub const & ULConnection::GetCStoreTransferSyntax( ) const
{
  TransferSyntaxSub ts1;
  ts1.SetNameFromUID( UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM );

  TransferSyntaxSub ts2;
  ts2.SetNameFromUID( UIDs::ExplicitVRLittleEndian );

  assert( strcmp(cstorets.GetName(), ts1.GetName()) == 0
       || strcmp(cstorets.GetName(), ts2.GetName()) == 0
  );

  return cstorets;
}

} // end namespace network
} // end namespace gdcm
