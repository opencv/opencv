/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSERVICECLASSUSER_H
#define GDCMSERVICECLASSUSER_H

#include "gdcmSubject.h"

#include "gdcmPresentationContext.h"
#include "gdcmFile.h"

#include "gdcmNetworkStateID.h" // EStateID

namespace gdcm
{
class ServiceClassUserInternals;
class BaseRootQuery;
namespace network{
class ULEvent;
class ULConnection;
class ULConnectionCallback;
}
/**
 * \brief ServiceClassUser
 */
class GDCM_EXPORT ServiceClassUser : public Subject
{
public:
  /// Construct a SCU with default:
  /// - hostname = localhost
  /// - port = 104
  ServiceClassUser();
  ~ServiceClassUser();

  /// Set the name of the called hostname (hostname or IP address)
  void SetHostname( const char *hostname );

  /// Set port of remote host (called application)
  void SetPort( uint16_t port );

  /// Set the port for any incoming C-STORE-SCP operation (typically in a return of C-MOVE)
  void SetPortSCP( uint16_t portscp );

  /// set calling ae title
  void SetAETitle(const char *aetitle);
  const char *GetAETitle() const;

  /// set called ae title
  void SetCalledAETitle(const char *aetitle);
  const char *GetCalledAETitle() const;

  /// set/get Timeout
  void SetTimeout(double t);
  double GetTimeout() const;

  /// Will try to connect
  /// This will setup the actual timeout used during the whole connection time. Need to call
  /// SetTimeout first
  bool InitializeConnection();

  /// Set the Presentation Context used for the Association
  void SetPresentationContexts(std::vector<PresentationContext> const & pcs);

  /// Return if the passed in presentation was accepted during association negotiation.
  bool IsPresentationContextAccepted(const PresentationContext& pc) const;

  /// Start the association. Need to call SetPresentationContexts before
  bool StartAssociation();

  /// Stop the running association
  bool StopAssociation();

  /// C-ECHO
  bool SendEcho();

  /// Execute a C-STORE on file on disk, named filename
  bool SendStore(const char *filename);
  /// Execute a C-STORE on a File, the transfer syntax used for the query is based on the
  /// file.
  bool SendStore(File const &file);
  /// Execute a C-STORE on a DataSet, the transfer syntax used will be Implicit
  bool SendStore(DataSet const &ds);

  /// C-FIND a query, return result are in retDatasets
  bool SendFind(const BaseRootQuery* query, std::vector<DataSet> &retDatasets);

  /// Execute a C-MOVE, based on query, return files are written in outputdir
  bool SendMove(const BaseRootQuery* query, const char *outputdir);
  /// Execute a C-MOVE, based on query, returned dataset are Implicit
  bool SendMove(const BaseRootQuery* query, std::vector<DataSet> &retDatasets);
  /// Execute a C-MOVE, based on query, returned Files are stored in vector
  bool SendMove(const BaseRootQuery* query, std::vector<File> &retFile);

  /// for wrapped language: instanciate a reference counted object
  static SmartPointer<ServiceClassUser> New() { return new ServiceClassUser; }

private:
  network::EStateID RunEventLoop(network::ULEvent& inEvent,
    network::ULConnection* inWhichConnection,
    network::ULConnectionCallback* inCallback, const bool& startWaiting);
  network::EStateID RunMoveEventLoop(network::ULEvent& inEvent,
    network::ULConnectionCallback* inCallback);

private:
  ServiceClassUser(const ServiceClassUser&);
  void operator=(const ServiceClassUser &);

private:
  ServiceClassUserInternals *Internals;
};

} // end namespace gdcm

#endif // GDCMSERVICECLASSUSER_H
