/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmServiceClassUser.h"
#include "gdcmDataEvent.h"
#include "gdcmSimpleSubjectWatcher.h"
#include "gdcmPresentationContextGenerator.h"
#include "gdcmAttribute.h"
#include "gdcmCompositeNetworkFunctions.h"
#include "gdcmTransferSyntax.h"
#include "gdcmWriter.h"
#include "gdcmReader.h"
#include "gdcmUIDGenerator.h"

#include "gdcmTesting.h"

/*
 * This test actually fails on D. Clunie PACS, but succeed on DicomObject (www.dicomserver.co.uk)...
 *
 * $ findscu 184.73.255.26 11112 --call AWSPIXELMEDPUB  -P -k 8,52=IMAGE -k 8,16="1.*" -k 20,d
 * $ gdcmscu --find 184.73.255.26 11112 --call AWSPIXELMEDPUB --patientroot --image -k 8,16="1.*" -k 20,d
 */

int TestServiceClassUser3(int argc, char *argv[])
{
  if( argc < 5 )
    {
    std::cerr << argv[0] << " aetitle call portno moveReturnPort remote" << std::endl;
    return 1;
    }
  std::string aetitle = argv[1]; // the ae title of this computer
  std::string call = argv[2]; // the ae title of the server
  uint16_t portno = (uint16_t)atoi(argv[3]); // the port of the server
  uint16_t moveReturnPort = (uint16_t)atoi(argv[4]); // the port over which return cstore scps are done for cmove
  std::string remote = argv[5]; //the ip address of the remote server
  (void)moveReturnPort;

  gdcm::SmartPointer<gdcm::ServiceClassUser> scup = new gdcm::ServiceClassUser;
  gdcm::ServiceClassUser &scu = *scup;
  gdcm::SimpleSubjectWatcher w( &scu, "TestServiceClassUser3" );

  std::ostringstream error_log;
  gdcm::Trace::SetErrorStream( error_log );

  scu.SetHostname( remote.c_str() );
  scu.SetPort( portno );
  scu.SetTimeout( 1000 );
  scu.SetCalledAETitle( call.c_str() );
  scu.SetAETitle( aetitle.c_str() );

  if( !scu.InitializeConnection() )
    {
    return 1;
    }

  gdcm::PresentationContextGenerator generator;
  if( !generator.GenerateFromUID( gdcm::UIDs::VerificationSOPClass ) )
    {
    return 1;
    }

  // make sure to fail if no pres contexts:
  if( scu.StartAssociation() )
    {
    return 1;
    }

  scu.SetPresentationContexts( generator.GetPresentationContexts() );

  if( !scu.StartAssociation() )
    {
    return 1;
    }

  // C-ECHO
  if( !scu.SendEcho() )
    {
    return 1;
    }

  if( !scu.StopAssociation() )
    {
    return 1;
    }

  // customize the find query
  gdcm::DataSet findds;
  gdcm::Attribute<0x8,0x16> sop ={"1.*"};
  findds.Insert( sop.GetAsDataElement() );
  gdcm::Attribute<0x20,0xd> uid;
  findds.Insert( uid.GetAsDataElement() );

  gdcm::SmartPointer<gdcm::BaseRootQuery> findquery =
    gdcm::CompositeNetworkFunctions::ConstructQuery(
      gdcm::ePatientRootType, gdcm::eImage, findds);

  // make sure the query is valid
  findquery->Print( std::cout );
  if (!findquery->ValidateQuery(false))
    {
    return 1;
    }

  // Generate the PresentationContext array from the query UID:
  if( !generator.GenerateFromUID( findquery->GetAbstractSyntaxUID() ) )
    {
    return 1;
    }

  scu.SetPresentationContexts( generator.GetPresentationContexts() );

  if( !scu.StartAssociation() )
    {
    return 1;
    }

  // C-FIND
  std::vector<gdcm::DataSet> datasets;
  // This is an error if the previous query succeed:
  if( scu.SendFind(findquery, datasets) )
    {
    std::cerr << "Could SendFind, this is not possible !" << std::endl;
    std::cerr << "Error log is:" << std::endl;
    std::cerr << error_log.str() << std::endl;
    return 1;
    }

  if( !scu.StopAssociation() )
    {
    return 1;
    }

  // scu dstor will close the connection (!= association)

  return 0;
}
