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

#include "gdcmTesting.h"

/*
 * This small example show how the association pipeline works.
 * the start association is an important step as it will define all the object
 * to be send in subsequence C-operation (c-echo, c-store, c-find, c-move).
 * In this example we will demonstrate how we can send a JPEG-Lossless object
 * and then further on, a non-jpeg encapsulated file
 *
 * The test also uses the Subject/Observer API for progress report.
 */

int TestServiceClassUser1(int argc, char *argv[])
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

  gdcm::SmartPointer<gdcm::ServiceClassUser> scup = new gdcm::ServiceClassUser;
  gdcm::ServiceClassUser &scu = *scup;
  gdcm::SimpleSubjectWatcher w( &scu, "TestServiceClassUser1" );

  scu.SetHostname( remote.c_str() );
  scu.SetPort( portno );
  scu.SetTimeout( 1000 );
  scu.SetCalledAETitle( call.c_str() );
  scu.SetAETitle( aetitle.c_str() );

  std::ostringstream error_log;
  gdcm::Trace::SetErrorStream( error_log );

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
    std::cerr << "Could not StartAssociation" << std::endl;
    std::cerr << "Error log is:" << std::endl;
    std::cerr << error_log.str() << std::endl;
    return 1;
    }

  // C-ECHO
  if( !scu.SendEcho() )
    {
    std::cerr << "Could not Echo" << std::endl;
    std::cerr << "Error log is:" << std::endl;
    std::cerr << error_log.str() << std::endl;
    return 1;
    }

  if( !scu.StopAssociation() )
    {
    return 1;
    }

  gdcm::Directory::FilenamesType filenames;
  const char *directory = gdcm::Testing::GetDataRoot();
  // DEBUG:
  // storescu -R -xs --call GDCM_STORE macminig4 11112 gdcmData/012345.002.050.dcm
  std::string filename = std::string(directory) + "/012345.002.050.dcm";
  filenames.push_back( filename );

  if( !generator.GenerateFromFilenames(filenames) )
    {
    return 1;
    }

  scu.SetPresentationContexts( generator.GetPresentationContexts() );

  if( !scu.StartAssociation() )
    {
    return 1;
    }

  // C-STORE MRImageStorage/JPEGLossless
  if( !scu.SendStore( filename.c_str() ) )
    {
    std::cerr << "Could not C-Store" << std::endl;
    std::cerr << "Error log is:" << std::endl;
    std::cerr << error_log.str() << std::endl;
    return 1;
    }

  if( !scu.StopAssociation() )
    {
    return 1;
    }

  //filename = std::string(directory) + "/MR-MONO2-12-an2.acr";
  filename = std::string(directory) + "/MR_Spectroscopy_SIEMENS_OF.dcm";
  filenames.clear();
  filenames.push_back( filename );

  if( !generator.GenerateFromFilenames(filenames) )
    {
    return 1;
    }

  scu.SetPresentationContexts( generator.GetPresentationContexts() );

  if( !scu.StartAssociation() )
    {
    return 1;
    }

  // C-STORE MRImageStorage/LittleEndianImplicit
  if( !scu.SendStore( filename.c_str() ) )
    {
    std::cerr << "Could not SendStore" << std::endl;
    std::cerr << "Error log is:" << std::endl;
    std::cerr << error_log.str() << std::endl;
    return 1;
    }

  if( !scu.StopAssociation() )
    {
    return 1;
    }

  // customize the find query
  gdcm::DataSet findds;
  gdcm::Attribute<0x10,0x10> pn1 ={"ABCDEFGH^IJKLM"};
  findds.Insert( pn1.GetAsDataElement() );
  gdcm::Attribute<0x10,0x20> pid;
  findds.Insert( pid.GetAsDataElement() );

  gdcm::SmartPointer<gdcm::BaseRootQuery> findquery =
    gdcm::CompositeNetworkFunctions::ConstructQuery(
      gdcm::ePatientRootType, gdcm::ePatient, findds);

  // make sure the query is valid
  if (!findquery->ValidateQuery())
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
  if( !scu.SendFind(findquery, datasets) )
    {
    return 1;
    }

  // Need to make sure we have one dataset
  if( datasets.size() != 1 )
    {
    std::cerr << "size: " << datasets.size() << std::endl;
    return 1;
    }
  datasets[0].Print( std::cout );

  // C-find the second patient
  gdcm::Attribute<0x10,0x10> pn2 ={"XXXXXXXXXXX"};
  findquery->GetQueryDataSet().Replace( pn2.GetAsDataElement() );
  if( !scu.SendFind(findquery, datasets) )
    {
    return 1;
    }

  if( datasets.size() != 2 )
    {
    std::cerr << "size: " << datasets.size() << std::endl;
    return 1;
    }
  std::cout << std::endl;

  datasets[1].Print( std::cout );

  if( !scu.StopAssociation() )
    {
    return 1;
    }

  // C-MOVE
  // customize the move query
  gdcm::DataSet moveds;
  // use results from the c-find to construct the c-move query:
  moveds.Insert( datasets[0].GetDataElement( pid.GetTag() ) );

  gdcm::SmartPointer<gdcm::BaseRootQuery> movequery =
    gdcm::CompositeNetworkFunctions::ConstructQuery(
      gdcm::ePatientRootType, gdcm::ePatient, moveds, true);
  // make sure the query is valid
  if (!movequery->ValidateQuery())
    {
    return 1;
    }

  //generator.SetDefaultTransferSyntax( gdcm::TransferSyntax::JPEGLosslessProcess14_1 );

  // Generate the PresentationContext array from the query UID:
  if( !generator.GenerateFromUID( movequery->GetAbstractSyntaxUID() ) )
    {
    return 1;
    }

  scu.SetPresentationContexts( generator.GetPresentationContexts() );

  scu.SetPortSCP( moveReturnPort );

  if( !scu.StartAssociation() )
    {
    return 1;
    }

  // C-MOVE
  std::vector<gdcm::DataSet> data;
  if( !scu.SendMove(movequery, data) )
    {
    return 1;
    }

  if( data.size() != 1 )
    {
    std::cerr << "data size: " << data.size() << std::endl;
    return 1;
    }

  // SendMove + dataset is implicit by default:
  gdcm::Writer writer;
  writer.GetFile().GetHeader().SetDataSetTransferSyntax(
    gdcm::TransferSyntax::ImplicitVRLittleEndian );
  writer.GetFile().SetDataSet( data[0] );
  const char *outfilename = "dummy.dcm";
  writer.SetFileName( outfilename );
  if( !writer.Write() )
    {
    return 1;
    }

  if( !gdcm::System::FileExists(outfilename) )
    {
    std::cerr << "FileExists: " << outfilename << std::endl;
    return 1;
    }

  char digest_str[33];
  if( !gdcm::Testing::ComputeFileMD5(outfilename, digest_str) )
    {
    return 1;
    }

  if( strcmp( digest_str, "ae1f9a1bfc617f73ae8f72f81777dc03") != 0 )
    {
    std::cerr << "md5: " << digest_str << std::endl;
    return 1;
    }

  // TODO: testing of CMove + JPEG Lossless is a *lot* more difficult
  // since we have to assume some behavior on the remote side (SCP) which
  // we cannot query.

  if( !scu.StopAssociation() )
    {
    return 1;
    }

  // scu dstor will close the connection (!= association)

  return 0;
}
