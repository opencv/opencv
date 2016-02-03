/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAnonymizer.h"
#include "gdcmSimpleSubjectWatcher.h"
#include "gdcmUIDGenerator.h"
#include "gdcmFilename.h"
#include "gdcmTesting.h"
#include "gdcmCryptographicMessageSyntax.h"
#include "gdcmSmartPointer.h"
#include "gdcmReader.h"
#include "gdcmWriter.h"
#include "gdcmGlobal.h"
#include "gdcmFileDerivation.h"
#include "gdcmSystem.h"

#include "gdcmCryptoFactory.h"
#include <memory> // std::auto_ptr

int TestAnonymizer3(int , char *[])
{
  using namespace gdcm;
  gdcm::Global& g = gdcm::Global::GetInstance();
  if( !g.LoadResourcesFiles() )
    {
    return 1;
    }
  const char *directory = gdcm::Testing::GetDataRoot();
  std::string sfilename = std::string(directory) + "/012345.002.050.dcm";
  const char *filename = sfilename.c_str();

  std::string certpath = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/certificate.pem" );
  std::string keypath = gdcm::Filename::Join(gdcm::Testing::GetSourceDirectory(), "/Testing/Source/Data/privatekey.pem" );

  // Create directory first:
  const char subdir[] = "TestAnonymizer3";
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );
  std::string outfilenamelossy = Testing::GetTempFilename( "012345.002.050.lossy.dcm" , subdir );
  std::string outfilenamelossy2 = Testing::GetTempFilename( "012345.002.050.lossy.anon.dcm" , subdir );

// Derive
{
  gdcm::Reader reader;
  reader.SetFileName( filename );
  if( !reader.Read() )
    {
    std::cerr << "Could not read: " << filename << std::endl;
    return 1;
    }

  File& file = reader.GetFile();
  DataSet &ds = file.GetDataSet();

  DataElement instanceuid = ds.GetDataElement( Tag(0x8,0x18) );
  UIDGenerator uid;
  const char *s = uid.Generate();
  instanceuid.SetByteValue( s, (uint32_t)strlen(s) );
  ds.Replace( instanceuid );

  FileDerivation fd;
  fd.SetFile( file );
  // FIXME hardcoded:
  fd.AddReference( "1.2.840.10008.5.1.4.1.1.4",
    "1.2.840.113619.2.5.1762386977.1328.985934491.693" );

  // CID 7202 Source Image Purposes of Reference
  // {"DCM",121320,"Uncompressed predecessor"},
  fd.SetPurposeOfReferenceCodeSequenceCodeValue( 121320 );

  // CID 7203 Image Derivation
  // { "DCM",113040,"Lossy Compression" },
  fd.SetDerivationCodeSequenceCodeValue( 113040 );
  fd.SetDerivationDescription( "lossy conversion" );
  if( !fd.Derive() )
    {
    std::cerr << "Sorry could not derive using input info" << std::endl;
    return 1;
    }

  gdcm::Writer writer;
  writer.SetFileName( outfilenamelossy.c_str() );
  writer.SetFile( fd.GetFile() );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilenamelossy << std::endl;
    return 1;
    }
}

// Encrypt
{
  gdcm::CryptoFactory* cryptoFactory = gdcm::CryptoFactory::GetFactoryInstance();
  if (cryptoFactory == NULL)
    {
    std::cerr << "Crypto library not available" << std::endl;
    return 1;
    }
  std::auto_ptr<gdcm::CryptographicMessageSyntax> cms_ptr(cryptoFactory->CreateCMSProvider());
  gdcm::CryptographicMessageSyntax& cms = *cms_ptr;
  if( !cms.ParseCertificateFile( certpath.c_str() ) )
    {
    return 1;
    }

  gdcm::SmartPointer<gdcm::Anonymizer> ano = new gdcm::Anonymizer;
  ano->SetCryptographicMessageSyntax( &cms );

{
  // original file
  gdcm::Reader reader;
  reader.SetFileName( filename );
  if( !reader.Read() )
    {
    std::cerr << "Could not read: " << filename << std::endl;
    return 1;
    }

  // order of operation is important
  ano->SetFile( reader.GetFile() );
  if( !ano->BasicApplicationLevelConfidentialityProfile() )
    {
    return 1;
    }

  gdcm::Writer writer;
  writer.SetFileName( outfilename.c_str() );
  writer.SetFile( reader.GetFile() );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }
}

{
  // derived file
  gdcm::Reader reader;
  reader.SetFileName( outfilenamelossy.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Could not read: " << outfilenamelossy << std::endl;
    return 1;
    }

  // order of operation is important
  ano->SetFile( reader.GetFile() );
  if( !ano->BasicApplicationLevelConfidentialityProfile() )
    {
    return 1;
    }

  gdcm::Writer writer;
  writer.SetFileName( outfilenamelossy2.c_str() );
  writer.SetFile( reader.GetFile() );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilenamelossy2 << std::endl;
    return 1;
    }
}
}

// Make sure UID is consistant:
{
  std::string sopinstanceuid_str1;
{
  // original anonymized file
  gdcm::Reader reader;
  reader.SetFileName( outfilename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Could not read: " << filename << std::endl;
    return 1;
    }
  File &file = reader.GetFile();
  DataSet &ds = file.GetDataSet();

  if( !ds.FindDataElement( Tag(0x0008,0x0018) )
    || ds.GetDataElement( Tag(0x0008,0x0018) ).IsEmpty() )
    {
    return 1;
    }
  const DataElement &sopinstanceuid = ds.GetDataElement( Tag(0x0008,0x0018) );
  sopinstanceuid_str1 = std::string( sopinstanceuid.GetByteValue()->GetPointer(), sopinstanceuid.GetByteValue()->GetLength() );
}

  std::string sopinstanceuid_str2;
  std::string refsopinstanceuid_str2;
{
  // derived anonymized file
  gdcm::Reader reader;
  reader.SetFileName( outfilenamelossy2.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Could not read: " << outfilenamelossy << std::endl;
    return 1;
    }
  File &file = reader.GetFile();
  DataSet &ds = file.GetDataSet();

  if( !ds.FindDataElement( Tag(0x0008,0x0018) )
    || ds.GetDataElement( Tag(0x0008,0x0018) ).IsEmpty() )
    {
    return 1;
    }
  const DataElement &sopinstanceuid = ds.GetDataElement( Tag(0x0008,0x0018) );
  sopinstanceuid_str2 = std::string( sopinstanceuid.GetByteValue()->GetPointer(), sopinstanceuid.GetByteValue()->GetLength() );

  // Source Image Sequence
  if( !ds.FindDataElement( Tag(0x0008,0x2112) )
    || ds.GetDataElement( Tag(0x0008,0x2112) ).IsEmpty() )
    {
    return 1;
    }

  const DataElement &sourceimagesq = ds.GetDataElement( Tag(0x0008,0x2112) );
  SmartPointer<SequenceOfItems> sq = sourceimagesq.GetValueAsSQ();
  gdcm::SequenceOfItems::SizeType n = sq->GetNumberOfItems();
  if( n != 1 ) return 1;
  Item &item = sq->GetItem( 1 );
  DataSet &nested = item.GetNestedDataSet();

  if( !nested.FindDataElement( Tag(0x0008,0x1155) )
    || nested.GetDataElement( Tag(0x0008,0x1155) ).IsEmpty() )
    {
    return 1;
    }
  const DataElement &refsopinstanceuid = nested.GetDataElement(
    Tag(0x0008,0x1155) );
  refsopinstanceuid_str2 = std::string( refsopinstanceuid.GetByteValue()->GetPointer(), refsopinstanceuid.GetByteValue()->GetLength() );

}
  std::cout << sopinstanceuid_str1 << std::endl;
  std::cout << sopinstanceuid_str2 << std::endl;
  std::cout << refsopinstanceuid_str2 << std::endl;

if( sopinstanceuid_str1 == sopinstanceuid_str2 )
{
  return 1;
}
if( sopinstanceuid_str1 != refsopinstanceuid_str2 )
{
  return 1;
}
}

  return 0;
}
