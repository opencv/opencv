/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileStreamer.h"

#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmReader.h"
#include "gdcmFilename.h"
#include "gdcmImageRegionReader.h"
#include "gdcmImageHelper.h"

int TestFileStream4(const char *filename, bool verbose = false)
{
  using namespace gdcm;

  // Create directory first:
  const char subdir[] = "TestFileStreamer4";
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();
  // Special handling:
  bool checktemplate = false;
  if( strcmp(name, "DMCPACS_ExplicitImplicit_BogusIOP.dcm" ) == 0
    || strcmp(name, "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm") == 0
    || strcmp(name, "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm") == 0 
  )
    {
    checktemplate = true;
    }

  gdcm::ImageRegionReader irr;
  irr.SetFileName( filename );
  if( !irr.ReadInformation() )
    {
    //std::cerr << "not an image: " << filename << std::endl;
    return 0;
    }

  gdcm::File & file = irr.GetFile();
  std::vector<unsigned int> dims =
    gdcm::ImageHelper::GetDimensionsValue(file);
  PixelFormat pf = gdcm::ImageHelper::GetPixelFormatValue(file);
  int pixsize = pf.GetPixelSize();
  const size_t computedlen = dims[0] * dims[1] * dims[2] * pixsize;

  const FileMetaInformation &header = file.GetHeader();
  const TransferSyntax &ts = header.GetDataSetTransferSyntax();

  if( verbose )
    {
    std::cout << "Processing: " << filename << std::endl;
    std::cout << "Generating: " << outfilename << std::endl;
    }

  std::vector<char> vbuffer;
  assert( dims[0] );
  vbuffer.resize( dims[0] * pixsize );
  char *buffer = &vbuffer[0];
  const size_t len = vbuffer.size();

  gdcm::FileStreamer fs;
  fs.ReserveDataElement( computedlen );
  fs.SetTemplateFileName( filename );
  fs.CheckTemplateFileName( checktemplate );
  fs.SetOutputFileName( outfilename.c_str() );

  const gdcm::Tag pixeldata(0x7fe0,0x0010);

  bool b;
  b = fs.CheckDataElement( pixeldata ); // will be checking file size
  if( !b )
    {
    std::cerr << "Failed to CheckDataElement: " << outfilename << std::endl;
    return 1;
    }
  b = fs.StartDataElement( pixeldata );
  if( !b )
    {
    std::cerr << "Failed to StartDataElement: " << outfilename << std::endl;
    return 1;
    }
  for( unsigned int z = 0; z < dims[2]; ++z )
    for( unsigned int y = 0; y < dims[1]; ++y )
      {
      b = fs.AppendToDataElement( pixeldata, buffer, len );
      if( !b )
        {
        std::cerr << "Failed to AppendToDataElement: " << outfilename << std::endl;
        return 1;
        }
      }
  if( !fs.StopDataElement( pixeldata ) )
    {
    if( ts.IsEncapsulated() )
      {
      // Everything is under control
      return 0;
      }
    std::cerr << "Failed to StopDataElement: " << outfilename << std::endl;
    return 1;
    }

  // Read back and check:
  gdcm::Reader r;
  r.SetFileName( outfilename.c_str() );
  if( !r.Read() )
    {
    std::cerr << "Failed to read: " << outfilename << std::endl;
    return 1;
    }

  gdcm::File & f = r.GetFile();
  gdcm::DataSet & ds = f.GetDataSet();

  if( !ds.FindDataElement( pixeldata ) )
    {
    std::cerr << "No pixel data: " << outfilename << std::endl;
    return 1;
    }
  const gdcm::DataElement & de = ds.GetDataElement( pixeldata );
  const gdcm::ByteValue *bv = de.GetByteValue();
  if( bv->GetLength() != computedlen )
    {
    std::cerr << "Mismatch len: " << outfilename << " : " << bv->GetLength() <<
      " vs " << computedlen << std::endl;
    return 1;
    }
  const char *p = bv->GetPointer();
  const char *end = p + dims[0] * dims[1] * dims[2];
  int res = 0;
  for( ; p != end; ++p )
    {
    res += *p;
    }
  if( res )
    {
    std::cerr << "Mismatch: " << outfilename << std::endl;
    }

  return res;
}

int TestFileStreamer4(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestFileStream4(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestFileStream4( filename );
    ++i;
    }

  return r;
}
