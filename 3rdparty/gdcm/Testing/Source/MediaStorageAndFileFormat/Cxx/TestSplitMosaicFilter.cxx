/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSplitMosaicFilter.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmImageReader.h"

static bool reorganize_mosaic_invert(unsigned short *input,
  const unsigned int *inputdims, unsigned int square,
  const unsigned int *outputdims, const unsigned short *output )
{
  for(unsigned x = 0; x < outputdims[0]; ++x)
    {
    for(unsigned y = 0; y < outputdims[1]; ++y)
      {
      for(unsigned z = 0; z < outputdims[2]; ++z)
        {
        size_t outputidx = x + y*outputdims[0] + z*outputdims[0]*outputdims[1];
        size_t inputidx = (x + (z%square)*outputdims[0]) +
          (y + (z/square)*outputdims[0])*inputdims[0];
        input[ inputidx ] = output[ outputidx ];
        }
      }
    }
  return true;
}

int TestSplitMosaicFilter(int argc, char *argv[])
{
  std::string filename;
  if( argc == 2 )
    {
    filename = argv[1];
    }
  else
    {
    const char *extradataroot = gdcm::Testing::GetDataExtraRoot();
    if( !extradataroot )
      {
      return 1;
      }
    if( !gdcm::System::FileIsDirectory(extradataroot) )
      {
      std::cerr << "No such directory: " << extradataroot <<  std::endl;
      return 1;
      }

    filename = extradataroot;
    filename += "/gdcmSampleData/images_of_interest/MR-sonata-3D-as-Tile.dcm";
    }
  gdcm::SplitMosaicFilter s;

  // std::cout << filename << std::endl;
  if( !gdcm::System::FileExists(filename.c_str()) )
    {
    return 1;
    }

  gdcm::ImageReader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "could not read: " << filename << std::endl;
    return 1;
    }
  const gdcm::Image &image = reader.GetImage();
  unsigned int inputdims[3] = { 0, 0, 1 };
  const unsigned int *dims = image.GetDimensions();
  inputdims[0] = dims[0];
  inputdims[1] = dims[1];

  gdcm::SplitMosaicFilter filter;
  filter.SetImage( reader.GetImage() );
  filter.SetFile( reader.GetFile() );
  bool b = filter.Split();
  if( !b )
    {
    std::cerr << "Could not split << " << filename << std::endl;
    return 1;
    }

//  const gdcm::Image &image = filter.GetImage();

  unsigned long l = image.GetBufferLength();
  std::vector<char> buf;
  buf.resize(l);
  if( !image.GetBuffer( &buf[0] ) )
    {
    std::cerr << "Could not GetBuffer << " << filename << std::endl;
    return 1;
    }

  std::vector<char> outbuf;
  unsigned long ll = inputdims[0] * inputdims[1] * sizeof( unsigned short );
  outbuf.resize(ll);

  const unsigned int *mos_dims = image.GetDimensions();
  unsigned int div = (unsigned int )ceil(sqrt( (double)mos_dims[2]) );

  reorganize_mosaic_invert((unsigned short *)&outbuf[0],  inputdims,
    div, mos_dims, (const unsigned short*)&buf[0] );

#if 0
  std::ofstream o( "/tmp/debug", std::ios::binary );
  o.write( &outbuf[0], ll );
  o.close();
#endif

  char digest[33];
  gdcm::Testing::ComputeMD5(&outbuf[0], ll, digest);

  // $ gdcminfo --md5sum gdcmSampleData/images_of_interest/MR-sonata-3D-as-Tile.dcm
  if( strcmp(digest, "be96c01db8a0ec0753bd43f6a985345c" ) != 0 )
    {
    std::cerr << "Problem found: " << digest << std::endl;
    return 1;
    }

  return 0;
}
