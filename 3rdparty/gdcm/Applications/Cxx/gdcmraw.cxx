/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 * gdcmraw - ACR/NEMA DICOM PS3 ... DICOM PS3 - DICOM image to raw file
 * Synopsis:
 * gdcmraw [ -t | --tag Tag# (default: 07fe,0010) ] -i inputfile
 * Description:
 * gdcmraw
 * reads the named dicom or acr-nema input file and copies the raw image
 * pixel data to a raw binary file without a header of any kind.
 * The byte order, packing or encapsulation of the raw result is dependent
 * only on the encoding of the input file and cannot be changed.
*/

#include "gdcmReader.h"
#include "gdcmImageReader.h"
#include "gdcmImage.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmDataSet.h"
#include "gdcmTag.h"
#include "gdcmByteValue.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmFragment.h"
#include "gdcmFilename.h"
#include "gdcmFilenameGenerator.h"
#include "gdcmVersion.h"

#include <string>
#include <iostream>

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
#include <string.h>

static void PrintVersion()
{
  std::cout << "gdcmraw: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmraw [OPTION]... FILE..." << std::endl;
  std::cout << "Extract Data Element Value Field" << std::endl;
  std::cout << "Parameter (required):" << std::endl;
  std::cout << "  -i --input       DICOM filename" << std::endl;
  std::cout << "  -o --output      DICOM filename" << std::endl;
  std::cout << "  -t --tag         Specify tag to extract value from." << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -S --split-frags Split fragments into multiple files." << std::endl;
  std::cout << "  -p --pattern     Specify trailing file pattern (see split-frags)." << std::endl;
  std::cout << "  -P --pixel-data  Pixel Data trailing 0." << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose   more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning   print warning info." << std::endl;
  std::cout << "  -D --debug     print debug info." << std::endl;
  std::cout << "  -E --error     print error info." << std::endl;
  std::cout << "  -h --help      print help." << std::endl;
  std::cout << "  -v --version   print version." << std::endl;
}

int main(int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;

  gdcm::Tag rawTag(0x7fe0, 0x0010); // Default to Pixel Data
  std::string filename;
  std::string outfilename;
  std::string pattern;
  int splitfrags = 0;
  int pixeldata = 0;
  int verbose = 0;
  int warning = 0;
  int debug = 0;
  int error = 0;
  int help = 0;
  int version = 0;
  while (1) {
    //int this_option_optind = optind ? optind : 1;
    int option_index = 0;
    static struct option long_options[] = {
        {"input", 1, 0, 0},                 // i
        {"output", 1, 0, 0},                // o
        {"tag", 1, 0, 0},                   // t
        {"split-frags", 0, &splitfrags, 1}, // f
/*
 * pixel-data flag is important for image like DermaColorLossLess.dcm since the bytevalue is
 * 63532, because of the DICOM \0 padding, but we would rather have the image buffer instead
 * which is simply one byte shorter, so add a special flag that simply mimic what TestImageReader
 * would expect
 */
        {"pixel-data", 0, &pixeldata, 1},   // P
        {"pattern", 1, 0, 0},               // p

        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},

        {0, 0, 0, 0}
    };

    c = getopt_long (argc, argv, "i:o:t:Sp:PVWDEhv",
      long_options, &option_index);
    if (c == -1)
      {
      break;
      }

    switch (c)
      {
    case 0:
        {
        const char *s = long_options[option_index].name; (void)s;
        //printf ("option %s", s);
        if (optarg)
          {
          if( option_index == 0 ) /* input */
            {
            assert( strcmp(s, "input") == 0 );
            assert( filename.empty() );
            filename = optarg;
            }
          else if( option_index == 2 ) /* tag */
            {
            assert( strcmp(s, "tag") == 0 );
            rawTag.ReadFromCommaSeparatedString(optarg);
            }
          else if( option_index == 5 ) /* pattern */
            {
            assert( strcmp(s, "pattern") == 0 );
            assert( pattern.empty() );
            pattern = optarg;
            }
          //printf (" with arg %s", optarg);
          }
        //printf ("\n");
        }
      break;

    case 'i':
      //printf ("option i with value '%s'\n", optarg);
      assert( filename.empty() );
      filename = optarg;
      break;

    case 'o':
      //printf ("option o with value '%s'\n", optarg);
      assert( outfilename.empty() );
      outfilename = optarg;
      break;

    case 'P':
      pixeldata = 1;
      break;

    case 'S':
      splitfrags = 1;
      break;

    case 'p':
      assert( pattern.empty() );
      pattern = optarg;
      break;

    case 't':
      //printf ("option t with value '%s'\n", optarg);
      rawTag.ReadFromCommaSeparatedString(optarg);
      //std::cerr << rawTag << std::endl;
      break;

    case 'V':
      verbose = 1;
      break;

    case 'W':
      warning = 1;
      break;

    case 'D':
      debug = 1;
      break;

    case 'E':
      error = 1;
      break;

    case 'h':
      help = 1;
      break;

    case 'v':
      version = 1;
      break;

    case '?':
      break;

    default:
      printf ("?? getopt returned character code 0%o ??\n", c);
      }
  }

  if (optind < argc)
    {
    std::vector<std::string> files;
    while (optind < argc)
      {
      //printf ("%s\n", argv[optind++]);
      files.push_back( argv[optind++] );
      }
    //printf ("\n");
    if( files.size() == 2
      && filename.empty()
      && outfilename.empty()
    )
      {
      filename = files[0];
      outfilename = files[1];
      }
    else
      {
      PrintHelp();
      return 1;
      }
    }

  if( version )
    {
    //std::cout << "version" << std::endl;
    PrintVersion();
    return 0;
    }

  if( help )
    {
    //std::cout << "help" << std::endl;
    PrintHelp();
    return 0;
    }

  if( filename.empty() )
    {
    //std::cerr << "Need input file (-i)\n";
    PrintHelp();
    return 1;
    }

  // Debug is a little too verbose
  gdcm::Trace::SetDebug( debug != 0);
  gdcm::Trace::SetWarning( warning != 0);
  gdcm::Trace::SetError( error != 0);
  // when verbose is true, make sure warning+error are turned on:
  if( verbose )
    {
    gdcm::Trace::SetWarning( verbose != 0);
    gdcm::Trace::SetError( verbose!= 0);
    }

  // else
  //std::cout << "Filename: " << filename << std::endl;

  // very special option, handle it first:
  if( pixeldata )
    {
    if( rawTag != gdcm::Tag(0x7fe0,0x0010) )
      {
      return 1;
      }
    gdcm::ImageReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "Failed to read: " << filename << std::endl;
      return 1;
      }
    const gdcm::Image& image = reader.GetImage();
    unsigned long len = image.GetBufferLength();
    char * buf = new char[len];
    image.GetBuffer( buf );

    std::ofstream output(outfilename.c_str(), std::ios::binary);
    output.write( buf, len );

    delete[] buf;
    return 0;
    }
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  //const gdcm::FileMetaInformation &h = reader.GetFile().GetHeader();
  //std::cout << h << std::endl;

  const gdcm::DataSet &ds = reader.GetFile().GetDataSet();
  //std::cout << ds << std::endl;

  if( !ds.FindDataElement( rawTag ) )
    {
    std::cerr << "Cannot find Tag: " << rawTag << std::endl;
    return 1;
    }

  if( outfilename.empty() )
    {
    std::cerr << "Need output file (-o)\n";
    return 1;
    }
  gdcm::Filename fn1(filename.c_str()), fn2(outfilename.c_str());
  if( fn1.IsIdentical(fn2) )
    {
    std::cerr << "Output is Input\n";
    return 1;
    }

  const gdcm::DataElement& pdde = ds.GetDataElement( rawTag );
  const gdcm::ByteValue *bv = pdde.GetByteValue();
  const gdcm::SequenceOfFragments *sf = pdde.GetSequenceOfFragments();
  if( bv )
    {
    std::ofstream output(outfilename.c_str(), std::ios::binary);
    bv->WriteBuffer(output);
    }
  else if( sf )
    {
    if( splitfrags )
      {
      size_t nfrags = sf->GetNumberOfFragments();
      gdcm::FilenameGenerator fg;
      fg.SetNumberOfFilenames( nfrags );
      fg.SetPrefix( outfilename.c_str() );
      fg.SetPattern( pattern.c_str() );
      if(!fg.Generate())
        {
        std::cerr << "Could not generate" << std::endl;
        return 1;
        }
      for(unsigned int i = 0; i < nfrags; ++i)
        {
        const gdcm::Fragment& frag = sf->GetFragment(i);
        const gdcm::ByteValue *fragbv = frag.GetByteValue();
        const char *outfilenamei = fg.GetFilename(i);
        std::ofstream outputi(outfilenamei, std::ios::binary);
        fragbv->WriteBuffer(outputi);
        }
      }
    else
      {
      std::ofstream output(outfilename.c_str(), std::ios::binary);
      sf->WriteBuffer(output);
      }
    }
  else
    {
    const gdcm::Value &value = pdde.GetValue();
    const gdcm::Value * v = &value;
    const gdcm::SequenceOfItems *sqi = dynamic_cast<const gdcm::SequenceOfItems*>( v );
    if( sqi )
      {
      //std::ofstream output(outfilename.c_str(), std::ios::binary);
      //sqi->Write<gdcm::ImplicitDataElement, gdcm::SwapperNoOp>(output);
      size_t nfrags = sqi->GetNumberOfItems();
      gdcm::FilenameGenerator fg;
      fg.SetNumberOfFilenames( nfrags );
      fg.SetPrefix( outfilename.c_str() );
      fg.SetPattern( pattern.c_str() );
      if(!fg.Generate())
        {
        std::cerr << "Could not generate" << std::endl;
        return 1;
        }
      for(unsigned int i = 0; i < nfrags; ++i)
        {
        const gdcm::Item& frag = sqi->GetItem(i+1);
        const gdcm::DataSet &subds = frag.GetNestedDataSet();
        const char *outfilenamei = fg.GetFilename(i);
        std::ofstream outputi(outfilenamei, std::ios::binary);
        // Let's imagine we found an undefined length Pixel Data attribute in
        // this sequence. Let's pick ExplicitDataElement for writing out then
        subds.Write<gdcm::ExplicitDataElement, gdcm::SwapperNoOp>(outputi);
        }

      }
    else
      {
      std::cerr << "Unhandled" << std::endl;
      return 1;
      }
    }

  return 0;
}
