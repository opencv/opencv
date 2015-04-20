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
 * a Scanner application
 * Usage:
 *
 * $ gdcmscanner -d /images/ -t 0020,000d -t 0020,000e
 *
 * Options:
 * -d : directory
 * -t : tag (can be specified multiple times)
 * -p : print
 * -r : recursive (enter subdir of main directory)
 *
 * TODO:
 * --bench...
 */

#include "gdcmScanner.h"
#include "gdcmTrace.h"
#include "gdcmVersion.h"
#include "gdcmSimpleSubjectWatcher.h"

#include <string>
#include <iostream>
#include <iterator>

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
#include <string.h>

static void PrintVersion()
{
  std::cout << "gdcmscanner: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmscanner [OPTION] -d directory -t tag(s)" << std::endl;
  std::cout << "Scan a directory containing DICOM files.\n";
  std::cout << "Parameter (required):" << std::endl;
  std::cout << "  -d --dir       DICOM directory" << std::endl;
  std::cout << "  -t --tag %d,%d DICOM tag(s) to look for" << std::endl;
  std::cout << "  -P --private-tag %d,%d,%s DICOM private tag(s) to look for" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -p --print      Print output." << std::endl;
  std::cout << "  -r --recursive  Recusively descend directory." << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose    more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning    print warning info." << std::endl;
  std::cout << "  -D --debug      print debug info." << std::endl;
  std::cout << "  -E --error      print error info." << std::endl;
  std::cout << "  -h --help       print help." << std::endl;
  std::cout << "  -v --version    print version." << std::endl;
}

int main(int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;

  bool print = false;
  bool recursive = false;
  std::string dirname;
  typedef std::vector<gdcm::Tag> VectorTags;
  typedef std::vector<gdcm::PrivateTag> VectorPrivateTags;
  VectorTags tags;
  VectorPrivateTags privatetags;
  gdcm::Tag tag;
  gdcm::PrivateTag privatetag;

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
        {"dir", 1, 0, 0},
        {"tag", 1, 0, 0},
        {"recursive", 1, 0, 0},
        {"print", 1, 0, 0},
        {"private-tag", 1, 0, 0},

// General options !
        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},

        {0, 0, 0, 0}
    };

    c = getopt_long (argc, argv, "d:t:rpP:VWDEhv",
      long_options, &option_index);
    if (c == -1)
      {
      break;
      }

    switch (c)
      {
    case 0:
        {
        //const char *s = long_options[option_index].name;
        //printf ("option %s", s);
        //if (optarg)
        //  {
        //  if( option_index == 0 ) /* input */
        //    {
        //    assert( strcmp(s, "input") == 0 );
        //    }
        //  printf (" with arg %s", optarg);
        //  }
        //printf ("\n");
        }
      break;

    case 'd':
      dirname = optarg;
      break;

    case 't':
      tag.ReadFromCommaSeparatedString(optarg);
      tags.push_back( tag );
      //std::cerr << optarg << std::endl;
      break;

    case 'P':
      privatetag.ReadFromCommaSeparatedString(optarg);
      privatetags.push_back( privatetag );
      //std::cerr << optarg << std::endl;
      break;

    case 'r':
      recursive = true;
      break;

    case 'p':
      print = true;
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
/*
    printf ("non-option ARGV-elements: ");
    while (optind < argc)
      {
      printf ("%s ", argv[optind++]);
      }
    printf ("\n");
*/
    PrintHelp();
    return 1;
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

  if( dirname.empty() )
    {
    //std::cerr << "Need dir (-d)\n";
    PrintHelp();
    return 1;
    }
  if( tags.empty() && privatetags.empty() )
    {
    //std::cerr << "Need tags (-t)\n";
    PrintHelp();
    return 1;
    }
  // Debug is a little too verbose
  gdcm::Trace::SetDebug( (debug  > 0 ? true : false));
  gdcm::Trace::SetWarning(  (warning  > 0 ? true : false));
  gdcm::Trace::SetError(  (error  > 0 ? true : false));
  // when verbose is true, make sure warning+error are turned on:
  if( verbose )
    {
    gdcm::Trace::SetWarning( (verbose  > 0 ? true : false) );
    gdcm::Trace::SetError( (verbose  > 0 ? true : false) );
    }

  if( verbose )
    {
    std::cout << "Will parse: " << dirname << std::endl;
    std::cout << "Looking for tags: \n";
    std::copy(tags.begin(), tags.end(),
      std::ostream_iterator<gdcm::Tag>( std::cout, "\n"));
    std::copy(privatetags.begin(), privatetags.end(),
      std::ostream_iterator<gdcm::PrivateTag>( std::cout, "\n"));
    //std::cout << std::endl;
    }

  gdcm::Directory d;
  unsigned int nfiles = d.Load( dirname.c_str(), recursive );
  if( verbose ) d.Print( std::cout );
  std::cout << "done retrieving file list " << nfiles << " files found." <<  std::endl;

  gdcm::SmartPointer<gdcm::Scanner> ps = new gdcm::Scanner;
  gdcm::Scanner &s = *ps;
  //gdcm::SimpleSubjectWatcher watcher(ps, "Scanner");
  for( VectorTags::const_iterator it = tags.begin(); it != tags.end(); ++it)
    {
    s.AddTag( *it );
    }
  for( VectorPrivateTags::const_iterator it = privatetags.begin(); it != privatetags.end(); ++it)
    {
    s.AddPrivateTag( *it );
    }
  bool b = s.Scan( d.GetFilenames() );
  if( !b )
    {
    std::cerr << "Scanner failed" << std::endl;
    return 1;
    }
  if (print) s.Print( std::cout );

  return 0;
}
