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
 * ./bin/ex_getopt -a -b -c foo -d bar -1
 */

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include "getopt.h"

int
main (int argc, char **argv) {
  int c;
  int digit_optind = 0;
  //int optind = 0;
  //int optarg = 0;

  while (1) {
    int this_option_optind = optind ? optind : 1;
    int option_index = 0;
    static struct option long_options[] = {
        {"add", 1, 0, 0},
        {"append", 0, 0, 0},
        {"delete", 1, 0, 0},
        {"verbose", 0, 0, 0},
        {"create", 1, 0, 'c'},
        {"file", 1, 0, 0},
        {0, 0, 0, 0}
    };

    c = getopt_long (argc, argv, "abc:d:012",
      long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 0:
      printf ("option %s", long_options[option_index].name);
      if (optarg)
        printf (" with arg %s", optarg);
      printf ("\n");
      break;

    case '0':
    case '1':
    case '2':
      if (digit_optind != 0 && digit_optind != this_option_optind)
        printf ("digits occur in two different argv-elements.\n");
      digit_optind = this_option_optind;
      printf ("option %c\n", c);
      break;

    case 'a':
      printf ("option a\n");
      break;

    case 'b':
      printf ("option b\n");
      break;

    case 'c':
      printf ("option c with value ’%s’\n", optarg);
      break;

    case 'd':
      printf ("option d with value ’%s’\n", optarg);
      break;

    case '?':
      break;

    default:
      printf ("?? getopt returned character code 0%o ??\n", c);
    }
  }

  if (optind < argc) {
    printf ("non-option ARGV-elements: ");
    while (optind < argc)
      printf ("%s ", argv[optind++]);
    printf ("\n");
  }

  exit (0);
}

