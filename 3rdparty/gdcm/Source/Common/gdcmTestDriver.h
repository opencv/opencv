/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// This header is included by all the C++ test drivers in GDCM.
#ifndef GDCMTESTDRIVER_H
#define GDCMTESTDRIVER_H

// CREATE_TEST_SOURCELIST supports the flag EXTRA_INCLUDE but only one per call.
// So there is no way to specify we want to include two files... instead
// gather the #include in a single file and include that one...
#include <clocale> // C setlocale()
#include <locale> // C++ locale

#endif // GDCMTESTDRIVER_H
