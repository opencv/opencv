/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCM_CHARLS_H
#define GDCM_CHARLS_H

/* Use the charls library configured for gdcm.  */
#include "gdcmTypes.h"
#ifdef GDCM_USE_SYSTEM_CHARLS
// It is expected that 1.0 API is used (JlsParameters instead of JlsParamaters)
# include <CharLS/interface.h>
# include <CharLS/util.h>
# include <CharLS/defaulttraits.h>
# include <CharLS/losslesstraits.h>
# include <CharLS/colortransform.h>
# include <CharLS/streams.h>
# include <CharLS/processline.h>
#else
#include "gdcmcharls/header.h"
#include "gdcmcharls/interface.h"
#include "gdcmcharls/util.h"
#include "gdcmcharls/defaulttraits.h"
#include "gdcmcharls/losslesstraits.h"
#include "gdcmcharls/colortransform.h"
#include "gdcmcharls/streams.h"
#include "gdcmcharls/processline.h"
#endif

#endif
