/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCODEC_H
#define GDCMCODEC_H

#include "gdcmCoder.h"
#include "gdcmDecoder.h"

namespace gdcm
{

/**
 * \brief Codec class
 */
class GDCM_EXPORT Codec : public Coder, public Decoder
{
};

} // end namespace gdcm

#endif //GDCMCODEC_H
