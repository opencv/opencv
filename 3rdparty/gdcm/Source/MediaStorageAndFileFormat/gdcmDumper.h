/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDUMPER_H
#define GDCMDUMPER_H

#include "gdcmPrinter.h"

namespace gdcm
{

// It's a sink there is no output
/**
 * \brief Codec class
 * \note
 * Use it to simply dump value read from the file. No interpretation is done.
 * But it is real fast ! Almost no overhead
 */
class GDCM_EXPORT Dumper : public Printer
{
public:
  Dumper() { PrintStyle = CONDENSED_STYLE; }
  ~Dumper() {};
};

} // end namespace gdcm

#endif //GDCMDUMPER_H
