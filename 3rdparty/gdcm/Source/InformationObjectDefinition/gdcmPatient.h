/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPATIENT_H
#define GDCMPATIENT_H

#include "gdcmTypes.h"
#include "gdcmStudy.h"

namespace gdcm
{
/**
 * \brief
 * See PS 3.3 - 2007
 * DICOM MODEL OF THE REAL-WORLD, p 54
 */
class GDCM_EXPORT Patient
{
public:
  Patient() {
  }

private:
  std::vector<Study> StudyList;
};

} // end namespace gdcm

#endif //GDCMPATIENT_H
