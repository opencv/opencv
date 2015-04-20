/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDATASETHELPER_H
#define GDCMDATASETHELPER_H

#include "gdcmTypes.h"
#include "gdcmVR.h"

namespace gdcm
{
class DataSet;
class File;
class Tag;
class SequenceOfItems;

/**
 * \brief DataSetHelper (internal class, not intended for user level)
 *
 * \details
 */
class GDCM_EXPORT DataSetHelper
{
public:
  /// ds -> current dataset, which is not the same as the root dataset
  /// return VR::INVALID in case of error
  static VR ComputeVR(File const & file, DataSet const &ds, const Tag& tag);

  //static SequenceOfItems* ComputeSQFromByteValue(File const & file, DataSet const &ds, const Tag &tag);

protected:
};

} // end namespace gdcm

#endif // GDCMDATASETHELPER_H
