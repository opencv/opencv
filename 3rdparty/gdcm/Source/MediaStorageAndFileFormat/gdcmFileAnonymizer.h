/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFILEANONYMIZER_H
#define GDCMFILEANONYMIZER_H

#include "gdcmSubject.h"
#include "gdcmEvent.h"
#include "gdcmTag.h"
#include "gdcmVL.h"

namespace gdcm
{
class FileAnonymizerInternals;

/**
 * \brief FileAnonymizer
 *
 * This Anonymizer is a file-based Anonymizer. It requires a valid DICOM
 * file and will use the Value Length to skip over any information.
 *
 * It will not load the DICOM dataset taken from SetInputFileName() into memory
 * and should consume much less memory than Anonymizer.
 *
 * \warning: Each time you call Replace() with a value. This value will copied,
 * and stored in memory.  The behavior is not ideal for extremely large data
 * (larger than memory size). This class is really meant to take a large DICOM
 * input file and then only changed some small attribute.
 *
 * caveats:
 * \li This class will NOT work with unordered attributes in a DICOM File,
 * \li This class does neither recompute nor update the Group Length element,
 * \li This class currently does not update the File Meta Information header.
 * \li Only strict inplace Replace operation is supported when input and output
 *     file are the same.
 */
class GDCM_EXPORT FileAnonymizer : public Subject
{
public:
  FileAnonymizer();
  ~FileAnonymizer();

  /// Make Tag t empty
  /// Warning: does not handle SQ element
  void Empty( Tag const &t );

  /// remove a tag (even a SQ can be removed)
  void Remove( Tag const &t );

  /// Replace tag with another value, if tag is not found it will be created:
  /// WARNING: this function can only execute if tag is a VRASCII
  /// WARNING: Do not ever try to write a value in a SQ Data Element !
  void Replace( Tag const &t, const char *value_str );

  /// when the value contains \0, it is a good idea to specify the length. This function
  /// is required when dealing with VRBINARY tag
  void Replace( Tag const &t, const char *value_data, VL const & vl );

  /// Set input filename
  void SetInputFileName(const char *filename_native);

  /// Set output filename
  void SetOutputFileName(const char *filename_native);

  /// Write the output file
  bool Write();

private:
  bool ComputeEmptyTagPosition();
  bool ComputeRemoveTagPosition();
  bool ComputeReplaceTagPosition();
  FileAnonymizerInternals *Internals;
};

} // end namespace gdcm

#endif //GDCMFILEANONYMIZER_H
