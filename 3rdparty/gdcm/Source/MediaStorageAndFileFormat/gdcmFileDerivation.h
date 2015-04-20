/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFILEDERIVATION_H
#define GDCMFILEDERIVATION_H

#include "gdcmFile.h"

namespace gdcm
{

class FileDerivationInternals;
class DataSet;
/**
 * \brief FileDerivation class
 * See PS 3.16 - 2008 For the list of Code Value that can be used for in
 * Derivation Code Sequence
 *
 * URL: http://medical.nema.org/medical/dicom/2008/08_16pu.pdf
 *
 * DICOM Part 16 has two Context Groups CID 7202 and CID 7203 which contain a
 * set of codes defining reason for a source image reference (ie. reason code
 * for referenced image sequence) and a coded description of the deriation
 * applied to the new image data from the original. Both these context groups
 * are extensible.
 *
 * File Derivation is compulsary when creating a lossy derived image.
 */
class GDCM_EXPORT FileDerivation
{
public:
  FileDerivation();
  ~FileDerivation();

  /// Create the proper reference. Need to pass the original SOP Class UID and the original
  /// SOP Instance UID, so that those value can be used as Reference.
  /// \warning referencedsopclassuid and referencedsopinstanceuid needs to be \0 padded. This
  /// is not compatible with how ByteValue->GetPointer works.
  bool AddReference(const char *referencedsopclassuid, const char *referencedsopinstanceuid);

  // CID 7202 Source Image Purposes of Reference
  // {"DCM",121320,"Uncompressed predecessor"},

  /// Specify the Purpose Of Reference Code Value. Eg. 121320
  void SetPurposeOfReferenceCodeSequenceCodeValue(unsigned int codevalue);

  // CID 7203 Image Derivation
  // { "DCM",113040,"Lossy Compression" },

  /// Specify the Derivation Code Sequence Code Value. Eg 113040
  void SetDerivationCodeSequenceCodeValue(unsigned int codevalue);

  /// Specify the Derivation Description. Eg "lossy conversion"
  void SetDerivationDescription( const char *dd );

  /// Change
  bool Derive();

  /// Set/Get File
  void SetFile(const File& f) { F = f; }
  File &GetFile() { return *F; }
  const File &GetFile() const { return *F; }

protected:
  bool AddDerivationDescription();
  bool AddSourceImageSequence();
  bool AddPurposeOfReferenceCodeSequence(DataSet &ds);

private:
  SmartPointer<File> F;
  FileDerivationInternals *Internals;
};

/**
 * \example GenFakeImage.cxx
 * \example ReformatFile.cs
 * This is a C++ example on how to use FileDerivation
 */


} // end namespace gdcm

#endif //GDCMFILEDERIVATION_H
