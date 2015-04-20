/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMFILECHANGETRANSFERSYNTAX_H
#define GDCMFILECHANGETRANSFERSYNTAX_H

#include "gdcmSubject.h"
#include "gdcmSmartPointer.h"

namespace gdcm
{
class FileChangeTransferSyntaxInternals;
class ImageCodec;
class TransferSyntax;

/**
 * \brief FileChangeTransferSyntax
 *
 * This class is a file-based (limited) replacement of the in-memory
 * ImageChangeTransferSyntax.
 *
 * This class provide a file-based compression-only mecanism. It will take in
 * an uncompressed DICOM image file (Pixel Data element). Then produced as
 * output a compressed DICOM file (Transfer Syntax will be updated).
 *
 * Currently it supports the following transfer syntax:
 * - JPEGLosslessProcess14_1
 */
class GDCM_EXPORT FileChangeTransferSyntax : public Subject
{
public:
  FileChangeTransferSyntax();
  ~FileChangeTransferSyntax();

  /// Set input filename (raw DICOM)
  void SetInputFileName(const char *filename_native);

  /// Set output filename (target compressed DICOM)
  void SetOutputFileName(const char *filename_native);

  /// Change the transfer syntax
  bool Change();

  /// Specify the Target Transfer Syntax
  void SetTransferSyntax( TransferSyntax const & ts );

  /// Retrieve the actual codec (valid after calling SetTransferSyntax)
  /// Only advanced users should call this function.
  ImageCodec * GetCodec();

  /// for wrapped language: instantiate a reference counted object
  static SmartPointer<FileChangeTransferSyntax> New() { return new FileChangeTransferSyntax; }

private:
  bool InitializeCopy();
  bool UpdateCompressionLevel(double level);
  FileChangeTransferSyntaxInternals *Internals;
};

} // end namespace gdcm

#endif //GDCMFILEANONYMIZER_H
