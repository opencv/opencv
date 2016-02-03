/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPRESENTATIONCONTEXTGENERATOR_H
#define GDCMPRESENTATIONCONTEXTGENERATOR_H

#include "gdcmDirectory.h"
#include "gdcmPresentationContext.h"

namespace gdcm
{
class TransferSyntax;

/**
 * \brief PresentationContextGenerator
 * This class is responsible for generating the proper PresentationContext that
 * will be used in subsequent operation during a DICOM Query/Retrieve
 * association. The step of the association is very sensible as special care
 * need to be taken to explicitly define what instance are going to be send
 * and how they are encoded.
 *
 * For example a PresentationContext will express that negotiation requires
 * that CT Image Storage are send using JPEG Lossless, while US Image Storage
 * are sent using RLE Transfer Syntax.
 *
 * Two very different API are exposed one which will always default to little
 * endian transfer syntax see GenerateFromUID()
 * This API is used for C-ECHO, C-FIND and C-MOVE (SCU).
 * Another API: GenerateFromFilenames() is used for C-STORE (SCU) as it will
 * loop over all filenames argument to detect the actual encoding. and
 * therefore find the proper encoding to be used.
 *
 * Two modes are available. The default mode (SetMergeModeToAbstractSyntax)
 * append PresentationContext (one AbstractSyntax and one TransferSyntax), as
 * long a they are different. Eg MR Image Storage/JPEG2000 and MR Image
 * Storage/JPEGLossless would be considered different.  the other mode
 * SetMergeModeToTransferSyntax merge any new TransferSyntax to the already
 * existing PresentationContext in order to re-use the same AbstractSyntax.
 *
 * \see PresentationContext
 */
class GDCM_EXPORT PresentationContextGenerator
{
public:
  PresentationContextGenerator();

  // Set MergeMode
  // Default mode, each pair AbstractSyntax/TransferSyntax are only merged when
  // exactly identical
  void SetMergeModeToAbstractSyntax();

  // Set MergeMode
  // Merge is done on a per AbstractSyntax basis. Any new TransferSyntax for a
  // given AbstractSyntax is merge to the existing PresentationContext refering
  // to that AbstractSyntax
  void SetMergeModeToTransferSyntax();

  /// Generate the PresentationContext array from a UID (eg. VerificationSOPClass)
  bool GenerateFromUID(UIDs::TSName asname);

  /// Generate the PresentationContext array from a File-Set. File specified needs to
  /// be valid DICOM files.
  /// Used for C-STORE operations
  bool GenerateFromFilenames(const Directory::FilenamesType &files);

  typedef std::vector<PresentationContext> PresentationContextArrayType;
  typedef PresentationContextArrayType::size_type SizeType;
  PresentationContextArrayType const &GetPresentationContexts() { return PresContext; }

  /// Not implemented for now. GDCM internally uses Implicit Little Endian
  void SetDefaultTransferSyntax( const TransferSyntax &ts );
protected:
  bool AddPresentationContext( const char *as, const char *ts );
  const char *GetDefaultTransferSyntax() const;

private:
  std::vector<PresentationContext> PresContext;
  static std::string DefaultTransferSyntax;
};

} // end namespace gdcm

#endif //GDCMPRESENTATIONCONTEXTGENERATOR_H
