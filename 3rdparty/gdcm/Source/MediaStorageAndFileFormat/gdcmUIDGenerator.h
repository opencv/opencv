/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMUIDGENERATOR_H
#define GDCMUIDGENERATOR_H

#include "gdcmTypes.h"

namespace gdcm
{

/**
 * \brief Class for generating unique UID
 * \note bla
 * Usage:
 * When constructing a Series or Study UID, user *has* to keep around the UID,
 * otherwise the UID Generator will simply forget the value and create a new UID.
 */
class GDCM_EXPORT UIDGenerator
{
public:
  /// By default the root of a UID is a GDCM Root...
  UIDGenerator():Unique() {}

  // Function to override the GDCM root with a user one:
  // WARNING: This need to be a valid root, otherwise call will fail
  // Implementation note. According to DICOM standard PS 3.5, Section 9 :
  // Unique Identifiers (UIDs), we have:
  /*
  ...
  The <org root> portion of the UID uniquely identifies an organization, (i.e., manufacturer, research
  organization, NEMA, etc.), and is composed of a number of numeric components as defined by ISO 8824.
  The <suffix> portion of the UID is also composed of a number of numeric components, and shall be
  unique within the scope of the <org root>. This implies that the organization identified in the <org root> is
  responsible for guaranteeing <suffix> uniqueness by providing registration policies. These policies shall
  guarantee <suffix> uniqueness for all UID's created by that organization. Unlike the <org root>, which may
  be common for UID's in an organization, the <suffix> shall take different unique values between different
  UID's that identify different objects.
  ...
   */
  /// The current implementation in GDCM make use of the UUID implementation (RFC 4122) and has been
  /// successfully been tested for a root of size 26 bytes. Any longer root should work (the Generate()
  /// function will return a string), but will truncate the high bits of the 128bits UUID until the
  /// generated string fits on 64 bits. The authors disclaims any
  /// responsabitlity for garanteeing uniqueness of UIDs when the root is longer than 26 bytes.
  static void SetRoot(const char * root);
  static const char *GetRoot();

  /// Internally uses a std::string, so two calls have the same pointer !
  /// save into a std::string
  /// In summary do not write code like that:
  /// const char *uid1 = uid.Generate();
  /// const char *uid2 = uid.Generate();
  /// since uid1 == uid2
  const char* Generate();

  /// Find out if the string is a valid UID or not
  /// \todo: Move that in DataStructureAndEncoding (see FileMetaInformation::CheckFileMetaInformation)
  static bool IsValid(const char *uid);

  /// Return the default (GDCM) root UID:
  static const char *GetGDCMUID(); // who would want that in the public API ??

protected:
  static bool GenerateUUID(unsigned char *uuid_data);

private:
  static const char GDCM_UID[];
  static std::string Root;
  static std::string EncodedHardwareAddress;
  std::string Unique; // Buffer
};


} // end namespace gdcm

#endif //GDCMUIDGENERATOR_H
