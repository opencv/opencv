/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMANONYMIZER_H
#define GDCMANONYMIZER_H

#include "gdcmFile.h"
#include "gdcmSubject.h"
#include "gdcmEvent.h"
#include "gdcmSmartPointer.h"

#include <map>

namespace gdcm
{
class TagPath;
class IOD;
class CryptographicMessageSyntax;

/**
 * \brief Anonymizer
 * This class is a multi purpose anonymizer. It can work in 2 mode:
 * - Full (irreversible) anonymizer (aka dumb mode)
 * - reversible de-identifier/re-identifier (aka smart mode). This implements the Basic Application Level Confidentiality Profile, DICOM PS 3.15-2009
 *
 * 1. dumb mode
 * This is a dumb anonymizer implementation. All it allows user is simple operation such as:
 *
 * Tag based functions:
 * - complete removal of DICOM attribute (Remove)
 * - make a tag empty, ie make it's length 0 (Empty)
 * - replace with another string-based value (Replace)
 *
 * DataSet based functions:
 * - Remove all group length attribute from a DICOM dataset (Group Length element are deprecated, DICOM 2008)
 * - Remove all private attributes
 * - Remove all retired attributes
 *
 * All function calls actually execute the user specified request. Previous
 * implementation were calling a general Anonymize function but traversing a
 * std::set is O(n) operation, while a simple user specified request is
 * O(log(n)) operation. So 'm' user interaction is O(m*log(n)) which is < O(n)
 * complexity.
 *
 * 2. smart mode
 * this mode implements the Basic Application Level Confidentiality Profile
 * (DICOM PS 3.15-2008) In this case, it is extremely important to use the same
 * Anonymizer class when anonymizing a FileSet. Once the Anonymizer
 * is destroyed its memory of known (already processed) UIDs will be lost.
 * which will make the anonymizer behaves incorrectly for attributes such as
 * Series UID Study UID where user want some consistency.  When attribute is
 * Type 1 / Type 1C, a dummy generator will take in the existing value and
 * produce a dummy value (a sha1 representation). sha1 algorithm is considered
 * to be cryptographically strong (compared to md5sum) so that we meet the
 * following two conditions:
 *  - Produce the same dummy value for the same input value
 *  - do not provide an easy way to retrieve the original value from the sha1 generated value
 *
 * This class implement the Subject/Observer pattern trigger the following event:
 * \li AnonymizeEvent
 * \li IterationEvent
 * \li StartEvent
 * \li EndEvent
 *
 * \see CryptographicMessageSyntax
 */
class GDCM_EXPORT Anonymizer : public Subject
{
public:
  Anonymizer():F(new File),CMS(NULL) {}
  ~Anonymizer();

  /// Make Tag t empty (if not found tag will be created)
  /// Warning: does not handle SQ element
  bool Empty( Tag const &t );
  //bool Empty( PrivateTag const &t );
  //bool Empty( TagPath const &t );

  /// remove a tag (even a SQ can be removed)
  /// Return code is false when tag t cannot be found
  bool Remove( Tag const &t );
  //bool Remove( PrivateTag const &t );
  //bool Remove( TagPath const &t );

  /// Replace tag with another value, if tag is not found it will be created:
  /// WARNING: this function can only execute if tag is a VRASCII
  bool Replace( Tag const &t, const char *value );

  /// when the value contains \0, it is a good idea to specify the length. This function
  /// is required when dealing with VRBINARY tag
  bool Replace( Tag const &t, const char *value, VL const & vl );
  //bool Replace( PrivateTag const &t, const char *value, VL const & vl );
  //bool Replace( TagPath const &t, const char *value, VL const & vl );

  /// Main function that loop over all elements and remove private tags
  bool RemovePrivateTags();

  /// Main function that loop over all elements and remove group length
  bool RemoveGroupLength();

  /// Main function that loop over all elements and remove retired element
  bool RemoveRetired();

  // TODO:
  // bool Remove( PRIVATE_TAGS | GROUP_LENGTH | RETIRED );

  /// Set/Get File
  void SetFile(const File& f) { F = f; }
  //const File &GetFile() const { return *F; }
  File &GetFile() { return *F; }

  /// PS 3.15 / E.1.1 De-Identifier
  /// An Application may claim conformance to the Basic Application Level Confidentiality Profile as a deidentifier
  /// if it protects all Attributes that might be used by unauthorized entities to identify the patient.
  /// NOT THREAD SAFE
  bool BasicApplicationLevelConfidentialityProfile(bool deidentify = true);

  /// Set/Get CMS key that will be used to encrypt the dataset within BasicApplicationLevelConfidentialityProfile
  void SetCryptographicMessageSyntax( CryptographicMessageSyntax *cms );
  const CryptographicMessageSyntax *GetCryptographicMessageSyntax() const;

  /// for wrapped language: instantiate a reference counted object
  static SmartPointer<Anonymizer> New() { return new Anonymizer; }

  /// Return the list of Tag that will be considered when anonymizing a DICOM file.
  static std::vector<Tag> GetBasicApplicationLevelConfidentialityProfileAttributes();

  /// Clear the internal mapping of real UIDs to generated UIDs
  /// \warning the mapping is definitely lost
  static void ClearInternalUIDs();

protected:
  // Internal function used to either empty a tag or set it's value to a dummy value (Type 1 vs Type 2)
  bool BALCPProtect(DataSet &ds, Tag const & tag, const IOD &iod);
  bool CanEmptyTag(Tag const &tag, const IOD &iod) const;
  void RecurseDataSet( DataSet & ds );

private:
  bool BasicApplicationLevelConfidentialityProfile1();
  bool BasicApplicationLevelConfidentialityProfile2();
  bool CheckIfSequenceContainsAttributeToAnonymize(File const &file, SequenceOfItems* sqi) const;

private:
  // I would prefer to have a smart pointer to DataSet but DataSet does not derive from Object...
  SmartPointer<File> F;
  CryptographicMessageSyntax *CMS;

  typedef std::pair< Tag, std::string > TagValueKey;
  typedef std::map< TagValueKey, std::string > DummyMapNonUIDTags;
  typedef std::map< std::string, std::string > DummyMapUIDTags;
  static DummyMapNonUIDTags dummyMapNonUIDTags;
  static DummyMapUIDTags dummyMapUIDTags;
};

/**
 * \example ManipulateFile.cs
 * \example ClinicalTrialIdentificationWorkflow.cs
 * This is a C# example on how to use Anonymizer
 */

} // end namespace gdcm

#endif //GDCMANONYMIZER_H
