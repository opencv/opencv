/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef GDCMQUERYFACTORY_H
#define GDCMQUERYFACTORY_H

#include "gdcmBaseRootQuery.h"

namespace gdcm{
    ///The character sets enumerated in PS 3.3 2009 Annex C, section C.12.1.1.2
    ///The resulting character set is stored in 0008,0005
    ///The conversion to the data element is performed by the QueryFactory itself
  enum ECharSet {
    eLatin1 = 0,
    eLatin2,
    eLatin3,
    eLatin4,
    eCyrillic,
    eArabic,
    eGreek,
    eHebrew,
    eLatin5, // Latin Alphabet No. 5 (Turkish) Extended
    eJapanese, // JIS X 0201 (Shift JIS) Extended
    eThai, // TIS 620-2533 (Thai) Extended
    eJapaneseKanjiMultibyte, // JIS X 0208 (Kanji) Extended
    eJapaneseSupplementaryKanjiMultibyte, // JIS X 0212 (Kanji) Extended
    eKoreanHangulHanjaMultibyte, // KS X 1001 (Hangul and Hanja) Extended
    eUTF8,
    eGB18030 // Chinese (Simplified) Extended
  };

/**
 * \brief QueryFactory.h
 * \note
 * contains: a class to produce a query based off of user-entered information
 *
 * Essentially, this class is used to construct a query based off of user input
 * (typically from the command line; if in code directly, the query itself
 * could just be instantiated)
 *
 * In theory, could also be used as the interface to validate incoming datasets
 * as belonging to a particular query style
 */
class GDCM_EXPORT QueryFactory
{
public:
  /// this function will produce a query (basically, a wrapper to a dataset that
  /// can validate whether or not the query is a valid cfind/cmove query) and the
  /// level of the query (patient, study, series, image). If the user provides
  /// an invalid instantiation (ie, study root type, query level of patient),
  /// then the result is NULL.
  static BaseRootQuery* ProduceQuery(ERootType inRootType, EQueryType inQueryType,
    EQueryLevel inQueryLevel);

  /// This function will produce the appropriate dataelement given a list of
  /// charsets. The first charset will be used directly, while the second and
  /// subsequent will be prepended with "ISO2022 ". Redundant character sets are
  /// not permitted, so if they are encountered, they will just be skipped. if
  /// UTF8 or GB18030 is used, no subsequent character sets will be used if the
  /// vector passed in is empty, then the dataelement that's passed out will be
  /// empty and Latin1 is the presumed encoding
  static DataElement ProduceCharacterSetDataElement(
    const std::vector<ECharSet>& inCharSetType);

  /// This function will return the corresponding ECharSet associated with the
  /// current locale of the running system (based on the value of locale() ).
  static ECharSet GetCharacterFromCurrentLocale();

  /// List all possible CharSet
  static void ListCharSets(std::ostream& os);
};

} // end namespace gdcm

#endif // GDCMQUERYFACTORY_H
