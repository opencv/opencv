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
#include "gdcmQueryFactory.h"
#include "gdcmFindPatientRootQuery.h"
#include "gdcmMovePatientRootQuery.h"
#include "gdcmFindStudyRootQuery.h"
#include "gdcmMoveStudyRootQuery.h"

#include <locale>

namespace gdcm
{
BaseRootQuery* QueryFactory::ProduceQuery(ERootType inRootType, EQueryType inQueryType,
 EQueryLevel inQueryLevel)
{
  BaseRootQuery* theReturn = NULL;
  switch (inQueryType)
    {
  case eFind:
    switch (inRootType)
      {
    case ePatientRootType:
      theReturn = new FindPatientRootQuery();
      break;
    case eStudyRootType:
      if (inQueryLevel != ePatient)
        theReturn = new FindStudyRootQuery();
      break;
      }
    break;
  case eMove:
    switch (inRootType)
      {
    case ePatientRootType:
      theReturn = new MovePatientRootQuery();
      break;
    case eStudyRootType:
      if (inQueryLevel != ePatient)
        theReturn = new MoveStudyRootQuery();
      break;
      }
    break;
    }
  if( theReturn )
    theReturn->InitializeDataSet(inQueryLevel);
  return theReturn;
}

/*
LATIN1        ISO 8859-1
LATIN2        ISO 8859-2
LATIN3        ISO 8859-3
LATIN4        ISO 8859-4
ISO_8859_5    ISO 8859-5 Latin/Cyrillic
ISO_8859_6    ISO 8859-6 Latin/Arabic
ISO_8859_7    ISO 8859-7 Latin/Greek
ISO_8859_8    ISO 8859-8 Latin/Hebrew
LATIN5        ISO 8859-9
LATIN6        ISO 8859-10
LATIN7        ISO 8859-13
LATIN8        ISO 8859-14
LATIN9        ISO 8859-15
LATIN10       ISO 8859-16
*/
ECharSet QueryFactory::GetCharacterFromCurrentLocale()
{
  const char *charset = System::GetLocaleCharset();
  if( charset )
    {
    // UTF-8 (this is the default for UNIX):
    if( strcmp( charset, "UTF-8" ) == 0 ) return eUTF8;
    // US-ASCII, simply return latin1
    else if( strcmp( charset, "US-ASCII" ) == 0 ) return eLatin1;
    else if( strcmp( charset, "ANSI_X3.4-1968" ) == 0 ) return eLatin1;
    // Latin-1
    else if( strcmp( charset, "ISO-8859-1" ) == 0 ) return eLatin1;
    else if( strcmp( charset, "ISO-8859-2" ) == 0 ) return eLatin2;
    else if( strcmp( charset, "ISO-8859-3" ) == 0 ) return eLatin3;
    else if( strcmp( charset, "ISO-8859-4" ) == 0 ) return eLatin4;
    else if( strcmp( charset, "ISO-8859-5" ) == 0 ) return eCyrillic;
    else if( strcmp( charset, "ISO-8859-6" ) == 0 ) return eArabic;
    else if( strcmp( charset, "ISO-8859-7" ) == 0 ) return eGreek;
    else if( strcmp( charset, "ISO-8859-8" ) == 0 ) return eHebrew;
    else if( strcmp( charset, "ISO-8859-9" ) == 0 ) return eLatin5;
    //
    else if( strcmp( charset, "EUC-JP" ) == 0 )     return eJapanese;
    else if( strcmp( charset, "TIS-620" ) == 0 )    return eThai;
    else if( strcmp( charset, "EUC-JP" ) == 0 )     return eJapaneseKanjiMultibyte;
    else if( strcmp( charset, "EUC-JP" ) == 0 )     return eJapaneseSupplementaryKanjiMultibyte;
    else if( strcmp( charset, "EUC-KR" ) == 0 )     return eKoreanHangulHanjaMultibyte;
    //else if( strcmp( charset, "UTF-8" ) == 0 )      return eUTF8;
    else if( strcmp( charset, "GB18030" ) == 0 )    return eGB18030;
    gdcmWarningMacro( "Problem mapping Locale Charset: " << charset );
    }
  gdcmWarningMacro( "Default to Latin-1" );
  return eLatin1;
}

///This function will produce the appropriate dataelement given a list of charsets.
///The first charset will be used directly, while the second and subsequent
///will be prepended with "ISO2022 ".  Redundant character sets are not permitted,
///so if they are encountered, they will just be skipped.
///if UTF8 or GB18030 is used, no subsequent character sets will be used
DataElement QueryFactory::ProduceCharacterSetDataElement(const std::vector<ECharSet>& inCharSetType)
{
  DataElement theReturn;
  //use the 'visited' array to make sure that if a redundant character set is entered,
  //it's skipped rather than produce a malformed tag.
  bool visited[eGB18030+1];
  memset(visited, 0, (eGB18030+1)*sizeof(bool));

  if (inCharSetType.empty())
    return theReturn;

  std::vector<ECharSet>::const_iterator itor;
  std::string theOutputString;
  for (itor = inCharSetType.begin(); itor < inCharSetType.end(); itor++)
    {
    if (itor > inCharSetType.begin())
      {
      theOutputString += "ISO 2022 ";
      }
    else
      {
      theOutputString += "ISO_IR ";
      }

    if (visited[*itor]) continue;
    switch (*itor)
      {
    default:
    case eLatin1:
      theOutputString += "100";
      break;
    case eLatin2:
      theOutputString += "101";
      break;
    case eLatin3:
      theOutputString += "109";
      break;
    case eLatin4:
      theOutputString += "110";
      break;
    case eCyrillic:
      theOutputString += "144";
      break;
    case eArabic:
      theOutputString += "127";
      break;
    case eGreek:
      theOutputString += "126";
      break;
    case eHebrew:
      theOutputString += "138";
      break;
    case eLatin5:
      theOutputString += "148";
      break;
    case eJapanese:
      theOutputString += "13";
      break;
    case eThai:
      theOutputString += "166";
      break;
    case eJapaneseKanjiMultibyte:
      theOutputString += "87";
      break;
    case eJapaneseSupplementaryKanjiMultibyte:
      theOutputString += "159";
      break;
    case eKoreanHangulHanjaMultibyte:
      theOutputString += "149";
      break;
      //for the next two, they are only valid if they are
      //the only ones that appear
    case eUTF8:
      theOutputString = "ISO_IR 192";
      itor = inCharSetType.end() - 1; //stop the loop
      break;
    case eGB18030:
      theOutputString = "GB13080";
      itor = inCharSetType.end() - 1; //stop the loop
      break;
      }
    if (itor < (inCharSetType.end()-1))
      {
      theOutputString += "\\";
      // the following code will not work for UTF-8 and eGB18030
      assert( itor < inCharSetType.end() );
      visited[*itor] = true;
      }
  }

  if( theOutputString.size() % 2 )
    theOutputString.push_back( ' ' ); // no \0 !
  theReturn.SetByteValue(theOutputString.c_str(),
    (uint32_t)theOutputString.length());
  theReturn.SetTag(Tag(0x0008, 0x0005));
  return theReturn;
}


void QueryFactory::ListCharSets(std::ostream& os)
{
  os << "The following character sets are supported by GDCM Network Queries." << std::endl;
  os << "The number in the parenthesis is the index to select." << std::endl;
  os << "Note that multiple selections are possible." << std::endl;
  os << "Latin1 (0): This is the default if nothing is specified." <<std::endl;
  os << "Latin2 (1)" << std::endl;
  os << "Latin3 (2)" << std::endl;
  os << "Latin4 (3)" << std::endl;
  os << "Cyrillic (4)" << std::endl;
  os << "Arabic (5)" << std::endl;
  os << "Greek (6)" << std::endl;
  os << "Hebrew (7)" << std::endl;
  os << "Latin5 (8)" << std::endl;
  os << "Japanese (9)" << std::endl;
  os << "Thai (10)" << std::endl;
  os << "Kanji (Japanese) (11)+" << std::endl;
  os << "Supplementary Kanji (12)+" << std::endl;
  os << "Hangul and Hanja (Korean) (13)+" << std::endl;
  os << "UTF-8 (14)++" << std::endl;
  os << "GB1308 (15)++" << std::endl;
  os << "+ These character sets must be chosen second or later in a set." << std::endl;
  os << "++ These character sets must be chosen alone, in no set." << std::endl;
}

} // end namespace gdcm
