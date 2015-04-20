/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDICOMDIRGenerator.h"
#include "gdcmFile.h"
#include "gdcmSmartPointer.h"

#include "gdcmUIDGenerator.h"
#include "gdcmFilename.h"
#include "gdcmScanner.h"
#include "gdcmAttribute.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmExplicitDataElement.h"
#include "gdcmTag.h"
#include "gdcmVR.h"
#include "gdcmCodeString.h"


namespace gdcm
{

class DICOMDIRGeneratorInternal
{
public:
  DICOMDIRGeneratorInternal():F(new File) {}
  SmartPointer<File> F;
  typedef Directory::FilenamesType  FilenamesType;
  FilenamesType fns;
  typedef Directory::FilenameType  FilenameType;
  FilenameType rootdir;
  Scanner scanner;
  std::vector<uint32_t> OffsetTable;
  std::string FileSetID;
};

bool DICOMDIRGenerator::ComputeDirectoryRecordsOffset(const SequenceOfItems *sqi, VL start)
{
  SequenceOfItems::SizeType nitems = sqi->GetNumberOfItems();
  std::vector<uint32_t> &offsets = Internals->OffsetTable;
  Internals->OffsetTable.resize( nitems + 1 );
  offsets[0] = start;
  for(SequenceOfItems::SizeType i = 1; i <= nitems; ++i)
    {
    const Item &item = sqi->GetItem(i);
    offsets[i] = offsets[i-1] + item.GetLength<ExplicitDataElement>();
    }

//#define MDEBUG
#ifdef MDEBUG
  for(unsigned int i = 0; i <= nitems; ++i)
    {
    std::cout << "offset #" << i << " -> "<< offsets[i] << std::endl;
    }
#endif

  return true;
}

static const char *GetLowerLevelDirectoryRecord(const char *input)
{
  if( !input ) return NULL;

  if( strcmp( input, "PATIENT " ) == 0 )
    {
    return "STUDY ";
    }
  else if( strcmp( input, "STUDY ") == 0 )
    {
    return "SERIES";
    }
  else if( strcmp( input, "SERIES" ) == 0 )
    {
    return "IMAGE ";
    }
  else if( strcmp( input, "IMAGE " ) == 0 )
    {
    return NULL;
    }
  assert( 0 );
  //std::cerr << "COULD NOT FIND:" << input << std::endl;
  return NULL;
}


DICOMDIRGenerator::MyPair DICOMDIRGenerator::GetReferenceValueForDirectoryType(size_t itemidx)
{
  /*
   * PS 3.11 - 2008 / D.3.3 Directory Information in DICOMDIR
   * The Patient ID at the patient level shall be unique for each patient directory record in one File Set.
  */
  /*
   * This function will return the Patient ID when directorytype          == "PATIENT"
   * This function will return the Study Instance UID when directorytype  == "STUDY"
   * This function will return the Series Instance UID when directorytype == "SERIES"
   * This function will return the SOP Instance UID when directorytype    == "IMAGE"
   */
  MyPair ret;
  const SequenceOfItems *sqi = GetDirectoryRecordSequence();
  const Item &item = sqi->GetItem(itemidx);
  const DataSet &ds = item.GetNestedDataSet();
  Attribute<0x4,0x1430> directoryrecordtype;
  directoryrecordtype.Set( ds );

  const char *input = directoryrecordtype.GetValue();
  assert( input );

  if( strcmp( input, "PATIENT " ) == 0 )
    {
    Attribute<0x10,0x20> patientid;
    patientid.Set( ds );
    assert( patientid.GetValue() );
    ret.first = patientid.GetValue();
    ret.second = patientid.GetTag();
    }
  else if( strcmp( input, "STUDY ") == 0 )
    {
    Attribute <0x20,0xd> studyuid;
    studyuid.Set( ds );
    assert( studyuid.GetValue() );
    ret.first = studyuid.GetValue();
    ret.second = studyuid.GetTag();
    }
  else if( strcmp( input, "SERIES" ) == 0 )
    {
    Attribute <0x20,0xe> seriesuid;
    seriesuid.Set( ds );
    assert( seriesuid.GetValue() );
    ret.first = seriesuid.GetValue();
    ret.second = seriesuid.GetTag();
    }
  else if( strcmp( input, "IMAGE " ) == 0 )
    {
    Attribute <0x04,0x1511> sopuid;
    sopuid.Set( ds );
    assert( sopuid.GetValue() );
    ret.first = sopuid.GetValue();
    ret.second = Tag(0x8,0x18); // watch out !
    }
  else
    {
    assert( 0 );
    }
  return ret;
}

static Tag GetParentTag(Tag const &t)
{
  Tag ret;
  if( t == Tag(0x8,0x18) )
    {
    ret = Tag(0x20,0xe);
    }
  else if( t == Tag(0x20,0xe) )
    {
    ret = Tag(0x20,0xd);
    }
  else if( t == Tag(0x20,0xd) )
    {
    ret = Tag(0x10,0x20);
    }
  else if( t == Tag(0x10,0x20) )
    {
    ret = Tag(0x0,0x0);
    }
  else
    {
    assert( 0 );
    }
  return ret;
}

bool DICOMDIRGenerator::SeriesBelongToStudy(const char *seriesuid, const char *studyuid)
{
  assert( seriesuid );
  assert( studyuid );
  const Scanner &scanner = GetScanner();

  Scanner::TagToValue const &ttv = scanner.GetMappingFromTagToValue(Tag(0x20,0xe), seriesuid);
  Tag tstudyuid(0x20,0xd);
  bool b = false;
  if( ttv.find( tstudyuid ) != ttv.end() )
    {
    const char *v = ttv.find(tstudyuid)->second;
    if( v && strcmp(v, studyuid ) == 0 )
      {
      b = true;
      }
    }

  return b;
}

bool DICOMDIRGenerator::ImageBelongToSeries(const char *sopuid, const char *seriesuid, Tag const &t1, Tag const &t2)
{
  assert( seriesuid );
  assert( sopuid );
  const Scanner &scanner = GetScanner();

  Scanner::TagToValue const &ttv = scanner.GetMappingFromTagToValue(t1, sopuid);
  bool b = false;
  if( ttv.find( t2 ) != ttv.end() )
    {
    const char *v = ttv.find(t2)->second;
    if( v && strcmp(v, seriesuid) == 0 )
      {
      b = true;
      }
    }

  return b;
}

bool DICOMDIRGenerator::ImageBelongToSameSeries(const char *sopuid1, const char *sopuid2, Tag const &t)
{
  assert( sopuid1 );
  assert( sopuid2 );
  const Scanner &scanner = GetScanner();

  Scanner::TagToValue const &ttv1 = scanner.GetMappingFromTagToValue(t, sopuid1);
  Scanner::TagToValue const &ttv2 = scanner.GetMappingFromTagToValue(t, sopuid2);
  Tag tseriesuid = GetParentTag( t );
  if( tseriesuid == Tag(0x0,0x0) )
    {
    // Let's pretend that Patient belong to the same 'root' element:
    return true;
    }
  bool b = false;
  const char *seriesuid1 = NULL;
  if( ttv1.find( tseriesuid ) != ttv1.end() )
    {
    seriesuid1 = ttv1.find(tseriesuid)->second;
    }
  const char *seriesuid2 = NULL;
  if( ttv2.find( tseriesuid ) != ttv2.end() )
    {
    seriesuid2 = ttv2.find(tseriesuid)->second;
    }
  assert( seriesuid1 );
  assert( seriesuid2 );

  b = strcmp( seriesuid1, seriesuid2) == 0;
  return b;
}

size_t DICOMDIRGenerator::FindLowerLevelDirectoryRecord( size_t item1, const char *directorytype )
{
//  return FindNextDirectoryRecord( item1, GetLowerLevelDirectoryRecord( directorytype ) );
  const char *lowerdirectorytype = GetLowerLevelDirectoryRecord( directorytype );
  if( !lowerdirectorytype ) return 0;

  const SequenceOfItems *sqi = GetDirectoryRecordSequence();
  SequenceOfItems::SizeType nitems = sqi->GetNumberOfItems();
  for(SequenceOfItems::SizeType i = item1 + 1; i <= nitems; ++i)
    {
    const Item &item = sqi->GetItem(i);
    const DataSet &ds = item.GetNestedDataSet();
    Attribute<0x4,0x1430> directoryrecordtype;
    directoryrecordtype.Set( ds );

    // found a match ?
    if( strcmp( lowerdirectorytype, directoryrecordtype.GetValue() ) == 0 )
      {
      // Need to make sure belong to same parent record:
      MyPair refval1 = GetReferenceValueForDirectoryType(item1);
      MyPair refval2 = GetReferenceValueForDirectoryType(i);
      bool b = ImageBelongToSeries(refval2.first.c_str(),
        refval1.first.c_str(), refval2.second, refval1.second);
      if( b ) return i;
      }
    //assert( strncmp( lowerdirectorytype, directoryrecordtype.GetValue(), strlen( lowerdirectorytype ) ) != 0 );
    }

  // Not found
  return 0;

}

/*
 * Finding the next Directory Record type is easy, simply starting from the start and iterating
 * to the end guarantee travering everything without omitting anyone.
 *
 * TODO: Need to make sure that Series belong to the same Study...
 */
size_t DICOMDIRGenerator::FindNextDirectoryRecord( size_t item1, const char *directorytype )
{
  if( !directorytype ) return 0;
  const SequenceOfItems *sqi = GetDirectoryRecordSequence();
  SequenceOfItems::SizeType nitems = sqi->GetNumberOfItems();
  for(SequenceOfItems::SizeType i = item1 + 1; i <= nitems; ++i)
    {
    const Item &item = sqi->GetItem(i);
    const DataSet &ds = item.GetNestedDataSet();
    Attribute<0x4,0x1430> directoryrecordtype;
    directoryrecordtype.Set( ds );

    // found a match ?
    if( strcmp( directorytype, directoryrecordtype.GetValue() ) == 0 )
      {
      // Need to make sure belong to same parent record:
      MyPair refval1 = GetReferenceValueForDirectoryType(item1);
      MyPair refval2 = GetReferenceValueForDirectoryType(i);
      bool b = ImageBelongToSameSeries(refval1.first.c_str(), refval2.first.c_str(), refval1.second);
      if( b ) return i;
      }
    //assert( strncmp( directorytype, directoryrecordtype.GetValue(), strlen( directorytype ) ) != 0 );
    }

  // Not found
  return 0;
}

bool DICOMDIRGenerator::TraverseDirectoryRecords(VL start )
{
  SequenceOfItems *sqi = GetDirectoryRecordSequence();

  ComputeDirectoryRecordsOffset(sqi, start);

  SequenceOfItems::SizeType nitems = sqi->GetNumberOfItems();
  for(SequenceOfItems::SizeType i = 1; i <= nitems; ++i)
    {
    Item &item = sqi->GetItem(i);
    DataSet &ds = item.GetNestedDataSet();
    Attribute<0x4,0x1430> directoryrecordtype;
    directoryrecordtype.Set( ds );
    //std::cout << "FOUND DIRECTORY TYPE:" << directoryrecordtype.GetValue() << std::endl;
    size_t next = FindNextDirectoryRecord( i, directoryrecordtype.GetValue() );
    if( next )
      {
      Attribute<0x4,0x1400> offsetofthenextdirectoryrecord = {0};
      offsetofthenextdirectoryrecord.SetValue( Internals->OffsetTable[ next - 1 ] );
      ds.Replace( offsetofthenextdirectoryrecord.GetAsDataElement() );
      }
    size_t lower = FindLowerLevelDirectoryRecord( i, directoryrecordtype.GetValue() );
    if( lower )
      {
      Attribute<0x4,0x1420> offsetofreferencedlowerleveldirectoryentity = {0};
      offsetofreferencedlowerleveldirectoryentity.SetValue( Internals->OffsetTable[ lower - 1 ] );
      ds.Replace( offsetofreferencedlowerleveldirectoryentity.GetAsDataElement() );
      }
    }
  return true;
}

template<uint16_t Group, uint16_t Element>
void SingleDataElementInserter(DataSet &ds, Scanner const & scanner)
{
  Attribute<Group,Element> patientsname;
  Scanner::ValuesType patientsnames = scanner.GetValues( patientsname.GetTag() );
#ifndef NDEBUG
  const unsigned int npatient = patientsnames.size();
  assert( npatient == 1 );
#endif

  Scanner::ValuesType::const_iterator it = patientsnames.begin();
  patientsname.SetValue( it->c_str() );
  ds.Insert( patientsname.GetAsDataElement() );
}


/*
  (fffe,e000) na "Directory Record" PATIENT #=8           # u/l, 1 Item
  #  offset=$374
    (0004,1400) up 0                                        #   4, 1 OffsetOfTheNextDirectoryRecord
    (0004,1410) US 65535                                    #   2, 1 RecordInUseFlag
    (0004,1420) up 502                                      #   4, 1 OffsetOfReferencedLowerLevelDirectoryEntity
    (0004,1430) CS [PATIENT]                                #   8, 1 DirectoryRecordType
    (0010,0010) PN [Test^PixelSpacing]                      #  18, 1 PatientsName
    (0010,0020) LO [62354PQGRRST]                           #  12, 1 PatientID
    (0010,0030) DA (no value available)                     #   0, 0 PatientsBirthDate
    (0010,0040) CS (no value available)                     #   0, 0 PatientsSex
  (fffe,e00d) na "ItemDelimitationItem"                   #   0, 0 ItemDelimitationItem
*/
bool DICOMDIRGenerator::AddPatientDirectoryRecord()
{
  DataSet &rootds = GetFile().GetDataSet();
  Scanner const & scanner = GetScanner();

  Attribute<0x10,0x20> patientid;
  Scanner::ValuesType patientids = scanner.GetValues( patientid.GetTag() );
  //unsigned int npatients = patientids.size();

  const DataElement &de = rootds.GetDataElement( Tag(0x4,0x1220) );
  //SequenceOfItems * sqi = (SequenceOfItems*)de.GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = de.GetValueAsSQ();

  Scanner::ValuesType::const_iterator it = patientids.begin();
  for( ; it  != patientids.end(); ++it)
    {
    Item item;
    item.SetVLToUndefined();
    DataSet &ds = item.GetNestedDataSet();

    // (0004,1400) up 0                                        #   4, 1 OffsetOfTheNextDirectoryRecord
    // (0004,1410) US 65535                                    #   2, 1 RecordInUseFlag
    // (0004,1420) up 502                                      #   4, 1 OffsetOfReferencedLowerLevelDirectoryEntity
    // (0004,1430) CS [PATIENT]                                #   8, 1 DirectoryRecordType
    Attribute<0x4,0x1400> offsetofthenextdirectoryrecord = {0};
    ds.Insert( offsetofthenextdirectoryrecord.GetAsDataElement() );
    Attribute<0x4,0x1410> recordinuseflag = {0xFFFF};
    ds.Insert( recordinuseflag.GetAsDataElement() );
    Attribute<0x4,0x1420> offsetofreferencedlowerleveldirectoryentity = {0};
    ds.Insert( offsetofreferencedlowerleveldirectoryentity.GetAsDataElement() );
    Attribute<0x4,0x1430> directoryrecordtype;
    directoryrecordtype.SetValue( "PATIENT" );
    ds.Insert( directoryrecordtype.GetAsDataElement() );
    const char *pid = it->c_str();
    if( ! (pid && *pid) )
      {
      const char *fn = scanner.GetFilenameFromTagToValue(patientid.GetTag(), pid);
      gdcmErrorMacro( "Missing Patient ID from file: " << fn );
      (void)fn; //warning removal
      return false;
      }
    gdcmAssertAlwaysMacro( pid && *pid );
    patientid.SetValue( pid );
    ds.Insert( patientid.GetAsDataElement() );

    Scanner::TagToValue const &ttv = scanner.GetMappingFromTagToValue(patientid.GetTag(), pid);
    Attribute<0x10,0x10> patientsname;
    if( ttv.find( patientsname.GetTag() ) != ttv.end() )
      {
      patientsname.SetValue( ttv.find(patientsname.GetTag())->second );
      ds.Insert( patientsname.GetAsDataElement() );
      }

    //SingleDataElementInserter<0x10,0x10>(ds, scanner);
    //SingleDataElementInserter<0x10,0x20>(ds, scanner);
    //SingleDataElementInserter<0x10,0x30>(ds, scanner);
    //SingleDataElementInserter<0x10,0x40>(ds, scanner);

    sqi->AddItem( item );
    }

  return true;
}

/*
  (fffe,e000) na "Directory Record" STUDY #=10            # u/l, 1 Item
  #  offset=$502
    (0004,1400) up 0                                        #   4, 1 OffsetOfTheNextDirectoryRecord
    (0004,1410) US 65535                                    #   2, 1 RecordInUseFlag
    (0004,1420) up 748                                      #   4, 1 OffsetOfReferencedLowerLevelDirectoryEntity
    (0004,1430) CS [STUDY]                                  #   6, 1 DirectoryRecordType
    (0008,0020) DA [20050624]                               #   8, 1 StudyDate
    (0008,0030) TM [104221]                                 #   6, 1 StudyTime
    (0008,0050) SH [8-13547713751]                          #  14, 1 AccessionNumber
    (0008,1030) LO [Test support of different pixel spacing attributes] #  50, 1 StudyDescription
    (0020,000d) UI [1.3.6.1.4.1.5962.1.2.65535.1119624141.7160.0] #  44, 1 StudyInstanceUID
    (0020,0010) SH [734591762345]                           #  12, 1 StudyID
  (fffe,e00d) na "ItemDelimitationItem"                   #   0, 0 ItemDelimitationItem
*/
bool DICOMDIRGenerator::AddStudyDirectoryRecord()
{
  DataSet &rootds = GetFile().GetDataSet();
  Scanner const & scanner = GetScanner();

  Attribute<0x20,0xd> studyinstanceuid;
  Scanner::ValuesType studyinstanceuids = scanner.GetValues( studyinstanceuid.GetTag() );

  const DataElement &de = rootds.GetDataElement( Tag(0x4,0x1220) );
  //SequenceOfItems * sqi = (SequenceOfItems*)de.GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = de.GetValueAsSQ();

  Scanner::ValuesType::const_iterator it = studyinstanceuids.begin();
  for( ; it  != studyinstanceuids.end(); ++it)
    {
    Item item;
    item.SetVLToUndefined();
    DataSet &ds = item.GetNestedDataSet();

    Attribute<0x4,0x1400> offsetofthenextdirectoryrecord = {0};
    ds.Insert( offsetofthenextdirectoryrecord.GetAsDataElement() );
    Attribute<0x4,0x1410> recordinuseflag = {0xFFFF};
    ds.Insert( recordinuseflag.GetAsDataElement() );
    Attribute<0x4,0x1420> offsetofreferencedlowerleveldirectoryentity = {0};
    ds.Insert( offsetofreferencedlowerleveldirectoryentity.GetAsDataElement() );
    Attribute<0x4,0x1430> directoryrecordtype;
    directoryrecordtype.SetValue( "STUDY" );
    ds.Insert( directoryrecordtype.GetAsDataElement() );
    const char *studyuid = it->c_str();
    if( ! (studyuid && *studyuid) )
      {
      const char *fn = scanner.GetFilenameFromTagToValue(studyinstanceuid.GetTag(), studyuid);
      gdcmErrorMacro( "Missing Study Instance UID from file: " << fn );
      (void)fn;//warning removal
      return false;
      }
    gdcmAssertAlwaysMacro( studyuid && *studyuid );
    studyinstanceuid.SetValue( studyuid );
    ds.Insert( studyinstanceuid.GetAsDataElement() );

    //SingleDataElementInserter<0x20,0xd>(ds, scanner);
    //SingleDataElementInserter<0x8,0x20>(ds, scanner);
    //SingleDataElementInserter<0x8,0x30>(ds, scanner);
    //SingleDataElementInserter<0x8,0x1030>(ds, scanner);
    //SingleDataElementInserter<0x8,0x50>(ds, scanner);
    //SingleDataElementInserter<0x20,0x10>(ds, scanner);
    Scanner::TagToValue const &ttv = scanner.GetMappingFromTagToValue(studyinstanceuid.GetTag(), studyuid);

    Attribute<0x8,0x20> studydate;
    if( ttv.find( studydate.GetTag() ) != ttv.end() )
      {
      studydate.SetValue( ttv.find(studydate.GetTag())->second );
      ds.Insert( studydate.GetAsDataElement() );
      }
    Attribute<0x8,0x30> studytime;
    if( ttv.find( studytime.GetTag() ) != ttv.end() )
      {
      studytime.SetValue( ttv.find(studytime.GetTag())->second );
      ds.Insert( studytime.GetAsDataElement() );
      }
    Attribute<0x8,0x1030> studydesc;
    if( ttv.find( studydesc.GetTag() ) != ttv.end() )
      {
      studydesc.SetValue( ttv.find(studydesc.GetTag())->second );
      ds.Insert( studydesc.GetAsDataElement() );
      }
    Attribute<0x8,0x50> accessionnumber;
    if( ttv.find( accessionnumber.GetTag() ) != ttv.end() )
      {
      accessionnumber.SetValue( ttv.find(accessionnumber.GetTag())->second );
      ds.Insert( accessionnumber.GetAsDataElement() );
      }
    Attribute<0x20,0x10> studyid;
    if( ttv.find( studyid.GetTag() ) != ttv.end() )
      {
      studyid.SetValue( ttv.find(studyid.GetTag())->second );
      ds.Insert( studyid.GetAsDataElement() );
      }

    sqi->AddItem( item );
    }

  return true;
}

/*
  (fffe,e000) na "Directory Record" SERIES #=11           # u/l, 1 Item
  #  offset=$748
    (0004,1400) up 1214                                     #   4, 1 OffsetOfTheNextDirectoryRecord
    (0004,1410) US 65535                                    #   2, 1 RecordInUseFlag
    (0004,1420) up 938                                      #   4, 1 OffsetOfReferencedLowerLevelDirectoryEntity
    (0004,1430) CS [SERIES]                                 #   6, 1 DirectoryRecordType
    (0008,0060) CS [CR]                                     #   2, 1 Modality
    (0008,0080) LO (no value available)                     #   0, 0 InstitutionName
    (0008,0081) ST (no value available)                     #   0, 0 InstitutionAddress
    (0008,103e) LO [Computed Radiography]                   #  20, 1 SeriesDescription
    (0008,1050) PN (no value available)                     #   0, 0 PerformingPhysiciansName
    (0020,000e) UI [1.3.6.1.4.1.5962.1.3.65535.4.1119624143.7187.0] #  46, 1 SeriesInstanceUID
    (0020,0011) IS [4]                                      #   2, 1 SeriesNumber
  (fffe,e00d) na "ItemDelimitationItem"                   #   0, 0 ItemDelimitationItem
*/
bool DICOMDIRGenerator::AddSeriesDirectoryRecord()
{
  DataSet &rootds = GetFile().GetDataSet();
  Scanner const & scanner = GetScanner();

  Attribute<0x20,0xe> seriesinstanceuid;
  Scanner::ValuesType seriesinstanceuids = scanner.GetValues( seriesinstanceuid.GetTag() );

  const DataElement &de = rootds.GetDataElement( Tag(0x4,0x1220) );
  //SequenceOfItems * sqi = (SequenceOfItems*)de.GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = de.GetValueAsSQ();

  Scanner::ValuesType::const_iterator it = seriesinstanceuids.begin();
  for( ; it  != seriesinstanceuids.end(); ++it)
    {
    Item item;
    item.SetVLToUndefined();
    DataSet &ds = item.GetNestedDataSet();

    Attribute<0x4,0x1400> offsetofthenextdirectoryrecord = {0};
    ds.Insert( offsetofthenextdirectoryrecord.GetAsDataElement() );
    Attribute<0x4,0x1410> recordinuseflag = {0xFFFF};
    ds.Insert( recordinuseflag.GetAsDataElement() );
    Attribute<0x4,0x1420> offsetofreferencedlowerleveldirectoryentity = {0};
    ds.Insert( offsetofreferencedlowerleveldirectoryentity.GetAsDataElement() );
    Attribute<0x4,0x1430> directoryrecordtype;
    directoryrecordtype.SetValue( "SERIES" );
    ds.Insert( directoryrecordtype.GetAsDataElement() );
    const char *seriesuid = it->c_str();
    if( ! (seriesuid && *seriesuid) )
      {
      const char *fn = scanner.GetFilenameFromTagToValue(seriesinstanceuid.GetTag(), seriesuid);
      gdcmErrorMacro( "Missing Study Instance UID from file: " << fn );
      (void)fn;//warning removal
      return false;
      }
    gdcmAssertAlwaysMacro( seriesuid && *seriesuid );
    seriesinstanceuid.SetValue( seriesuid );
    ds.Insert( seriesinstanceuid.GetAsDataElement() );

    Scanner::TagToValue const &ttv = scanner.GetMappingFromTagToValue(seriesinstanceuid.GetTag(), seriesuid);
    Attribute<0x8,0x60> modality;
    if( ttv.find( modality.GetTag() ) != ttv.end() )
      {
      modality.SetValue( ttv.find(modality.GetTag())->second );
      ds.Insert( modality.GetAsDataElement() );
      }
    Attribute<0x20,0x11> seriesnumber;
    if( ttv.find( seriesnumber.GetTag() ) != ttv.end() )
      {
      seriesnumber.SetValue( atoi(ttv.find(seriesnumber.GetTag())->second) );
      ds.Insert( seriesnumber.GetAsDataElement() );
      }

    sqi->AddItem( item );
    }

  return true;
}

/*
  (fffe,e000) na "Directory Record" IMAGE #=13            # u/l, 1 Item
  #  offset=$1398  refFileID="IMAGES\DXIMAGE"
    (0004,1400) up 0                                        #   4, 1 OffsetOfTheNextDirectoryRecord
    (0004,1410) US 65535                                    #   2, 1 RecordInUseFlag
    (0004,1420) up 0                                        #   4, 1 OffsetOfReferencedLowerLevelDirectoryEntity
    (0004,1430) CS [IMAGE]                                  #   6, 1 DirectoryRecordType
    (0004,1500) CS [IMAGES\DXIMAGE]                         #  14, 2 ReferencedFileID
    (0004,1510) UI =DigitalXRayImageStorageForPresentation  #  28, 1 ReferencedSOPClassUIDInFile
    (0004,1511) UI [1.3.6.1.4.1.5962.1.1.65535.3.1.1119624143.7180.0] #  48, 1 ReferencedSOPInstanceUIDInFile
    (0004,1512) UI =LittleEndianExplicit                    #  20, 1 ReferencedTransferSyntaxUIDInFile
    (0008,0008) CS [ORIGINAL\PRIMARY]                       #  16, 2 ImageType
    (0020,0013) IS [1]                                      #   2, 1 InstanceNumber
    (0028,0004) CS [MONOCHROME2]                            #  12, 1 PhotometricInterpretation
    (0028,0008) IS [1]                                      #   2, 1 NumberOfFrames
    (0050,0004) CS (no value available)                     #   0, 0 CalibrationImage
  (fffe,e00d) na "ItemDelimitationItem"                   #   0, 0 ItemDelimitationItem
*/
bool DICOMDIRGenerator::AddImageDirectoryRecord()
{
  DataSet &rootds = GetFile().GetDataSet();
  Scanner const & scanner = GetScanner();

  const Attribute<0x8,0x18> sopinstanceuid = { "" };
  Scanner::ValuesType sopinstanceuids = scanner.GetValues( sopinstanceuid.GetTag() );
  //unsigned int ninstance = sopinstanceuids.size();

  const DataElement &de = rootds.GetDataElement( Tag(0x4,0x1220) );
  //SequenceOfItems * sqi = (SequenceOfItems*)de.GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = de.GetValueAsSQ();

  Scanner::ValuesType::const_iterator it = sopinstanceuids.begin();
  Filename rootdir = Internals->rootdir.c_str();
  const char *rd = rootdir.ToWindowsSlashes();
  size_t strlen_rd = strlen( rd );
  for( ; it  != sopinstanceuids.end(); ++it)
    {
    Item item;
    item.SetVLToUndefined();
    DataSet &ds = item.GetNestedDataSet();

    Attribute<0x4,0x1400> offsetofthenextdirectoryrecord = {0};
    ds.Insert( offsetofthenextdirectoryrecord.GetAsDataElement() );
    Attribute<0x4,0x1410> recordinuseflag = {0xFFFF};
    ds.Insert( recordinuseflag.GetAsDataElement() );
    Attribute<0x4,0x1420> offsetofreferencedlowerleveldirectoryentity = {0};
    ds.Insert( offsetofreferencedlowerleveldirectoryentity.GetAsDataElement() );
    Attribute<0x4,0x1430> directoryrecordtype;
    directoryrecordtype.SetValue( "IMAGE" );
    ds.Insert( directoryrecordtype.GetAsDataElement() );

    const char *sopuid = it->c_str();
    Scanner::TagToValue const &ttv = scanner.GetMappingFromTagToValue(sopinstanceuid.GetTag(), sopuid);
    Attribute<0x0004,0x1500> referencedfileid;
    const char *fn_str = scanner.GetFilenameFromTagToValue(sopinstanceuid.GetTag(), sopuid);
    referencedfileid.SetNumberOfValues( 1 );
    Filename fn = fn_str;
    std::string relative = fn.ToWindowsSlashes();
    std::string::size_type l = relative.find( rd );
    if( l != std::string::npos )
      {
      assert( l == 0 ); // FIXME
      relative.replace( l, strlen_rd, "" );
      fn = relative.c_str() + 1;
      }
    referencedfileid.SetValue( fn.ToWindowsSlashes() );
    ds.Insert( referencedfileid.GetAsDataElement() );
    Attribute<0x0004,0x1510> referencedsopclassuidinfile;
    Attribute<0x8,0x16> sopclassuid;
    if( ttv.find( sopclassuid.GetTag() ) != ttv.end() )
      {
      referencedsopclassuidinfile.SetValue( ttv.find(sopclassuid.GetTag())->second );
      }
    ds.Insert( referencedsopclassuidinfile.GetAsDataElement() );
    Attribute<0x0004,0x1511> referencedsopinstanceuidinfile;
    if( ttv.find( sopinstanceuid.GetTag() ) != ttv.end() )
      {
      referencedsopinstanceuidinfile.SetValue( ttv.find(sopinstanceuid.GetTag())->second );
      }
    ds.Insert( referencedsopinstanceuidinfile.GetAsDataElement() );
    Attribute<0x0004,0x1512> referencedtransfersyntaxuidinfile;
    Attribute<0x2,0x10> transfersyntaxuid;
    if( ttv.find( transfersyntaxuid.GetTag() ) != ttv.end() )
      {
      referencedtransfersyntaxuidinfile.SetValue( ttv.find(transfersyntaxuid.GetTag())->second );
      }
    ds.Insert( referencedtransfersyntaxuidinfile.GetAsDataElement() );

    Attribute<0x20,0x13> instancenumber = { 0 };
    if( ttv.find( instancenumber.GetTag() ) != ttv.end() )
      {
      instancenumber.SetValue( atoi(ttv.find(instancenumber.GetTag())->second) );
      }
    ds.Insert( instancenumber.GetAsDataElement() );

    Attribute<0x8,0x8> imagetype;
    //Scanner::ValuesType imagetypes = scanner.GetValues( imagetype.GetTag() );
    //Scanner::ValuesType::const_iterator it = imagetypes.begin();
    //assert( imagetypes.size() == 1 );
    //imagetype.SetNumberOfValues( 1 );
    //imagetype.SetValue( it->c_str() );
    //ds.Replace( imagetype.GetAsDataElement() );
    DataElement de2( imagetype.GetTag() );
    de2.SetVR( imagetype.GetVR() );
    if( ttv.find( imagetype.GetTag() ) != ttv.end() )
      {
      const char *v = ttv.find(imagetype.GetTag())->second;
      VL::Type strlenV = (VL::Type)strlen(v);
      de2.SetByteValue( v, strlenV );
      }
    ds.Insert( de2 );

    sqi->AddItem( item );
    }

  return true;
}

DICOMDIRGenerator::DICOMDIRGenerator(  )
{
  Internals = new DICOMDIRGeneratorInternal;
}

DICOMDIRGenerator::~DICOMDIRGenerator(  )
{
  delete Internals;
}

static bool IsCompatibleWithISOIEC9660MediaFormat(const char *filename)
{
  if(!filename) return false;
  // (0004,1500) CS [IMG001]                                 #   6, 1 ReferencedFileID
  // <entry group="0004" element="1500" vr="CS" vm="1-8" name="Referenced File ID"/>
  Attribute< 0x4, 0x1500 > at;
  DataElement de( at.GetTag() );
  std::string copy = filename;
  if( copy.size() % 2 )
    {
    copy.push_back( ' ' );
    }
  VL::Type copySize = (VL::Type)copy.size();
  de.SetByteValue( copy.c_str(), copySize ) ;
  at.SetFromDataElement( de );
  unsigned int n = at.GetNumberOfValues();
  // A volume may have at most 8 levels of directories, where the root
  // directory is defined as level 1.
  if( n > 8 )
    {
    gdcmDebugMacro( "8 Levels of directories" );
    return false;
    }

  for( unsigned int i = 0; i < n; ++i)
    {
    CodeString cs = at.GetValue( i );
    if( !cs.IsValid() || cs.Size() > 8 )
      {
      gdcmDebugMacro( "Problem with CS: " << cs );
      return false;
      }
    }
  return true;
}

void DICOMDIRGenerator::SetFilenames( FilenamesType const & fns )
{
  Internals->fns = fns;
}

void DICOMDIRGenerator::SetRootDirectory( FilenameType const & root )
{
  Internals->rootdir = root;
}

bool DICOMDIRGenerator::Generate()
{
  Scanner &scanner = GetScanner();
  // <entry group="0002" element="0010" vr="UI" vm="1" name="Transfer Syntax UID"/>
  scanner.AddTag( Tag(0x02,0x10) );
  // <entry group="0010" element="0010" vr="PN" vm="1" name="Patient's Name"/>
  scanner.AddTag( Tag(0x10,0x10) );
  // <entry group="0010" element="0020" vr="LO" vm="1" name="Patient ID"/>
  scanner.AddTag( Tag(0x10,0x20) );
  // <entry group="0008" element="0060" vr="CS" vm="1" name="Modality"/>
  scanner.AddTag( Tag(0x8,0x60) );
  // <entry group="0020" element="0011" vr="IS" vm="1" name="Series Number"/>
  scanner.AddTag( Tag(0x20,0x11) );
  // <entry group="0008" element="0018" vr="UI" vm="1" name="SOP Instance UID"/>
  scanner.AddTag( Tag(0x8,0x18) );
  // <entry group="0008" element="0020" vr="DA" vm="1" name="Study Date"/>
  scanner.AddTag( Tag(0x8,0x20) );
  // <entry group="0008" element="0030" vr="TM" vm="1" name="Study Time"/>
  scanner.AddTag( Tag(0x8,0x30) );
  // <entry group="0008" element="1030" vr="LO" vm="1" name="Study Description"/>
  scanner.AddTag( Tag(0x8,0x1030) );
  // <entry group="0008" element="0050" vr="SH" vm="1" name="Accession Number"/>
  scanner.AddTag( Tag(0x8,0x50) );
  // <entry group="0020" element="0013" vr="IS" vm="1" name="Instance Number"/>
  scanner.AddTag( Tag(0x20,0x13) );
  // <entry group="0020" element="000d" vr="UI" vm="1" name="Study Instance UID"/>
  scanner.AddTag( Tag(0x20,0xd) );
  // <entry group="0020" element="0010" vr="SH" vm="1" name="Study ID"/>
  scanner.AddTag( Tag(0x20,0x10) );
  // <entry group="0020" element="000e" vr="UI" vm="1" name="Series Instance UID"/>
  scanner.AddTag( Tag(0x20,0xe) );
  // <entry group="0028" element="0004" vr="CS" vm="1" name="Photometric Interpretation"/>
  scanner.AddTag( Tag(0x28,0x4) );
  // <entry group="0028" element="0008" vr="IS" vm="1" name="Number of Frames"/>
  scanner.AddTag( Tag(0x28,0x8) );
  // <entry group="0050" element="0004" vr="CS" vm="1" name="Calibration Image"/>
  scanner.AddTag( Tag(0x50,0x4) );
  // <entry group="0010" element="0030" vr="DA" vm="1" name="Patient's Birth Date"/>
  scanner.AddTag( Tag(0x10,0x30) );
  // <entry group="0010" element="0040" vr="CS" vm="1" name="Patient's Sex"/>
  scanner.AddTag( Tag(0x10,0x40) );
  // <entry group="0008" element="0080" vr="LO" vm="1" name="Institution Name"/>
  scanner.AddTag( Tag(0x8,0x80) );
  // <entry group="0008" element="0016" vr="UI" vm="1" name="SOP Class UID"/>
  scanner.AddTag( Tag(0x8,0x16) );
  // <entry group="0002" element="0010" vr="UI" vm="1" name="Transfer Syntax UID"/>
  scanner.AddTag( Tag(0x2,0x10) );
  // <entry group="0008" element="0008" vr="CS" vm="2-n" name="Image Type"/>
  scanner.AddTag( Tag(0x8,0x8) );

  FilenamesType const &filenames = Internals->fns;
  Filename rootdir = Internals->rootdir.c_str();
  const char *rd = rootdir.ToWindowsSlashes();
  size_t strlen_rd = strlen( rd );

  // Let's check that filenames are ok for iso9660 + compatible with VR:CS
{
  FilenamesType::const_iterator it = filenames.begin();
  for( ; it != filenames.end(); ++it )
    {
    Filename fn = it->c_str();
    const char *f = fn.ToWindowsSlashes();
    std::string relative = f;
    std::string::size_type l = relative.find( rd );
    if( l != std::string::npos )
      {
      assert( l == 0 ); // FIXME
      relative.replace( l, strlen_rd, "" );
      f = relative.c_str() + 1;
      }
    if( !IsCompatibleWithISOIEC9660MediaFormat( f ) )
      {
      gdcmErrorMacro( "Invalid file name: " << f );
      return false;
      }
    }
}

  if( !scanner.Scan( filenames ) )
    {
    return false;
    }

  //scanner.Print( std::cout );

  Scanner::ValuesType vt = scanner.GetValues( Tag(0x2,0x10) );
  Scanner::ValuesType vtref;
  vtref.insert( TransferSyntax::GetTSString( TransferSyntax::ExplicitVRLittleEndian ) );
  if( vt == vtref )
    {
    // All files are ExplicitVRLittleEndian which is required for DICOMDIR
    }
  else
    {
    gdcmErrorMacro( "Found Transfer Syntax not ExplicitVRLittleEndian." );
    return false;
    }

  // (0004,1220) SQ (Sequence with undefined length #=8)     # u/l, 1 DirectoryRecordSequence

  DataSet &ds = GetFile().GetDataSet();

  Attribute<0x4,0x1130> filesetid;
  filesetid.SetValue( Internals->FileSetID.c_str() );
  ds.Insert( filesetid.GetAsDataElement() );

  CodeString cs = filesetid.GetValue();
  if( !cs.IsValid() )
    {
    gdcmErrorMacro( "Invalid File Set ID: " << filesetid.GetValue() );
    return false;
    }

  Attribute<0x4,0x1200> offsetofthefirstdirectoryrecordoftherootdirectoryentity = {0};
  ds.Insert( offsetofthefirstdirectoryrecordoftherootdirectoryentity.GetAsDataElement() );
  Attribute<0x4,0x1202> offsetofthelastdirectoryrecordoftherootdirectoryentity = { 0 };
  ds.Insert( offsetofthelastdirectoryrecordoftherootdirectoryentity.GetAsDataElement() );
  Attribute<0x4,0x1212> filesetconsistencyflag = {0};
  ds.Insert( filesetconsistencyflag.GetAsDataElement() );


  DataElement de_drs( Tag(0x4,0x1220) ); // DirectoryRecordSequence

  SequenceOfItems * sqi0 = new SequenceOfItems;
  de_drs.SetVR( VR::SQ );
  de_drs.SetValue( *sqi0 );
  de_drs.SetVLToUndefined();

  ds.Insert( de_drs );

  bool b;
  b = AddPatientDirectoryRecord();
  if( !b ) return false;
  b = AddStudyDirectoryRecord();
  if( !b ) return false;
  b = AddSeriesDirectoryRecord();
  if( !b ) return false;
  b = AddImageDirectoryRecord();
  if( !b ) return false;

/*
The DICOMDIR File shall use the Explicit VR Little Endian Transfer Syntax (UID=1.2.840.10008.1.2.1) to
encode the Media Storage Directory SOP Class. The DICOMDIR File shall comply with the DICOM File
Format specified in Section 7 of this Standard. In particular the:
a. SOP Class UID in the File Meta Information (header of the DICOMDIR File) shall have the
Value specified in PS 3.4 of this Standard for the Media Storage Directory SOP Class;
b. SOP Instance UID in the File Meta Information (header of the DICOMDIR File) shall contain
the File-set UID Value. The File-set UID is assigned by the Application Entity which created
the File-set (FSC role, see Section 8.3) with zero or more DICOM Files. This File-set UID
Value shall not be changed by any other Application Entities reading or updating the content of
the File-set.
*/
  FileMetaInformation &h = GetFile().GetHeader();
  Attribute<0x2,0x2> at1;
  MediaStorage ms = MediaStorage::MediaStorageDirectoryStorage;
  const char* msstr = MediaStorage::GetMSString(ms);
  at1.SetValue( msstr );
  h.Insert( at1.GetAsDataElement() );

  Attribute<0x2,0x3> at2;
  UIDGenerator uid;
  const char *mediastoragesopinstanceuid = uid.Generate();
  if( !UIDGenerator::IsValid( mediastoragesopinstanceuid ) )
    {
    return 1;
    }
  at2.SetValue( mediastoragesopinstanceuid );
  h.Insert( at2.GetAsDataElement() );

  TransferSyntax ts = TransferSyntax::ExplicitVRLittleEndian;
  h.SetDataSetTransferSyntax( ts );

  //std::cout << ds << std::endl;
  //std::cout << h << std::endl;


  /* Very important step it should be the *VERY* last one */
  // We need to compute all offset, which can be only done when all attributes have been inserted.
  // Let's start with offsetofthefirstdirectoryrecordoftherootdirectoryentity :
  h.FillFromDataSet( ds );
  VL fmi_len = h.GetFullLength();
  VL fmi_len_offset = 0;
{
  DataSet::ConstIterator it = ds.Begin();
  for(; it != ds.End() && it->GetTag() != Tag(0x0004,0x1220); ++it)
    {
    const DataElement &detmp = *it;
    fmi_len_offset += detmp.GetLength<ExplicitDataElement>();
    }
  // Now add the partial length for attribute 0004,1220:
  fmi_len_offset += it->GetTag().GetLength();
  fmi_len_offset += it->GetVR().GetLength();
  fmi_len_offset += it->GetVR().GetLength();
}
  //std::cerr << fmi_len << " and " << fmi_len_offset << std::endl;
  offsetofthefirstdirectoryrecordoftherootdirectoryentity.SetValue( fmi_len + fmi_len_offset );
  ds.Replace( offsetofthefirstdirectoryrecordoftherootdirectoryentity.GetAsDataElement() );

  VL fmi_len_offset2 = 0;
{
  DataSet::ConstIterator it = ds.Begin();
  for( ; it != ds.End() && it->GetTag() <= Tag(0x0004,0x1220); ++it)
    {
    const DataElement &detmp = *it;
    fmi_len_offset2 += detmp.GetLength<ExplicitDataElement>();
    }
}

{
  //const DataElement &de_drs = ds.GetDataElement( Tag(0x4,0x1220) ); // DirectoryRecordSequence
  SmartPointer<SequenceOfItems> sqi = de_drs.GetValueAsSQ();
  SequenceOfItems::SizeType n = sqi->GetNumberOfItems();
  const Item &item = sqi->GetItem( n ); // last item
  VL sub = item.GetLength<ExplicitDataElement>();
  // Let's substract item length as well as the item sequence delimiter end (tag + vl => 8)
  offsetofthelastdirectoryrecordoftherootdirectoryentity.SetValue( fmi_len + fmi_len_offset2 - sub - 8 );

  ds.Replace( offsetofthelastdirectoryrecordoftherootdirectoryentity.GetAsDataElement() );

  TraverseDirectoryRecords(offsetofthefirstdirectoryrecordoftherootdirectoryentity.GetValue() );
}

  return true;
}

void DICOMDIRGenerator::SetFile(const File& f)
{
  Internals->F = f;
}

File &DICOMDIRGenerator::GetFile()
{
  return *Internals->F;
}

Scanner &DICOMDIRGenerator::GetScanner()
{
  return Internals->scanner;
}

SequenceOfItems *DICOMDIRGenerator::GetDirectoryRecordSequence()
{
  DataSet &ds = GetFile().GetDataSet();
  const DataElement &de = ds.GetDataElement( Tag(0x4,0x1220) );
  //SequenceOfItems * sqi = (SequenceOfItems*)de.GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = de.GetValueAsSQ();
  return sqi;
}

const char *DICOMDIRGenerator::ComputeFileID(const char *input)
{
  assert( 0 ); (void)input;
  return NULL;
}

void DICOMDIRGenerator::SetDescriptor( const char *d )
{
  Internals->FileSetID = d;
}

} // end namespace gdcm
