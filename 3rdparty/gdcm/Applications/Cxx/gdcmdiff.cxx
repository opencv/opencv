/*=========================================================================

  Program: gdcmdiff for GDCM (Grassroots DICOM)

  Copyright (c) 2011 Andy Buckle
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmReader.h"
#include "gdcmAttribute.h"
#include "gdcmDataSet.h"
#include "gdcmDicts.h"
#include "gdcmDict.h"
#include "gdcmGlobal.h"
#include "gdcmCSAHeader.h"
#include "gdcmPrivateTag.h"

#include <iostream>

static void usage();
static void difference_of_datasets(const gdcm::DataSet& ds1, const gdcm::DataSet& ds2, int depthSQ);
static void display_element(std::ostream& os, const gdcm::DataElement& de,
  const gdcm::DictEntry& dictentry, const char *note, int depthSQ);
static void underline(int depthSQ);
static void difference_of_sequences(const gdcm::DataElement& sqde1,
  const gdcm::DataElement& sqde2, const gdcm::DictEntry& dictentry, int depthSQ);

// previous declaration of 'int truncate(const char*, __off_t)'
static uint32_t Truncate=30; // trim dumped string values to this number of chars. zero means no trimming.
static std::stringstream sq_disp; // store SQ output while recursing through: only displayed if a difference is found within SQ

int main( int argc, const char* argv[] )
{
  // Check number of args
  if (3 > argc)
    {
    std::cerr << "Must supply the filenames of 2 DICOM files\n" << std::endl;
    usage();
    return 1;
    }
  // Check last 2 args are readable DICOM files
  gdcm::Reader reader1, reader2;
  reader1.SetFileName( argv[argc-2] );
  reader2.SetFileName( argv[argc-1] );
  if( !reader1.Read() || !reader2.Read() )
    {
    std::cerr << "At least one of the DICOM files could not be read: " <<
      '\"' << argv[argc-2] << "\", \"" << argv[argc-1] << '\"' << std::endl;
    return 1;
    }
  // Parse other args
  bool include_meta = false;
  for(int i=1; i<(argc-2) ;i++)
    {
    std::string arg = std::string(argv[i]);
    if ("-h" == arg || "--help" == arg)
      {
      usage();
      return 0;
      }
    else if ("-m" == arg || "--meta" == arg)
      {
      include_meta = true;
      }
    else if ("-t" == arg || "--truncate" == arg)
      {
      Truncate = atoi(argv[++i]);
      }
    else
      {
      std::cerr << "Warning: command argument not understood: "
        << arg << std::endl;
      usage();
      return 1;
      }
    }
  // Start comparison
  const gdcm::File &file1 = reader1.GetFile();
  const gdcm::File &file2 = reader2.GetFile();

  const gdcm::FileMetaInformation &hds1 = file1.GetHeader();
  const gdcm::FileMetaInformation  &hds2 = file2.GetHeader();

  const gdcm::DataSet &ds1 = file1.GetDataSet();
  const gdcm::DataSet  &ds2 = file2.GetDataSet();

  if(include_meta)
    {
    difference_of_datasets(hds1, hds2, 0);
    }
  difference_of_datasets(ds1, ds2, 0);

  return 0;
}

static void usage()
{
  std::cout <<
    "Usage: gdcmdiff [OPTIONS] DICOM_FILE1 DICOM_FILE2\n\n"
    "  -h      --help          (This) help and exit.\n"
    "  -m      --meta          Compare metainformation. Default is off.\n"
    "  -t <n>  --truncate <n>  String values trimmed to n characters.\n"
    "                          0 means no trimmming. Default 30." << std::endl;
}

static void difference_of_datasets(const gdcm::DataSet& ds1, const gdcm::DataSet& ds2, int depthSQ)
{
  gdcm::DataSet::ConstIterator it1 = ds1.Begin();
  gdcm::DataSet::ConstIterator it2 = ds2.Begin();
  do {
    // find lowest value tag, being careful not to pick one for an iterator that has finished
    const gdcm::Tag tag1 = (it1!=ds1.End()) ? it1->GetTag() : gdcm::Tag(0xffff,0xffff);
    const gdcm::Tag tag2 = (it2!=ds2.End()) ? it2->GetTag() : gdcm::Tag(0xffff,0xffff);
    gdcm::Tag tag= (tag1<tag2) ? tag1 : tag2 ;
    // get VR for tag
    const gdcm::Dicts &dicts = gdcm::Global::GetInstance().GetDicts();
    const gdcm::DictEntry &dictentry = dicts.GetDictEntry(tag);
    // check for tags that are only in one of the files
    if (!ds1.FindDataElement(tag))
      {
      std::cout << sq_disp.str();
      display_element(std::cout, ds2.GetDataElement(tag), dictentry, "[only file 2]", depthSQ);
      underline(depthSQ);
      }
    if (!ds2.FindDataElement(tag))
      {
      std::cout << sq_disp.str();
      display_element(std::cout,ds1.GetDataElement(tag), dictentry, "[only file 1]", depthSQ);
      underline(depthSQ);
      }
    // compare values and increment iterator(s)
    if(tag1==tag2)
      {
      if(*it1!=*it2)
        {// TODO: can this logical iterator spaghetti be tidied up?
        if (dictentry.GetVR() & gdcm::VR::SQ)
          {
          difference_of_sequences(ds1.GetDataElement(tag), ds2.GetDataElement(tag), dictentry, depthSQ+1);
          }
        else if (dictentry.GetVR() & gdcm::VR::VRASCII)
          {
          std::cout << sq_disp.str();
          display_element(std::cout,ds1.GetDataElement(tag), dictentry, "[from file 1]", depthSQ);
          display_element(std::cout,ds2.GetDataElement(tag), dictentry, "[from file 2]", depthSQ);
          underline(depthSQ);
          }
        else
          {
          std::cout << sq_disp.str();
          display_element(std::cout,ds1.GetDataElement(tag), dictentry, "[elem differ]", depthSQ);
          underline(depthSQ);
          }
        }
      if (it1 != ds1.End()) it1++; // unless an iterator has finished, inc it
      if (it2 != ds2.End()) it2++;
      }
    else
      { // if tags out of sync, only increment the iterator that is behind. this happens when a tag is only in one set
      if (it1 == ds1.End())
        {
        it2++;
        }
      else if (it2 == ds2.End())
        {
        it1++;
        }
      else
        {
        if (tag1 < tag2)
          {
          it1++;
          }
        else
          {
          it2++;
          }
        }
      }
  } while (it1 != ds1.End() || it2 != ds2.End() ); // with && we might miss some off the end that are only in one dataset
}

static void display_element(std::ostream& os, const gdcm::DataElement& de,
  const gdcm::DictEntry& dictentry, const char * note, int depthSQ)
 {
  const gdcm::VR & vr = dictentry.GetVR();
  const gdcm::VR & filevr = de.GetVR();
  os << std::string(depthSQ, ' '); //indent for SQ
  os << de.GetTag() << ' ' << filevr << ' ' << note;
  if (vr & gdcm::VR::VRBINARY)
    {
    os << " binary";
    }
  else if (vr & gdcm::VR::VRASCII)
    {
    if (de.GetByteValue() != NULL)
      { // is this OK? it worked when de was a pointer, without the != NULL
      gdcm::VL vl = de.GetByteValue()->GetLength();
      // error: operands to ?: have different types 'gdcm::VL' and 'uint32_t'
      uint32_t val_vl = vl;
      uint32_t trimto = (Truncate > val_vl ) ? val_vl : Truncate;
      if (0 == Truncate)
        {
        trimto = de.GetByteValue()->GetLength();
        }
      os << " [" << std::string(de.GetByteValue()->GetPointer(),trimto) << ']';
      }
    else
      {
      os << " null";
      }
    }
  else
    {
    os << " VR unknown";
    }
  os << " # " << dictentry.GetName() << std::endl  ;
}

static void underline(int depthSQ)
{
  std::cout << std::string(depthSQ, ' '); //indent for SQ
  std::cout << "               -------------" << std::endl;
}

static void difference_of_sequences(const gdcm::DataElement& sqde1,
  const gdcm::DataElement& sqde2, const gdcm::DictEntry& dictentry, int depthSQ)
{
  gdcm::SmartPointer<gdcm::SequenceOfItems> sqi1 = sqde1.GetValueAsSQ();
  gdcm::SmartPointer<gdcm::SequenceOfItems> sqi2 = sqde2.GetValueAsSQ();
  size_t n1 = sqi1->GetNumberOfItems();
  size_t n2 = sqi2->GetNumberOfItems();
  size_t n = (n1 < n2) ? n1 : n2 ;
  std::stringstream sq_note;
  if (n1 != n2)
    {
    sq_note << "[sequence, file 1 has " << n1 << " datasets, file 2 has " << n2
      << " datasets]";
    }
  else
    {
    sq_note << "[sequence]";
    }
  display_element(sq_disp, sqde1, dictentry, sq_note.str().c_str(),depthSQ);
  for(size_t i=1; i <= n; i++)
    {
    difference_of_datasets( sqi1->GetItem(i).GetNestedDataSet(),
      sqi2->GetItem(i).GetNestedDataSet(), depthSQ+1);
    }
  sq_disp.str(std::string()); // clear stringstream
}
