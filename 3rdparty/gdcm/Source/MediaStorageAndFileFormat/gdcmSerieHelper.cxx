/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSerieHelper.h"

#include "gdcmStringFilter.h"
#include "gdcmDirectory.h"
#include "gdcmIPPSorter.h"
#include "gdcmImageReader.h"
#include "gdcmImageHelper.h"
#include "gdcmTrace.h"

namespace gdcm
{

SerieHelper::SerieHelper()
{
  Trace::WarningOff();
  UseSeriesDetails = false;
  Clear();
  UserLessThanFunction = 0;
  DirectOrder = true;
  //LoadMode = 0;
}

SerieHelper::~SerieHelper()
{
  Clear();
}

void SerieHelper::AddRestriction(uint16_t group, uint16_t elem, std::string const &value, int op)
{
  Rule r;
  r.group = group;
  r.elem  = elem;
  r.value = value;
  r.op    = op;
  Restrictions.push_back( r );
}

void SerieHelper::AddRestriction(const std::string & tag)
{
  Tag t;
  t.ReadFromPipeSeparatedString(tag.c_str());
  AddRestriction( Tag(t.GetGroup(), t.GetElement()) );
}

void SerieHelper::AddRestriction(const Tag& tag)
{
  Rule r;
  r.group = tag.GetGroup();
  r.elem  = tag.GetElement();
  Refine.push_back( r );
}

void SerieHelper::SetUseSeriesDetails( bool useSeriesDetails )
{
  UseSeriesDetails = useSeriesDetails;
}

void SerieHelper::CreateDefaultUniqueSeriesIdentifier()
{
  // If the user requests, additional information can be appended
  // to the SeriesUID to further differentiate volumes in the DICOM
  // objects being processed.

  // 0020 0011 Series Number
  // A scout scan prior to a CT volume scan can share the same
  //   SeriesUID, but they will sometimes have a different Series Number
  AddRestriction( Tag(0x0020, 0x0011) );
  // 0018 0024 Sequence Name
  // For T1-map and phase-contrast MRA, the different flip angles and
  //   directions are only distinguished by the Sequence Name
  AddRestriction( Tag(0x0018, 0x0024) );
  // 0018 0050 Slice Thickness
  // On some CT systems, scout scans and subsequence volume scans will
  //   have the same SeriesUID and Series Number - YET the slice
  //   thickness will differ from the scout slice and the volume slices.
  AddRestriction( Tag(0x0018, 0x0050) );
  // 0028 0010 Rows
  // If the 2D images in a sequence don't have the same number of rows,
  // then it is difficult to reconstruct them into a 3D volume.
  AddRestriction( Tag(0x0028, 0x0010));
  // 0028 0011 Columns
  // If the 2D images in a sequence don't have the same number of columns,
  // then it is difficult to reconstruct them into a 3D volume.
  AddRestriction( Tag(0x0028, 0x0011));
}

void SerieHelper::Clear()
{
  // For all the 'Single SerieUID' Filesets that may already exist
  FileList *l = GetFirstSingleSerieUIDFileSet();
  while (l)
    {
    // For all the File of a File set
    for (FileList::iterator it  = l->begin();
      it != l->end();
      ++it)
      {
      //delete *it; // remove each entry
      }
    l->clear();
    delete l;     // remove the container
    l = GetNextSingleSerieUIDFileSet();
    }
  // Need to clear that too:
  SingleSerieUIDFileSetHT.clear();
}

void SerieHelper::SetDirectory(std::string const &dir, bool recursive)
{
  Directory dirList;
  unsigned int nfiles = dirList.Load(dir, recursive); (void)nfiles;

  Directory::FilenamesType const &filenames = dirList.GetFilenames();
  for( Directory::FilenamesType::const_iterator it = filenames.begin();
    it != filenames.end(); ++it)
    {
    AddFileName( *it );
    }
}

void SerieHelper::AddFileName(std::string const &filename)
{
  // Only accept DICOM file containing Image (Pixel Data element):
  ImageReader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    gdcmWarningMacro("Could not read file: " << filename );
    }
  else
    {
    SmartPointer<FileWithName> f = new FileWithName( reader.GetFile() );
    f->filename = filename;
    (void)AddFile( *f /*reader.GetFile()*/ ); // discard return value
    }
}

bool CompareDicomString(const std::string &s1, const char *s2, int op)
{
  // s2 is the string from the DICOM reference e.g. : 'MONOCHROME1'
  std::string s1_even = s1; //Never change input parameter
  std::string s2_even = /*DicomString(*/ s2 ;
  assert( s2_even.size() % 2 == 0 );
  if ( s1_even[s1_even.size()-1] == ' ' )
    {
    s1_even[s1_even.size()-1] = '\0'; //replace space character by null
    }
  switch (op)
    {
  case GDCM_EQUAL :
    return s1_even == s2_even;
  case GDCM_DIFFERENT :
    return s1_even != s2_even;
  case GDCM_GREATER :
    return s1_even >  s2_even;
  case GDCM_GREATEROREQUAL :
    return s1_even >= s2_even;
  case GDCM_LESS :
    return s1_even <  s2_even;
  case GDCM_LESSOREQUAL :
    return s1_even <= s2_even;
  default :
    gdcmDebugMacro(" Wrong operator : " << op);
    return false;
    }
}
bool SerieHelper::AddFile(FileWithName &header)
{
  StringFilter sf;
  sf.SetFile( header );
  int allrules = 1;
  // First step the user has defined a set of rules for the DICOM
  // he is looking for.
  // make sure the file correspond to his set of rules:

  std::string s;
  for(SerieRestrictions::iterator it2 = Restrictions.begin();
    it2 != Restrictions.end();
    ++it2)
    {
    const Rule &r = *it2;
    //s = header->GetEntryValue( r.group, r.elem );
    s = sf.ToString( Tag(r.group,r.elem) );
    if ( !CompareDicomString(s, r.value.c_str(), r.op) )
      {
      // Argh ! This rule is unmatched; let's just quit
      allrules = 0;
      break;
      }
    }

  if ( allrules ) // all rules are respected:
    {
    // Allright! we have a found a DICOM that matches the user expectation.
    // Let's add it to the specific 'id' which by default is uid (Serie UID)
    // but can be `refined` by user with more paramater (see AddRestriction(g,e))

    std::string id = CreateUniqueSeriesIdentifier( &header );
    // if id == GDCM_UNFOUND then consistently we should find GDCM_UNFOUND
    // no need here to do anything special

    if ( SingleSerieUIDFileSetHT.count(id) == 0 )
      {
      gdcmDebugMacro(" New Serie UID :[" << id << "]");
      // create a std::list in 'id' position
      SingleSerieUIDFileSetHT[id] = new FileList;
      }
    // Current Serie UID and DICOM header seems to match add the file:
    SingleSerieUIDFileSetHT[id]->push_back( header );
    }
  else
    {
    // one rule not matched, tell user:
    return false;
    }
  return true;
}

FileList *SerieHelper::GetFirstSingleSerieUIDFileSet()
{
  ItFileSetHt = SingleSerieUIDFileSetHT.begin();
  if ( ItFileSetHt != SingleSerieUIDFileSetHT.end() )
    return ItFileSetHt->second;
  return NULL;
}

FileList *SerieHelper::GetNextSingleSerieUIDFileSet()
{
  //gdcmAssertMacro (ItFileSetHt != SingleSerieUIDFileSetHT.end());

  ++ItFileSetHt;
  if ( ItFileSetHt != SingleSerieUIDFileSetHT.end() )
    return ItFileSetHt->second;
  return NULL;
}

bool SerieHelper::UserOrdering(FileList *fileList)
{
  std::sort(fileList->begin(), fileList->end(), SerieHelper::UserLessThanFunction);
  if (!DirectOrder)
    {
    std::reverse(fileList->begin(), fileList->end());
    }
  return true;
}

namespace details {
bool MyFileNameSortPredicate(const SmartPointer<FileWithName>& d1, const SmartPointer<FileWithName>& d2)
{
  return d1->filename < d2->filename;
}
}

bool SerieHelper::FileNameOrdering( FileList *fileList )
{
  std::sort(fileList->begin(), fileList->end(), details::MyFileNameSortPredicate);

  return true;
}

bool SerieHelper::ImagePositionPatientOrdering( FileList *fileList )
{
  //iop is calculated based on the file file
  std::vector<double> cosines;
  double normal[3] = {};
  std::vector<double> ipp;
  double dist;
  double min = 0, max = 0;
  bool first = true;

  std::multimap<double,SmartPointer<FileWithName> > distmultimap;
  // Use a multimap to sort the distances from 0,0,0
  for ( FileList::const_iterator
    it = fileList->begin();
    it != fileList->end(); ++it )
    {
    if ( first )
      {
      //(*it)->GetImageOrientationPatient( cosines );
      cosines = ImageHelper::GetDirectionCosinesValue( **it );

      // You only have to do this once for all slices in the volume. Next,
      // for each slice, calculate the distance along the slice normal
      // using the IPP ("Image Position Patient") tag.
      // ("dist" is initialized to zero before reading the first slice) :
      normal[0] = cosines[1]*cosines[5] - cosines[2]*cosines[4];
      normal[1] = cosines[2]*cosines[3] - cosines[0]*cosines[5];
      normal[2] = cosines[0]*cosines[4] - cosines[1]*cosines[3];

      ipp = ImageHelper::GetOriginValue( **it );
      //ipp[0] = (*it)->GetXOrigin();
      //ipp[1] = (*it)->GetYOrigin();
      //ipp[2] = (*it)->GetZOrigin();

      dist = 0;
      for ( int i = 0; i < 3; ++i )
        {
        dist += normal[i]*ipp[i];
        }

      distmultimap.insert(std::pair<const double,SmartPointer<FileWithName> >(dist, *it));

      max = min = dist;
      first = false;
      }
    else
      {
      ipp = ImageHelper::GetOriginValue( **it );
      //ipp[0] = (*it)->GetXOrigin();
      //ipp[1] = (*it)->GetYOrigin();
      //ipp[2] = (*it)->GetZOrigin();

      dist = 0;
      for ( int i = 0; i < 3; ++i )
        {
        dist += normal[i]*ipp[i];
        }

      distmultimap.insert(std::pair<const double,SmartPointer<FileWithName> >(dist, *it));

      min = (min < dist) ? min : dist;
      max = (max > dist) ? max : dist;
      }
    }

  // Find out if min/max are coherent
  if ( min == max )
    {
    gdcmWarningMacro("Looks like all images have the exact same image position"
      << ". No PositionPatientOrdering sort performed" );
    return false;
    }

  // Check to see if image shares a common position
  bool ok = true;
  for (std::multimap<double, SmartPointer<FileWithName> >::iterator it2 = distmultimap.begin();
    it2 != distmultimap.end();
    ++it2)
    {
    if (distmultimap.count((*it2).first) != 1)
      {
      gdcmErrorMacro("File: "
        //<< ((*it2).second->GetFileName())
        << " Distance: "
        << (*it2).first
        << " position is not unique");

      ok = false;
      }
    }
  if (!ok)
    {
    return false;
    }

  fileList->clear();  // doesn't delete list elements, only nodes

  if (DirectOrder)
    {
    for (std::multimap<double, SmartPointer<FileWithName> >::iterator it3 = distmultimap.begin();
      it3 != distmultimap.end();
      ++it3)
      {
      fileList->push_back( (*it3).second );
      }
    }
  else // user asked for reverse order
    {
    std::multimap<double, SmartPointer<FileWithName> >::const_iterator it4;
    it4 = distmultimap.end();
    do
      {
      it4--;
      fileList->push_back( (*it4).second );
      } while (it4 != distmultimap.begin() );
    }

  distmultimap.clear();

  return true;
}

void SerieHelper::OrderFileList(FileList *fileSet)
{
  IPPSorter ipps;
  if ( SerieHelper::UserLessThanFunction )
    {
    UserOrdering( fileSet );
    return;
    }
  else if ( ImagePositionPatientOrdering( fileSet ) )
    {
    return ;
    }
  /*
  else if ( ImageNumberOrdering(fileSet ) )
  {
  return ;
  }*/
  else
  {
  FileNameOrdering(fileSet );
  }
}


std::string SerieHelper::CreateUniqueSeriesIdentifier( File * inFile )
{
  StringFilter sf;
  sf.SetFile( *inFile );
  if( true /*inFile->IsReadable()*/ )
    {
    // 0020 000e UI REL Series Instance UID
    //std::string uid = inFile->GetEntryValue (0x0020, 0x000e);
    std::string uid = sf.ToString( Tag(0x0020, 0x000e) );
    std::string id = uid.c_str();
    if(UseSeriesDetails)
      {
      for(SerieRestrictions::iterator it2 = Refine.begin();
        it2 != Refine.end();
        ++it2)
        {
        const Rule &r = *it2;
        //std::string s = inFile->GetEntryValue( r.group, r.elem );
        std::string s = sf.ToString( Tag(r.group, r.elem) );
        //if( s == GDCM_UNFOUND )
        //  {
        //  s = "";
        //  }
        if( id == uid && !s.empty() )
          {
          id += "."; // add separator
          }
        id += s;
        }
      }
    // Eliminate non-alnum characters, including whitespace...
    //   that may have been introduced by concats.
    for(size_t i=0; i<id.size(); i++)
      {
      while(i<id.size()
        && !( id[i] == '.'
          || (id[i] >= 'a' && id[i] <= 'z')
          || (id[i] >= '0' && id[i] <= '9')
          || (id[i] >= 'A' && id[i] <= 'Z')))
        {
        id.erase(i, 1);
        }
      }
    return id;
    }
  else // Could not open inFile
    {
    gdcmWarningMacro("Could not parse series info.");
    std::string id = "GDCM_UNFOUND"; //GDCM_UNFOUND;
    return id;
    }
}

} // end namespace gdcm
