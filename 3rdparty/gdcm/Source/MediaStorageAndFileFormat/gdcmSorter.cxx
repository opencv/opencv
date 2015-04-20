/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSorter.h"
#include "gdcmElement.h"
#include "gdcmSerieHelper.h"
#include "gdcmFile.h"
#include "gdcmReader.h"

#include <map>
#include <algorithm>

namespace gdcm
{

Sorter::Sorter()
{
  SortFunc = NULL;
}


Sorter::~Sorter()
{
}

void Sorter::SetSortFunction( SortFunction f )
{
  SortFunc = f;
}


namespace {
class SortFunctor
{
public:
  bool operator() (File const *file1, File const *file2)
    {
    return (SortFunc)(file1->GetDataSet(), file2->GetDataSet());
    }
  Sorter::SortFunction SortFunc;
  SortFunctor()
    {
    SortFunc = 0;
    }
  SortFunctor(SortFunctor const &sf)
    {
    SortFunc = sf.SortFunc;
    }
  void operator=(Sorter::SortFunction sf)
    {
    SortFunc = sf;
    }
};
}

bool Sorter::StableSort(std::vector<std::string> const & filenames)
{
  // BUG: I cannot clear Filenames since input filenames could also be the output of ourself...
  // Filenames.clear();
  if( filenames.empty() || !SortFunc )
    {
    Filenames.clear();
    return true;
    }

  std::vector< SmartPointer<FileWithName> > filelist;
  filelist.resize( filenames.size() );

  std::vector< SmartPointer<FileWithName> >::iterator it2 = filelist.begin();
  for( Directory::FilenamesType::const_iterator it = filenames.begin();
    it != filenames.end() && it2 != filelist.end(); ++it, ++it2)
    {
    Reader reader;
    reader.SetFileName( it->c_str() );
    SmartPointer<FileWithName> &f = *it2;
    if( reader.Read() )
      {
      f = new FileWithName( reader.GetFile() );
      f->filename = *it;
      }
    else
      {
      gdcmErrorMacro( "File could not be read: " << it->c_str() );
      return false;
      }
    }
  SortFunctor sf;
  sf = Sorter::SortFunc;
  std::stable_sort( filelist.begin(), filelist.end(), sf);

  Filenames.clear(); // cleanup any previous call
  for(it2 = filelist.begin(); it2 != filelist.end(); ++it2 )
    {
    SmartPointer<FileWithName> const & f = *it2;
    Filenames.push_back( f->filename );
    }

  return true;
}

bool Sorter::Sort(std::vector<std::string> const & filenames)
{
  (void)filenames;
  Filenames.clear();

  if( filenames.empty() || !SortFunc ) return true;

  std::vector< SmartPointer<FileWithName> > filelist;
  filelist.resize( filenames.size() );

  std::vector< SmartPointer<FileWithName> >::iterator it2 = filelist.begin();
  for( Directory::FilenamesType::const_iterator it = filenames.begin();
    it != filenames.end() && it2 != filelist.end(); ++it, ++it2)
    {
    Reader reader;
    reader.SetFileName( it->c_str() );
    SmartPointer<FileWithName> &f = *it2;
    if( reader.Read() )
      {
      f = new FileWithName( reader.GetFile() );
      f->filename = *it;
      }
    else
      {
      gdcmErrorMacro( "File could not be read: " << it->c_str() );
      return false;
      }
    }
  //std::sort( filelist.begin(), filelist.end(), Sorter::SortFunc);
  SortFunctor sf;
  sf = Sorter::SortFunc;
  std::sort( filelist.begin(), filelist.end(), sf);

  for(it2 = filelist.begin(); it2 != filelist.end(); ++it2 )
    {
    SmartPointer<FileWithName> const & f = *it2;
    Filenames.push_back( f->filename );
    }

  return true;
}

bool Sorter::AddSelect( Tag const &tag, const char *value )
{
  Selection.insert( SelectionMap::value_type(tag,value) );
  return true;
}


void Sorter::Print( std::ostream &os) const
{
  std::vector<std::string>::const_iterator it = Filenames.begin();
  for( ; it != Filenames.end(); ++it)
    {
    os << *it <<std::endl;
    }
}

} // end namespace gdcm
