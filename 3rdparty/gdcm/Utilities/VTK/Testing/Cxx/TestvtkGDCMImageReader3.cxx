/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMImageReader.h"
#include "vtkImageData.h"
#include "vtkStringArray.h"

#include "gdcmIPPSorter.h"
#include "gdcmFilename.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmDirectory.h"
#include "gdcmScanner.h"

#include <iterator>

/*
 * There is special case we need to handle here:
 * What if the Series we are trying to read contained a changing shift/scale ?
 */

int TestvtkGDCMImageRead3(const char *dir, const char *studyuid)
{
  std::cout << "Working on : " << dir << std::endl;
  int ret = 0;
  gdcm::Directory d;
  d.Load( dir );
  const gdcm::Directory::FilenamesType &l1 = d.GetFilenames();
  const size_t nfiles = l1.size(); (void)nfiles;

  // Sub-select only the DICOM files in this directory:
  gdcm::Scanner s;
  const gdcm::Tag t1(0x0020,0x000d); // Study Instance UID
  const gdcm::Tag t2(0x0020,0x000e); // Series Instance UID
  const gdcm::Tag t3(0x0028,0x1052); // Rescale Intercept
  const gdcm::Tag t4(0x0028,0x1053); // Rescale Slope
  s.AddTag( t1 );
  s.AddTag( t2 );
  s.AddTag( t3 );
  s.AddTag( t4 );
  bool b = s.Scan( d.GetFilenames() );
  if( !b )
    {
    std::cerr << "Scanner failed" << std::endl;
    return 1;
    }
  //s.Print( std::cout );
  //std::cout << dir1 << std::endl;
  //gdcm::Scanner::ValuesType const &v = s.GetValues();
  //std::cout << v.size() << std::endl;
  //std::copy(v.begin(), v.end(),
  //  std::ostream_iterator<std::string>(std::cout, "\n"));


  // Let take all the following files and pretend there are part of the
  // same Series:
  // (0020,000d) UI [1.3.46.670589.11.30.6.106253130282775287] #  40, 1 StudyInstanceUID
  // ... footnote : they are part of the same Study, but separate Series
  gdcm::Directory::FilenamesType keys = s.GetKeys();
  gdcm::Directory::FilenamesType::const_iterator it = keys.begin();

  std::vector<char> wholebuffer;
  vtkStringArray *sarray = vtkStringArray::New();
  for(; it != keys.end() /*&& i < 2*/; ++it)
    {
    const char *filename = it->c_str();
    assert( s.IsKey( filename ) );
    const gdcm::Tag &reftag = t1;
    const char *value =  s.GetValue( filename, reftag );
    if( value && strcmp( value, studyuid ) == 0 )
      {
      //std::cout << "file: " << filename << std::endl;
      sarray->InsertNextValue( filename );

      // Read each file
      vtkGDCMImageReader *singlereader = vtkGDCMImageReader::New();
      singlereader->SetFileName( filename );
      singlereader->Update();
      vtkImageData* img = singlereader->GetOutput();
      int ssize = img->GetScalarSize();
      vtkIdType npts = img->GetNumberOfPoints();
      char * ptr = (char*)img->GetScalarPointer();
      //std::vector<char> buffer(ptr, ptr+npts*ssize);
      wholebuffer.insert(wholebuffer.end(), ptr, ptr+npts*ssize);
      singlereader->Delete();
      }
    }
  std::cout << "Found " << sarray->GetSize() << " files belonging to StudyUID: " << studyuid << std::endl;
  vtkGDCMImageReader *reader = vtkGDCMImageReader::New();
  reader->SetFileNames( sarray );
  sarray->Delete();

  reader->Update();

  vtkImageData* img = reader->GetOutput();
  size_t ssize = img->GetScalarSize();
  vtkIdType npts = img->GetNumberOfPoints();
  char * ptr = (char*)img->GetScalarPointer();
  if( wholebuffer.size() != npts * ssize || wholebuffer.empty() )
    {
    std::cerr << "Something went terribly wrong" << std::endl;
    ret = 1;
    }

  if( memcmp(&wholebuffer[0], ptr, wholebuffer.size() ) != 0 )
    {
    std::cerr << "BUG: (n) Readers are not equivalent to a single reader !" << std::endl;
    ret = 1;
    }
#if 0
std::ofstream o1("/tmp/debug1.raw", std::ios::binary);
o1.write(&wholebuffer[0], wholebuffer.size());
o1.close();

std::ofstream o2("/tmp/debug2.raw", std::ios::binary);
o2.write(ptr, wholebuffer.size());
o2.close();
#endif

  reader->Delete();
  return ret;
}


int TestvtkGDCMImageReader3(int , char *[])
{
  int ret = 0;
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  const char *root = gdcm::Testing::GetDataExtraRoot();
  std::string dir1 = root;
  std::string dir2 = root;
  std::string dir3 = root;
  // dir1 & dir2 have changing 'Rescale Slope':
  dir1 += "/gdcmSampleData/Philips_Medical_Images/mr711-mr712/";
  dir2 += "/gdcmSampleData/ForSeriesTesting/Perfusion/images/";
  // dir3 has RescaleSlope == 0 !
  dir3 += "/gdcmSampleData/ForSeriesTesting/Dentist/images/";

  const char *studyuids[] = {
    "1.3.46.670589.11.30.6.106253130282775287",
    "1.2.250.1.38.2.1.12.7118916513228.20041110114508.431746279",
    "1.76.380.18.1.10713.1.1335"
  };

  ret += TestvtkGDCMImageRead3(dir1.c_str(), studyuids[0]);
  ret += TestvtkGDCMImageRead3(dir2.c_str(), studyuids[1]);
  ret += TestvtkGDCMImageRead3(dir3.c_str(), studyuids[2]);


  return ret;
}
