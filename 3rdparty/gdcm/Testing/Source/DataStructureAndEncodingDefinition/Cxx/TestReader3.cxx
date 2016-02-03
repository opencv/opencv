/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 *  OS Specific: need a POSIX system with mmap functionality
 */
#include <sstream>
#include <fstream>
#include <iostream>

// fstat
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h> /* close */

// open
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// mmap
#include <sys/mman.h>

#include "gdcmFile.h"
#include "gdcmDataSet.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmSmartPointer.h"
#include "gdcmReader.h"
#include "gdcmWriter.h"
#include "gdcmSystem.h"
#include "gdcmTesting.h"
#include "gdcmAttribute.h"

#if 0
class membuf: public std::streambuf
{
   public:
      membuf(const char* filename)
         :std::streambuf()
      {
         hFile = CreateFileA(
               filename,
               GENERIC_READ,
               0,
               NULL,
               OPEN_EXISTING,
               FILE_ATTRIBUTE_READONLY,
               NULL);
         hFileMappingObject = CreateFileMapping(
               hFile,
               NULL,
               PAGE_READONLY,
               0,
               0,
               NULL);
         beg = MapViewOfFile(hFileMappingObject,
               FILE_MAP_READ,
               0,
               0,
               0);
         char* e = (char*)beg;
         e += GetFileSize( hFile, NULL );
         setg((char*)beg,(char*)beg,e);
         setp(&outBuf,&outBuf);
      }
      /// Close the memory mapping and the opened
      /// file.
      ~membuf()
      {
         UnmapViewOfFile(beg);
         CloseHandle( hFileMappingObject );
         CloseHandle( hFile );
      }
   private:
      /// We can't copy this.
      membuf(const membuf&);
      membuf& operator=(const membuf&);
      /// Outbuf buffer that is not used since we
      /// don't write to the memory map.
      char outBuf;
      HANDLE hFile,hFileMappingObject;
      /// Pointer to the beginning of the mapped
      /// memory area.
      LPVOID beg;
};
#endif

/*
 * http://beej.us/guide/bgipc/output/html/multipage/mmap.html
 * http://www.dba-oracle.com/oracle_tips_mount_options.htm
 */

class membuf : public std::streambuf
{
public:
  membuf(char* mem, size_t length)
  {
    setg(mem, mem, mem + length);
    setp(mem, mem + length);
  }
  std::streampos seekpos(std::streampos pos, std::ios_base::openmode)
    {
    char *p = eback() + pos;
    if(p>=eback() && p <=egptr())
      {
      setg(eback(),p,egptr());
      return pos;
      }
    else
      return -1;
    }

  std::streampos seekoff(std::streamoff off,
    std::ios_base::seekdir dir, std::ios_base::openmode)
    {
    char *p;
    switch(dir)
      {
    case std::ios_base::beg:
      p = eback() + off;
      break;
    case std::ios_base::cur:
      p = gptr() + off;
      break;
    case std::ios_base::end:
      p = egptr() + off;
      break;
    default:
      p = 0;
      break;
      }
    if(p>=eback() && p <= egptr())
      {
      setg(eback(),p,egptr());
      return std::streampos(p-egptr());
      }
    else
      return -1;
    }
};

std::istream & DoTheMMapRead(std::istream &is)
{
  gdcm::Reader reader;
  reader.SetStream(is);
  reader.Read();

  //gdcm::Dumper printer;
  //printer.SetFile ( reader.GetFile() );
  //printer.Print( std::cout );
  return is;
}

int TestRead3(const char *subdir, const char * filename)
{
/// FIXME Because GDCM is seeging back and forth in the DICOM file
// we cannot just apply mmap on any file, so let's clean them first:
//
  gdcm::Reader r;
  r.SetFileName( filename );
  if( !r.Read() )
    {
    return 1;
    }
  //
  // Create directory first:
  const char * tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir ) )
    {
    gdcm::System::MakeDirectory( tmpdir );
    //return 1;
    }
  const char * outfilename = gdcm::Testing::GetTempFilename( filename, subdir );

  // HACK:
  gdcm::DataSet &ds = r.GetFile().GetDataSet();
  gdcm::Attribute<0x0008,0x0018> at;
  if( !ds.FindDataElement( at.GetTag() ) || ds.GetDataElement( at.GetTag() ).IsEmpty() )
    {
    const gdcm::UIComp dummyuid = "1.2.3.4.5.6.7.8.9.0";
    at.SetValue( dummyuid );
    ds.Replace( at.GetAsDataElement() );
    }
  gdcm::Writer w;
  w.SetFile( r.GetFile() );
  w.SetFileName( outfilename );
  if( !w.Write() )
    {
    return 1;
    }
  const char *path = outfilename;
  bool readonly = true;
  int flags = (readonly ? O_RDONLY : O_RDWR);

  int handle = ::open(path, flags, S_IRWXU);

  struct stat info;
  const bool success = ::fstat(handle, &info) != -1;
  if( !success ) return 1;
  off_t size = info.st_size;

  off_t offset = 0;
  char* hint = 0;
  void* data = ::mmap( hint, size,
    readonly ? PROT_READ : (PROT_READ | PROT_WRITE),
    readonly ? MAP_PRIVATE : MAP_SHARED,
    handle, offset );
  if (data == MAP_FAILED) {
    return 1;
  }
  char *chardata = reinterpret_cast<char*>(data);

  membuf mb( chardata, size );
  std::istream is(&mb) ;

  DoTheMMapRead(is);

  // cleanup
  assert( handle );
  bool error = false;
  error = ::munmap(data, size) != 0 || error;
  error = ::close(handle) != 0 || error;
  handle = 0;

  if ( error )  return 1;

  return 0;
}

int TestReader3(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestRead3(argv[0], filename);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestRead3( argv[0], filename );
    ++i;
    }

  return r;
}
