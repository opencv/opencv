/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileStreamer.h"

#include "gdcmTag.h"
#include "gdcmPrivateTag.h"
#include "gdcmDataElement.h"
#include "gdcmReader.h"
#include "gdcmWriter.h"
#include "gdcmAttribute.h"
#include "gdcmEvent.h"
#include "gdcmProgressEvent.h"

#define _FILE_OFFSET_BITS 64

#include <limits>
#include <sys/stat.h> // fstat
#include <stdio.h>

#if defined(_WIN32) && (defined(_MSC_VER) || defined(__MINGW32__))
#include <io.h>
typedef int64_t off64_t;
#else
#if defined(__APPLE__) || defined(__FreeBSD__)
#  define off64_t off_t
#endif
#include <unistd.h> // ftruncate
#endif

namespace gdcm
{

// Implementation detail:
// FILE* have been chosen over std::fstream since it has been reported to lead
// to vastly superior speed.  The only issue in doing so, is the manual
// handling of 64bits offset. Create thin wrapper:
// See here for discussion:
// http://stackoverflow.com/questions/17863594/size-of-off-t-at-compilation-time
// Basically enforce use of off64_t over off_t since on windows off_t is pretty
// much guarantee to be 32bits only.
static inline int FSeeko(FILE *stream, off64_t offset, int whence)
{
#if _WIN32
#if defined(__MINGW32__)
  return fseek(stream, offset, whence); // 32bits
#else
  return _fseeki64(stream, offset, whence);
#endif
#else
  return fseeko(stream, offset, whence);
#endif
}

static inline off64_t FTello(FILE *stream)
{
#if _WIN32
#if defined(__MINGW32__)
  return ftell( stream ); // 32bits
#else
  return _ftelli64( stream );
#endif
#else
  return ftello( stream );
#endif
}

static inline bool FTruncate( const int fd, const off64_t len )
{
#if _WIN32
#if defined(__MINGW32__)
  const long size = len;
  const int ret = _chsize( fd, size ); // 32bits
  return ret == 0 ? true : false;
#else
  const __int64 size  = len;
  const errno_t err = _chsize_s( fd, size );
  return err == 0 ? true : false;
#endif
#else
  const int ret = ftruncate(fd, len);
  return ret == 0 ? true : false;
#endif
}

static bool prepare_file( FILE * pFile, const off64_t offset, const off64_t inslen )
{
  // fast path
  if( inslen == 0 ) return true;

  const size_t BUFFERSIZE = 4096;
  char buffer[BUFFERSIZE];
  struct stat sb;

  int fd = fileno( pFile );
  if (fstat(fd, &sb) == 0)
    {
    if( inslen < 0 )
      {
      off64_t bytes_to_move = sb.st_size - offset;
      off64_t read_start_offset = offset;
      while (bytes_to_move != 0)
        {
        const size_t bytes_this_time = std::min((off64_t)BUFFERSIZE, bytes_to_move);
        const off64_t rd_off = read_start_offset;
        const off64_t wr_off = rd_off + inslen;
        if( FSeeko(pFile, rd_off, SEEK_SET) )
          {
          return false;
          }
        if (fread(buffer, 1, bytes_this_time, pFile) != bytes_this_time)
          {
          return false;
          }
        assert( wr_off < rd_off );
        if( FSeeko(pFile, wr_off, SEEK_SET) )
          {
          return false;
          }
        if (fwrite(buffer, 1, bytes_this_time, pFile) != bytes_this_time)
          {
          return false;
          }
        bytes_to_move -= bytes_this_time;
        read_start_offset += bytes_this_time;
        }
      assert( read_start_offset == sb.st_size );
      if( !FTruncate( fd, sb.st_size + inslen ) )
        {
        return false;
        }
      }
    else
      {
#if 0
      assert(sb.st_size >= offset);
#endif
      if (sb.st_size > offset)
        {
        size_t bytes_to_move = sb.st_size - offset;
        off64_t read_end_offset = sb.st_size;
        while (bytes_to_move != 0)
          {
          const size_t bytes_this_time = std::min(BUFFERSIZE, bytes_to_move);
          const off64_t rd_off = read_end_offset - bytes_this_time;
          assert( (off64_t)rd_off >= offset );
          const off64_t wr_off = rd_off + inslen;
          if( FSeeko(pFile, rd_off, SEEK_SET) )
            {
            return false;
            }
          if (fread(buffer, 1, bytes_this_time, pFile) != bytes_this_time)
            {
            return false;
            }
          if( FSeeko(pFile, wr_off, SEEK_SET) )
            {
            return false;
            }
          if (fwrite(buffer, 1, bytes_this_time, pFile) != bytes_this_time)
            {
            return false;
            }
          bytes_to_move -= bytes_this_time;
          read_end_offset = rd_off;
          }
        assert( read_end_offset == offset );
        }
      }
    // easy case when sb.st_size == offset ...
    } // fstat
  return true;
}

enum Operation
{
  NOOPERATION,
  DATAELEMENT,
  GROUPDATAELEMENT
};

class FileStreamerInternals
{
public:
  FileStreamerInternals():operation(NOOPERATION),
  CurrentTag(0x0,0x0),
  MaxSizeDE(0),
  StartOffset(0),
  CheckTemplateFileName(false),
  InitializeCopy(false),
  CheckPixelDataElement(false),
  pFile(NULL),
  ReservedDataLength(0),
  ReservedGroupDataElement(0),
  Self(NULL)
    {
    PrivateCreator.SetByteValue("",0);
    }
  bool SetTag( const Tag & t ) {
    if( !IsValid() ) return false;
    // else
    CurrentTag = t;
    operation = DATAELEMENT;
    return true;
  }
  bool SetPrivateCreator( const DataElement & de, const size_t maxsizde, const uint8_t startoffset ) {
    if( !IsValid() ) return false;
    // else
    PrivateCreator = de;
    operation = GROUPDATAELEMENT;
    StartOffset = startoffset;
    static const size_t limitmax = std::numeric_limits<uint32_t>::max();
    if( maxsizde % 2 == 0
      && maxsizde < limitmax )
      {
      MaxSizeDE = maxsizde;
      return true;
      }
    return false;
  }
  bool Match( Operation op, Tag const & t )
    {
    if( operation != op ) return false;
    if( CurrentTag != t ) return false;
    return true;
    }
  bool Match( Operation op, DataElement const & de )
    {
    if( operation != op ) return false;
    if( PrivateCreator != de ) return false;
    return true;
    }
  bool Reset( const Tag & )
    {
    CurrentTag = Tag(0x0,0x0);
    operation = NOOPERATION;
    return true;
    }
  bool Reset( const DataElement & )
    {
    PrivateCreator.SetTag( Tag(0x0,0x0) );
    operation = NOOPERATION;
    return true;
    }
  bool IsValid()
    {
    if( TemplateFilename.empty() ) return false;
    if( OutFilename.empty() ) return false;
    if( operation != NOOPERATION ) return false;
    if( CurrentTag != Tag(0x0,0x0) ) return false;
    if( PrivateCreator.GetTag() != Tag(0x0,0x0) ) return false;
    if( PrivateCreator.GetVL() != 0 ) return false;
    return true;
    }
  bool CheckDataElement( const Tag & t )
    {
    static const Tag pixeldata(0x7fe0,0x0010);
    if( t != pixeldata ) return false;
    CheckPixelDataElement = true;
    return true;
    }
  bool StartDataElement( const Tag & t )
    {
    Self->InvokeEvent( StartEvent() );
    const char *outfilename = OutFilename.c_str();
    assert( outfilename );
    actualde = 0;
      {
      std::ifstream is( outfilename, std::ios::binary );
      if( !is.good() ) return false;

      std::set<Tag> tagset;
      tagset.insert( t );

      Reader reader;
      reader.SetStream( is );
      if( !reader.ReadSelectedTags( tagset, false ) )
        {
        return false;
        }

      const File & f = reader.GetFile();
      const DataSet &ds = f.GetDataSet();
      const TransferSyntax &ts = f.GetHeader().GetDataSetTransferSyntax();
      TS = ts;

      // At least on Visual Studio compiler, I need to call clear(), to get
      // proper tellg() value:
      if( is.eof() ) is.clear();

      thepos = is.tellg();
      is.close();

      if( ds.FindDataElement( t ) )
        {
        const DataElement & de = ds.GetDataElement( t );
        // Here is the actual trick:
        if( !de.IsEmpty() )
          {
          // if you trigger this assertion, this means we have been allocating
          // memory for an element when not needed.
          assert( (de.GetByteValue() && de.GetByteValue()->GetPointer() == 0) || de.GetSequenceOfFragments() );
          }
        actualde = de.GetVL() + 2 * de.GetVR().GetLength() + 4;
        thepos -= actualde;
        }
      else
        {
        // no attribute found, easy case !
        }
      }
    pFile = fopen(outfilename, "r+b");
    assert( pFile );
    CurrentDataLenth = 0;
    return true;
    }
  bool AppendToDataElement( const Tag & t, const char *data, size_t len )
    {
    // copy trailing stuff
    if( CurrentDataLenth == 0 )
      {
      size_t dicomlen = 4 + 4; // Tag + VL for Implicit
      if( TS.GetNegociatedType() == TransferSyntax::Explicit )
        dicomlen += 4;
      off64_t newlen = len;
      assert( (size_t)newlen == len );
      newlen += dicomlen;
      newlen -= actualde;
      off64_t plength = newlen;
      assert( ReservedDataLength >= 0 );
      if( ReservedDataLength )
        {
        if( (newlen + ReservedDataLength) >= (off64_t)len )
          {
          plength = newlen + ReservedDataLength - len;
          }
        else
          {
          plength = newlen + ReservedDataLength - len;
          }
        ReservedDataLength -= len;
        assert( ReservedDataLength >= 0 );
        }
      //if( !prepare_file( pFile, (off64_t)thepos + actualde, newlen ) )
      if( !prepare_file( pFile, (off64_t)thepos + actualde, plength ) )
        {
        return false;
        }
      // insert new data in between
      FSeeko(pFile, thepos, SEEK_SET);
      std::stringstream ss;
      const Tag tag = t;
      if( TS.GetSwapCode() == SwapCode::BigEndian )
        tag.Write<SwapperDoOp>(ss);
      else
        tag.Write<SwapperNoOp>(ss);
      if( TS.GetNegociatedType() == TransferSyntax::Explicit )
        {
        VR un = VR::UN;
        un.Write(ss);
        }
      const VL vl = 0; // will be updated later (UpdateDataElement)
      if( TS.GetSwapCode() == SwapCode::BigEndian )
        vl.Write<SwapperDoOp>(ss);
      else
        vl.Write<SwapperNoOp>(ss);
      const std::string dicomdata = ss.str();
      fwrite(dicomdata.c_str(), 1, dicomdata.size(), pFile);
      assert( dicomdata.size() == dicomlen );
      thepos += dicomlen;
      }
    else
      {
      const off64_t curpos = FTello(pFile);
      assert( curpos == thepos );
      if( ReservedDataLength >= (off64_t)len )
        {
        // simply update remaining reserved buffer:
        ReservedDataLength -= len;
        }
      else
        {
        const off64_t plength = len - ReservedDataLength;
        assert( plength >= 0 );
        if( !prepare_file( pFile, (off64_t)curpos, plength) )
          {
          return false;
          }
        ReservedDataLength = 0; // no more reserved buffer
        }
      FSeeko(pFile, curpos, SEEK_SET);
      }

    assert( ReservedDataLength >= 0 );
    fwrite(data, 1, len, pFile);
    thepos += len;
    CurrentDataLenth += len;
    assert( CurrentDataLenth < std::numeric_limits<uint32_t>::max() );
    return true;
    }
  bool StopDataElement( const Tag & t )
    {
    // Update DataElement:
    const size_t currentdatalenth = CurrentDataLenth;
    assert( ReservedDataLength >= 0);
    //const off64_t refpos = FTello(pFile);
    if( !UpdateDataElement( t ) )
      {
      return false;
      }
    if( ReservedDataLength > 0)
      {
      const off64_t curpos = thepos;
      if( !prepare_file( pFile, curpos + ReservedDataLength, - ReservedDataLength) )
        {
        return false;
        }
      ReservedDataLength = 0;
      }
    assert( ReservedDataLength == 0);
    fclose(pFile);
    pFile = NULL;
    // Do some extra work:
    if( CheckPixelDataElement )
      {
      const char *outfilename = OutFilename.c_str();
      Reader reader;
      reader.SetFileName( outfilename );
      std::set<Tag> tagset;
      // (0028,0002) US 3                               # 2,1 Samples per Pixel
      // (0028,0010) US 256                             # 2,1 Rows
      // (0028,0011) US 256                             # 2,1 Columns
      // (0028,0100) US 16                              # 2,1 Bits Allocated
      // (0028,0008) IS [56]                            # 2,1 Number of Frames
      Attribute<0x28,0x02> spp = { 1 };
      Attribute<0x28,0x10> rows;
      Attribute<0x28,0x11> cols;
      Attribute<0x28,0x100> ba = { 0 };
      Attribute<0x28,0x08> nframes = { 1 };
      tagset.insert( spp.GetTag() );
      tagset.insert( rows.GetTag() );
      tagset.insert( cols.GetTag() );
      tagset.insert( ba.GetTag() );
      tagset.insert( nframes.GetTag() );
      if( !reader.ReadSelectedTags( tagset, true) )
        {
        return false;
        }
      const File & f = reader.GetFile();
      const DataSet &ds = f.GetDataSet();
      const FileMetaInformation &fmi = f.GetHeader();
      const TransferSyntax &ts = fmi.GetDataSetTransferSyntax();
      if( ts.IsEncapsulated() )
        {
        gdcmDebugMacro( "Only RAW (uncompressed) Pixel Data is supported for now" );
        return false;
        }
      spp.SetFromDataSet( ds );
      rows.SetFromDataSet( ds );
      cols.SetFromDataSet( ds );
      ba.SetFromDataSet( ds );
      nframes.SetFromDataSet( ds );
      assert( ba.GetValue() % 8 == 0 );
      const size_t computedlength = spp.GetValue() * nframes.GetValue() * rows.GetValue() * cols.GetValue() * ( ba.GetValue() / 8 );
      if( computedlength != currentdatalenth )
        {
        gdcmDebugMacro( "Invalid size for Pixel Data Element: " << computedlength << " vs " << currentdatalenth );
        return false;
        }
      }
    Self->InvokeEvent( EndEvent() );
    return true;
    }
  bool ReserveDataElement( size_t len )
    {
    ReservedDataLength = len;
    return true;
    }
  bool ReserveGroupDataElement( unsigned short ndataelement )
    {
    if( ndataelement <= 256 )
      {
      this->ReservedGroupDataElement = ndataelement;
      return true;
      }
    return false;
    }
  bool StartGroupDataElement( const PrivateTag & ori_pt )
    {
    Self->InvokeEvent( StartEvent() );
    // Need to cleanup the whole group, well not really, since as per DICOM
    // mechanism we can simply append
    const char *outfilename = OutFilename.c_str();
    DataElement private_creator = ori_pt.GetAsDataElement();
    assert( outfilename );
    Tag curtag = ori_pt;
      {
      bool cont = false;
      do
        {
        Self->InvokeEvent( IterationEvent() );
        std::set<PrivateTag> tagset;
        PrivateTag pt = curtag;
        tagset.insert( pt );

        std::ifstream is( outfilename, std::ios::binary );
        if( !is.good() ) return false;

        Reader reader;
        reader.SetStream( is );
        if( !reader.ReadSelectedPrivateTags( tagset, false ) )
          {
          return false;
          }

        const File & f = reader.GetFile();
        const DataSet &ds = f.GetDataSet();
        cont = ds.FindDataElement( curtag );
        if( cont )
          {
          curtag.SetElement( (uint16_t)(curtag.GetElement() + 0x1) );
          }
        else
          {
          const TransferSyntax &ts = f.GetHeader().GetDataSetTransferSyntax();
          TS = ts;
          thepos = is.tellg();
          }
        is.close();
        }
      while( cont );
      // found a free spot:
      private_creator.GetTag().SetElement( curtag.GetElement() );
      actualde = 0;
      }

    // copy trailing stuff
    std::string dicomdata;
      {
      std::stringstream ss;
      assert( private_creator.GetTag().IsPrivateCreator() );
      if( TS.GetSwapCode() == SwapCode::BigEndian )
        {
        if( TS.GetNegociatedType() == TransferSyntax::Explicit )
          {
          private_creator.Write<ExplicitDataElement,SwapperDoOp>( ss );
          }
        else
          {
          return 1;
          }
        }
      else
        {
        if( TS.GetNegociatedType() == TransferSyntax::Explicit )
          {
          private_creator.Write<ExplicitDataElement,SwapperNoOp>( ss );
          }
        else
          {
          private_creator.Write<ImplicitDataElement,SwapperNoOp>( ss );
          }
        }
      dicomdata = ss.str();
      }

    // find thepcpos:
    std::streampos thepcpos = 0;
      {
      std::set<Tag> tagset;
      Tag prev = private_creator.GetTag();
      //assert( prev.GetElement() );
      prev.SetElement( (uint16_t)(prev.GetElement() - 0x1) );
      tagset.insert( prev );

      std::ifstream is( outfilename, std::ios::binary );
      if( !is.good() ) return false;

      Reader reader;
      reader.SetStream( is );
      if( !reader.ReadSelectedTags( tagset, false ) )
        {
        return false;
        }
      thepcpos = is.tellg();
      is.close();
      }

    const size_t pclen = dicomdata.size();
    pFile = fopen(outfilename, "r+b");

    if( !prepare_file( pFile, (off64_t)thepcpos, pclen ) )
      {
      return false;
      }

    FSeeko(pFile, thepcpos, SEEK_SET);
      {
      fwrite(dicomdata.c_str(), 1, dicomdata.size(), pFile);
#if 0
      fflush(pFile); // need to flush so that prepareFile works as expected.
#endif
      thepos += pclen;
      }

    CurrentGroupTag.SetElement( this->StartOffset ); // First possible
    CurrentGroupTag.SetPrivateCreator( private_creator.GetTag() );
    CurrentDataLenth = 0;

    if( this->ReservedGroupDataElement )
      {
      if( !this->ReserveDataElement( MaxSizeDE ) )
        {
        return false;
        }
      }
    return true;
    }
  bool AppendToGroupDataElement( const DataElement & , const char *data, size_t len )
    {
    size_t len_to_move = len;
    while( len_to_move != 0 )
      {
      Self->InvokeEvent( IterationEvent() );
      const size_t len_this_time = std::min(MaxSizeDE - CurrentDataLenth, len_to_move);
      assert( len_this_time % 2 == 0 );
      if( !AppendToDataElement( CurrentGroupTag, data, len_this_time ) )
        {
        return false;
        }
      assert( CurrentDataLenth <= MaxSizeDE );
      len_to_move -= len_this_time;
      if( CurrentDataLenth == MaxSizeDE )
        {
        // flush
        assert( CurrentDataLenth % 2 == 0 );
        if( !UpdateDataElement( CurrentGroupTag ) )
          {
          return false;
          }
        assert( CurrentDataLenth == 0 );
        CurrentGroupTag.SetElement( (uint16_t)(CurrentGroupTag.GetElement() + 1) );
        const int lowbits = CurrentGroupTag.GetElement() & 0x00ff;
        if( lowbits == 0 )
          {
          // we are wrapping, this is not handled:
          gdcmDebugMacro( "Too many data elements. Giving up" );
          return false;
          }
        }
      }
    return true;
    }
  bool StopGroupDataElement( const DataElement & )
    {
    return StopDataElement( CurrentGroupTag );
    }
  TransferSyntax TS;
  std::string TemplateFilename;
  std::string OutFilename;
private:
  Operation operation;
  Tag CurrentTag;
  DataElement PrivateCreator;
  size_t MaxSizeDE;
  uint8_t StartOffset;
public:
  bool CheckTemplateFileName;
  bool InitializeCopy;
  bool CheckPixelDataElement;
private:
  // really private !
  FILE* pFile;
  std::streampos thepos;
  size_t actualde;
  size_t CurrentDataLenth;
  Tag CurrentGroupTag;
  off64_t ReservedDataLength;
  unsigned short ReservedGroupDataElement;
public:
  FileStreamer *Self;
private:
  bool UpdateDataElement( const Tag & t )
    {
    // This function will set the VL for current DataElement:
    if( CurrentDataLenth )
      {
      if( CurrentDataLenth % 2 == 1 )
        {
        const off64_t curpos = FTello(pFile);
        if( ReservedDataLength >= 1 )
          {
          // simply update remaining reserved buffer:
          ReservedDataLength -= 1;
          }
        else
          {
          if( !prepare_file( pFile, (off64_t)curpos, 1) )
            {
            return false;
            }
          }
        FSeeko(pFile, curpos, SEEK_SET);
        int ret = fputc(0, pFile); // Set to NULL padding ?
        thepos += 1;
        assert( ret != EOF ); (void)ret;
        CurrentDataLenth += 1;
        }
      assert( CurrentDataLenth % 2 == 0 );
      off64_t vlpos = thepos;
      vlpos -= CurrentDataLenth;
      vlpos -= 4; // VL
      if( TS.GetNegociatedType() == TransferSyntax::Explicit )
        {
        vlpos -= 4; // VR
        }
      vlpos -= 4; // Tag
      gdcmAssertAlwaysMacro( vlpos >= 0 );
      FSeeko(pFile, vlpos, SEEK_SET);
      std::stringstream ss;
      const Tag tag = t;
      if( TS.GetSwapCode() == SwapCode::BigEndian )
        tag.Write<SwapperDoOp>(ss);
      else
        tag.Write<SwapperNoOp>(ss);
      if( TS.GetNegociatedType() == TransferSyntax::Explicit )
        {
        VR un = VR::UN;
        un.Write(ss);
        }
      gdcmAssertAlwaysMacro( CurrentDataLenth < std::numeric_limits<uint32_t>::max() );
      const VL vl = (uint32_t)CurrentDataLenth;
      if( TS.GetSwapCode() == SwapCode::BigEndian )
        vl.Write<SwapperDoOp>(ss);
      else
        vl.Write<SwapperNoOp>(ss);
      const std::string dicomdata = ss.str();
      fwrite(dicomdata.c_str(), 1, dicomdata.size(), pFile);
      CurrentDataLenth = 0;
      }
    return true;
    }
};

FileStreamer::FileStreamer()
{
  Internals = new FileStreamerInternals;
}

FileStreamer::~FileStreamer()
{
  delete Internals;
}

void FileStreamer::SetTemplateFileName(const char *filename_native)
{
  if( filename_native )
    Internals->TemplateFilename = filename_native;
}

bool FileStreamer::InitializeCopy()
{
#if 0
  static int checksize = 0;
  if( !checksize )
    {
    const int soff = sizeof( off64_t );
    const int si64 = sizeof( int64_t );
    if( soff != si64 ) return false;
    if( !(sizeof(sb.st_size) > 4) ) // LFS ?
      {
      return false;
      }
    ++checksize;
    }
#endif

  if( !this->Internals->InitializeCopy )
    {
    const char *filename = this->Internals->TemplateFilename.c_str();
    const char *outfilename = this->Internals->OutFilename.c_str();
    if( this->Internals->CheckTemplateFileName )
      {
      // Prefer a GDCM copy, even if this is slower in most cases, this
      // guarantee that the output file will be correct as per-DICOM spec,
      // which will greatly simplify the rest of the process.
      Reader reader;
      reader.SetFileName( filename );
      if( !reader.Read() ) return false;
      Writer writer;
      writer.SetFileName( outfilename );
      writer.SetFile( reader.GetFile() );
      if( !writer.Write() ) return false;
      }
    else
      {
      assert( filename );
      assert( outfilename );
      std::ifstream is( filename, std::ios::binary );
      if( !is.good() ) return false;
      std::ofstream of( outfilename, std::ios::binary );
      if( !of.good() ) return false;
      of << is.rdbuf();
      of.close();
      is.close();
      }
    this->Internals->InitializeCopy = true;
    this->Internals->Self = this;
    }
  return true;
}

bool FileStreamer::StartDataElement( const Tag & t )
{
  if( !this->Internals->SetTag( t ) )
    {
    gdcmDebugMacro( "Could not StartDataElement" );
    return false;
    }
  if( !InitializeCopy() )
    {
    gdcmDebugMacro( "Could not InitializeCopy" );
    return false;
    }
  return Internals->StartDataElement( t );
}

bool FileStreamer::AppendToDataElement( const Tag & t, const char *data, size_t len )
{
  if( !this->Internals->Match( DATAELEMENT, t) )
    {
    gdcmDebugMacro( "Could not AppendToDataElement" );
    return false;
    }
  return this->Internals->AppendToDataElement( t, data, len );
}

bool FileStreamer::StopDataElement( const Tag & t )
{
  if( !this->Internals->Reset(t) )
    {
    return false;
    }
  return this->Internals->StopDataElement( t );
}

bool FileStreamer::StartGroupDataElement( const PrivateTag & pt, size_t maxsizede, uint8_t startoffset )
{
  const DataElement private_creator = pt.GetAsDataElement();
  if( !this->Internals->SetPrivateCreator( private_creator, maxsizede, startoffset ) )
    {
    gdcmDebugMacro( "Could not StartGroupDataElement" );
    return false;
    }
  if( !InitializeCopy() )
    {
    gdcmDebugMacro( "Could not InitializeCopy" );
    return false;
    }
  return Internals->StartGroupDataElement( pt );
}

bool FileStreamer::AppendToGroupDataElement( const PrivateTag & pt, const char *data, size_t len )
{
  const DataElement private_creator = pt.GetAsDataElement();
  if( !this->Internals->Match( GROUPDATAELEMENT, private_creator) )
    {
    gdcmDebugMacro( "Could not AppendToGroupDataElement" );
    return false;
    }
  return this->Internals->AppendToGroupDataElement( private_creator, data, len );
}

bool FileStreamer::StopGroupDataElement( const PrivateTag & pt )
{
  const DataElement private_creator = pt.GetAsDataElement();
  if( !this->Internals->Reset( private_creator ) )
    {
    return false;
    }
  return this->Internals->StopGroupDataElement( private_creator );
}

bool FileStreamer::ReserveGroupDataElement( unsigned short ndataelement )
{
  return Internals->ReserveGroupDataElement( ndataelement );
}
bool FileStreamer::ReserveDataElement( size_t len )
{
  return Internals->ReserveDataElement( len );
}

void FileStreamer::SetOutputFileName(const char *filename_native)
{
  if( filename_native )
    Internals->OutFilename = filename_native;
}

void FileStreamer::CheckTemplateFileName(bool check)
{
  this->Internals->CheckTemplateFileName = check;
}

bool FileStreamer::CheckDataElement( const Tag & t )
{
  return this->Internals->CheckDataElement( t );
}

} // end namespace gdcm
