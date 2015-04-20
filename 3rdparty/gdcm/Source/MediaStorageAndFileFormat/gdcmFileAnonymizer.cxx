/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileAnonymizer.h"

#include "gdcmReader.h"

#include <fstream>
#include <set>
#include <vector>
#include <map>
#include <algorithm> // sort

namespace gdcm
{

enum Action
{
  EMPTY,
  REMOVE,
  REPLACE
};

struct PositionEmpty
{
  std::streampos BeginPos;
  std::streampos EndPos;
  Action action;
  bool IsTagFound; // Required for EMPTY
  DataElement DE;
  bool operator() (const PositionEmpty & i, const PositionEmpty & j)
    {
    if( i.BeginPos == j.BeginPos )
      {
      return i.DE.GetTag() < j.DE.GetTag();
      }
    // else
    return (int)i.BeginPos < (int)j.BeginPos;
    }
};

class FileAnonymizerInternals
{
public:
  std::string InputFilename;
  std::string OutputFilename;
  std::set<Tag> EmptyTags;
  std::set<Tag> RemoveTags;
  std::map<Tag, std::string> ReplaceTags;
  TransferSyntax TS;
  std::vector<PositionEmpty> PositionEmptyArray;
};

FileAnonymizer::FileAnonymizer()
{
  Internals = new FileAnonymizerInternals;
}

FileAnonymizer::~FileAnonymizer()
{
  delete Internals;
}

void FileAnonymizer::Empty( Tag const &t )
{
  if( t.GetGroup() >= 0x0008 )
    {
    Internals->EmptyTags.insert( t );
    }
}

void FileAnonymizer::Remove( Tag const &t )
{
  if( t.GetGroup() >= 0x0008 )
    {
    Internals->RemoveTags.insert( t );
    }
}

void FileAnonymizer::Replace( Tag const &t, const char *value )
{
  if( value && t.GetGroup() >= 0x0008 )
    {
    Internals->ReplaceTags.insert(
      std::make_pair( t, value) );
    }
}

void FileAnonymizer::Replace( Tag const &t, const char *value, VL const & vl )
{
  if( value && t.GetGroup() >= 0x0008 )
    {
    Internals->ReplaceTags.insert(
      std::make_pair( t, std::string(value, vl) ) );
    }
}

void FileAnonymizer::SetInputFileName(const char *filename_native)
{
  if( filename_native )
    Internals->InputFilename = filename_native;
}

void FileAnonymizer::SetOutputFileName(const char *filename_native)
{
  if( filename_native )
    Internals->OutputFilename = filename_native;
}

// portable way to check for existence (actually: accessibility):
static inline bool file_exist(const char *filename)
{
  std::ifstream infile(filename, std::ios::binary);
  return infile.good();
}


bool FileAnonymizer::ComputeReplaceTagPosition()
{
/*
 Implementation details:
 We should make sure that the user know what she is doing in case of SQ.
 Let's assume a User call Replace( Tag(0x0008,0x2112), "FOOBAR" )
 For quite a lot of DICOM implementation this Tag is required to be a SQ.
 Therefore even if the Attribute is declared with VR:UN, some implementation
 will try very hard to decode it as SQ...which obviously will fail
 Instead do not support SQ at all here and document it should not be used for SQ
 */
  assert( !Internals->InputFilename.empty() );
  const char *filename = Internals->InputFilename.c_str();
  assert( filename );
  const bool inplace = file_exist(Internals->OutputFilename.c_str());

  std::map<Tag, std::string>::reverse_iterator rit = Internals->ReplaceTags.rbegin();
  for ( ; rit != Internals->ReplaceTags.rend(); rit++ )
    {
    PositionEmpty pe;

    std::set<Tag> removeme;
    const Tag & t = rit->first;
    const std::string & valuereplace = rit->second;
    removeme.insert( t );

    std::ifstream is( filename, std::ios::binary );
    Reader reader;
    reader.SetStream( is );
    if( !reader.ReadSelectedTags( removeme ) )
      {
      return false;
      }

    pe.EndPos = pe.BeginPos = is.tellg();
    pe.action = REPLACE;
    pe.IsTagFound = false;
    const File & f = reader.GetFile();
    const DataSet &ds = f.GetDataSet();
    const TransferSyntax &ts = f.GetHeader().GetDataSetTransferSyntax();
    Internals->TS = ts;

    pe.DE.SetTag( t );
    if( ds.FindDataElement( t ) )
      {
      const DataElement &de = ds.GetDataElement( t );
      pe.IsTagFound = true;
      pe.DE.SetVL( de.GetVL() ); // Length is not used, unless to check undefined flag
      pe.DE.SetVR( de.GetVR() );
      assert( pe.DE.GetVL().IsUndefined() == de.GetVL().IsUndefined() );
      assert( pe.DE.GetVR() == de.GetVR() );
      assert( pe.DE.GetTag() == de.GetTag() );
      if( de.GetVL().IsUndefined() )
        {
        // This is a SQ
        gdcmErrorMacro( "Replacing a SQ is not supported. Use Remove() or Empty()" );
        return false;
        }
      else
        {
        if( inplace && valuereplace.size() != de.GetVL() )
          {
          gdcmErrorMacro( "inplace mode requires same length attribute" ); // TODO we could allow smaller size (and pad with space...)
          return false;
          }
        assert( !de.GetVL().IsUndefined() );
        pe.BeginPos -= de.GetVL();
        pe.BeginPos -= 2 * de.GetVR().GetLength(); // (VR+) VL
        pe.BeginPos -= 4; // Tag
        assert( (int)pe.EndPos ==
          (int)pe.BeginPos + (int)de.GetVL() + 2 * de.GetVR().GetLength() + 4 );
        }
      pe.DE.SetByteValue( valuereplace.c_str(), (uint32_t)valuereplace.size() );
      assert( pe.DE.GetVL() == valuereplace.size() );
      }
    else
      {
      if( inplace )
        {
        gdcmErrorMacro( "inplace mode requires existing tag (cannot insert). Tag: " << t );
        return false;
        }
      // We need to insert an Empty Data Element !
      //FIXME, for some public element we could do something nicer than VR:UN
      pe.DE.SetVR( VR::UN );
      pe.DE.SetByteValue( valuereplace.c_str(), (uint32_t)valuereplace.size() );
      assert( pe.DE.GetVL() == valuereplace.size() );
      }

    // We need to push_back outside of if() since Action:Replace
    // on a missing tag, means insert it !
    Internals->PositionEmptyArray.push_back( pe );

    is.close();
    }
  return true;
}

bool FileAnonymizer::ComputeRemoveTagPosition()
{
  assert( !Internals->InputFilename.empty() );
  const char *filename = Internals->InputFilename.c_str();
  assert( filename );
  const bool inplace = file_exist(Internals->OutputFilename.c_str());
  if( inplace && !Internals->RemoveTags.empty())
    {
    gdcmErrorMacro( "inplace mode requires existing tag (cannot remove)" );
    return false;
    }

  std::set<Tag>::reverse_iterator rit = Internals->RemoveTags.rbegin();
  for ( ; rit != Internals->RemoveTags.rend(); rit++ )
    {
    PositionEmpty pe;

    std::set<Tag> removeme;
    const Tag & t = *rit;
    removeme.insert( t );

    std::ifstream is( filename, std::ios::binary );
    Reader reader;
    reader.SetStream( is );
    if( !reader.ReadSelectedTags( removeme ) )
      {
      return false;
      }

    pe.EndPos = pe.BeginPos = is.tellg();
    pe.action = REMOVE;
    pe.IsTagFound = false;
    const File & f = reader.GetFile();
    const DataSet &ds = f.GetDataSet();
    const TransferSyntax &ts = f.GetHeader().GetDataSetTransferSyntax();
    Internals->TS = ts;

    pe.DE.SetTag( t );
    if( ds.FindDataElement( t ) )
      {
      const DataElement &de = ds.GetDataElement( t );
      pe.IsTagFound = true;
      pe.DE.SetVL( de.GetVL() ); // Length is not used, unless to check undefined flag
      pe.DE.SetVR( de.GetVR() );
      assert( pe.DE.GetVL().IsUndefined() == de.GetVL().IsUndefined() );
      assert( pe.DE.GetVR() == de.GetVR() );
      assert( pe.DE.GetTag() == de.GetTag() );
      if( de.GetVL().IsUndefined() )
        {
        // This is a SQ
        VL vl;
        if( ts.GetNegociatedType() == TransferSyntax::Implicit )
          {
          vl = de.GetLength<ImplicitDataElement>();
          }
        else
          {
          vl = de.GetLength<ExplicitDataElement>();
          }
        assert( pe.BeginPos > vl );
        pe.BeginPos -= vl;
        }
      else
        {
        assert( !de.GetVL().IsUndefined() );
        pe.BeginPos -= de.GetVL();
        pe.BeginPos -= 2 * de.GetVR().GetLength(); // (VR+) VL
        pe.BeginPos -= 4; // Tag
        assert( (int)pe.EndPos ==
          (int)pe.BeginPos + (int)de.GetVL() + 2 * de.GetVR().GetLength() + 4 );
        }
      Internals->PositionEmptyArray.push_back( pe );
      }
    else
      {
      // Yay no need to do anything !
      }

    is.close();
    }

  return true;
}

bool FileAnonymizer::ComputeEmptyTagPosition()
{
  // FIXME we sometime empty, attributes that are already empty...
  assert( !Internals->InputFilename.empty() );
  const char *filename = Internals->InputFilename.c_str();
  assert( filename );
  const bool inplace = file_exist(Internals->OutputFilename.c_str());
  if( inplace && !Internals->EmptyTags.empty())
    {
    gdcmErrorMacro( "inplace mode requires existing tag (cannot empty)" );
    return false;
    }

  std::set<Tag>::reverse_iterator rit = Internals->EmptyTags.rbegin();
  for ( ; rit != Internals->EmptyTags.rend(); rit++ )
    {
    PositionEmpty pe;

    std::set<Tag> removeme;
    const Tag & t = *rit;
    removeme.insert( t );

    std::ifstream is( filename, std::ios::binary );
    Reader reader;
    reader.SetStream( is );
    if( !reader.ReadSelectedTags( removeme ) )
      {
      return false;
      }

    pe.EndPos = pe.BeginPos = is.tellg();
    pe.action = EMPTY;
    pe.IsTagFound = false;
    const File & f = reader.GetFile();
    const DataSet &ds = f.GetDataSet();
    const TransferSyntax &ts = f.GetHeader().GetDataSetTransferSyntax();
    Internals->TS = ts;

    pe.DE.SetTag( t );
    if( ds.FindDataElement( t ) )
      {
      const DataElement &de = ds.GetDataElement( t );
      pe.IsTagFound = true;
      pe.DE.SetVL( de.GetVL() ); // Length is not used, unless to check undefined flag
      pe.DE.SetVR( de.GetVR() );
      assert( pe.DE.GetVL().IsUndefined() == de.GetVL().IsUndefined() );
      assert( pe.DE.GetVR() == de.GetVR() );
      assert( pe.DE.GetTag() == de.GetTag() );
      if( de.GetVL().IsUndefined() )
        {
        // This is a SQ
        VL vl;
        if( ts.GetNegociatedType() == TransferSyntax::Implicit )
          {
          vl = de.GetLength<ImplicitDataElement>();
          }
        else
          {
          vl = de.GetLength<ExplicitDataElement>();
          }
        assert( pe.BeginPos > vl );
        pe.BeginPos -= vl;
        pe.BeginPos += 4; // Tag
        if( ts.GetNegociatedType() == TransferSyntax::Implicit )
          {
          pe.BeginPos += 0;
          }
        else
          {
          pe.BeginPos += de.GetVR().GetLength();
          }
        }
      else
        {
        assert( !de.GetVL().IsUndefined() );
        pe.BeginPos -= de.GetVL();
        if( ts.GetNegociatedType() == TransferSyntax::Implicit )
          {
          pe.BeginPos -= 4;
          }
        else
          {
          pe.BeginPos -= de.GetVR().GetLength();
          assert( (int)pe.EndPos ==
            (int)pe.BeginPos + (int)de.GetVL() + de.GetVR().GetLength() );
          }
        }
      }
    else
      {
      // We need to insert an Empty Data Element !
      //FIXME, for some public element we could do something nicer than VR:UN
      pe.DE.SetVR( VR::UN );
      pe.DE.SetVL( 0 );
      }

    // We need to push_back outside of if() since Action:Empty
    // on a missing tag, means insert it !
    Internals->PositionEmptyArray.push_back( pe );

    is.close();
    }

  return true;
}

bool FileAnonymizer::Write()
{
  if( Internals->OutputFilename.empty() ) return false;
  const char *outfilename = Internals->OutputFilename.c_str();
  if( Internals->InputFilename.empty() ) return false;
  const char *filename = Internals->InputFilename.c_str();

  Internals->PositionEmptyArray.clear();

  // Compute offsets
  if( !ComputeRemoveTagPosition()
    || !ComputeEmptyTagPosition()
    || !ComputeReplaceTagPosition() )
    {
    return false;
    }

  // Make sure we will copy from lower offset to highest:
  // need to loop from the end. Sometimes a replace operation will have *exact*
  // same file offset for multiple attributes. In which case we need to insert
  // first the last attribute, and at the end the first attribute
  PositionEmpty pe_sort = {};
  std::sort (Internals->PositionEmptyArray.begin(),
    Internals->PositionEmptyArray.end(), pe_sort);

  // Step 2. Copy & skip proper portion
  std::ios::openmode om;
  const bool inplace = file_exist(outfilename);
  if( inplace )
    {
    // overwrite:
    om = std::ofstream::in | std::ofstream::out | std::ios::binary;
    }
  else
    {
    // create
    om = std::ofstream::out | std::ios::binary;
    }
  std::fstream of( outfilename, om );
  std::ifstream is( filename, std::ios::binary );
  std::streampos prev = 0;
  const TransferSyntax &ts = Internals->TS;
  std::vector<PositionEmpty>::const_iterator it =
    Internals->PositionEmptyArray.begin();
  for( ; it != Internals->PositionEmptyArray.end(); ++it )
    {
    const PositionEmpty & pe = *it;
    Action action = pe.action;

    if( pe.IsTagFound )
      {
      const DataElement & de = pe.DE;
      int vrlen = de.GetVR().GetLength();
      if( ts.GetNegociatedType() == TransferSyntax::Implicit )
        {
        vrlen = 4;
        }

      std::streampos end = pe.BeginPos;

      // FIXME: most efficient way to copy chunk of file in c++ ?
      for( int i = (int)prev; i < end; ++i)
        {
        of.put( (char)is.get() );
        }
      if( action == EMPTY )
        {
        assert( !inplace );
        // Create a 0 Value Length (VR+Tag was copied in previous loop)
        for( int i = 0; i < vrlen; ++i)
          {
          of.put( 0 );
          }
        }
      else if( action == REPLACE )
        {
        if( ts.GetSwapCode() == SwapCode::BigEndian )
          {
          if( ts.GetNegociatedType() == TransferSyntax::Implicit )
            {
            gdcmErrorMacro( "Cant write Virtual Big Endian" );
            return 1;
            }
          else
            {
            pe.DE.Write<ExplicitDataElement,SwapperDoOp>( of );
            }
          }
        else
          {
          if( ts.GetNegociatedType() == TransferSyntax::Implicit )
            {
            pe.DE.Write<ImplicitDataElement,SwapperNoOp>( of );
            }
          else
            {
            pe.DE.Write<ExplicitDataElement,SwapperNoOp>( of );
            }
          }
        }
      // Skip the Value
      assert( is.good() );
      is.seekg( pe.EndPos );
      assert( is.good() );
      prev = is.tellg();
      assert( prev == pe.EndPos );
      }
    else
      {
      std::streampos end = pe.BeginPos;

      // FIXME: most efficient way to copy chunk of file in c++ ?
      for( int i = (int)prev; i < end; ++i)
        {
        of.put( (char)is.get() );
        }
      if( ts.GetSwapCode() == SwapCode::BigEndian )
        {
        if( ts.GetNegociatedType() == TransferSyntax::Implicit )
          {
          gdcmErrorMacro( "Cant write Virtual Big Endian" );
          return 1;
          }
        else
          {
          pe.DE.Write<ExplicitDataElement,SwapperDoOp>( of );
          }
        }
      else
        {
        if( ts.GetNegociatedType() == TransferSyntax::Implicit )
          {
          pe.DE.Write<ImplicitDataElement,SwapperNoOp>( of );
          }
        else
          {
          pe.DE.Write<ExplicitDataElement,SwapperNoOp>( of );
          }
        }
      prev = is.tellg();
      }
    }

  of << is.rdbuf();
  of.close();
  is.close();

#if 0
  Reader r;
  r.SetFileName( outfilename );
  if( !r.Read() )
    {
    gdcmErrorMacro( "Output file got corrupted, please report" );
    return false;
    }
#endif

  return true;
}

} // end namespace gdcm
