/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSEQUENCEOFFRAGMENTS_H
#define GDCMSEQUENCEOFFRAGMENTS_H

#include "gdcmValue.h"
#include "gdcmVL.h"
#include "gdcmFragment.h"
#include "gdcmBasicOffsetTable.h"

namespace gdcm
{

  // FIXME gdcmSequenceOfItems and gdcmSequenceOfFragments
  // should be rethink (duplicate code)
/**
 * \brief Class to represent a Sequence Of Fragments
 * \todo I do not enforce that Sequence of Fragments ends with a SQ end del
 */
class GDCM_EXPORT SequenceOfFragments : public Value
{
public:
  // Typdefs:
  typedef std::vector<Fragment> FragmentVector;
  typedef FragmentVector::size_type SizeType;
  typedef FragmentVector::iterator Iterator;
  typedef FragmentVector::const_iterator ConstIterator;
  Iterator Begin() { return Fragments.begin(); }
  Iterator End() { return Fragments.end(); }
  ConstIterator Begin() const { return Fragments.begin(); }
  ConstIterator End() const { return Fragments.end(); }

/// \brief constructor (UndefinedLength by default)
  SequenceOfFragments():Table(),SequenceLengthField(0xFFFFFFFF) { }

  /// \brief Returns the SQ length, as read from disk
  VL GetLength() const {
    return SequenceLengthField;
  }

  /// \brief Sets the actual SQ length
  void SetLength(VL length) {
    SequenceLengthField = length;
  }

  /// \brief Clear
  void Clear();

  /// \brief Appends a Fragment to the already added ones
  void AddFragment(Fragment const &item);

  // Compute the length of all fragments (and framents only!).
  // Basically the size of the PixelData as stored (in bytes).
  unsigned long ComputeByteLength() const;

  // Compute the length of fragments (in bytes)+ length of tag...
  // to be used for computation of Group Length
  VL ComputeLength() const;

  // Get the buffer
  bool GetBuffer(char *buffer, unsigned long length) const;
  bool GetFragBuffer(unsigned int fragNb, char *buffer, unsigned long &length) const;

  SizeType GetNumberOfFragments() const;
  const Fragment& GetFragment(SizeType num) const;

  // Write the buffer of each fragment (call WriteBuffer on all Fragments, which are
  // ByteValue). No Table information is written.
  bool WriteBuffer(std::ostream &os) const;

  const BasicOffsetTable &GetTable() const { return Table; }
  BasicOffsetTable &GetTable() { return Table; }

template <typename TSwap>
std::istream& Read(std::istream &is, bool readvalues = true)
{
  assert( SequenceLengthField.IsUndefined() );
  ReadPreValue<TSwap>(is);
  return ReadValue<TSwap>(is, readvalues);
}

template <typename TSwap>
std::istream& ReadPreValue(std::istream &is)
{
  //if( SequenceLengthField.IsUndefined() )
  // First item is the basic offset table:
  try
    {
    Table.Read<TSwap>(is);
    gdcmDebugMacro( "Table: " << Table );
    }
  catch(...)
    {
    // throw "SIEMENS Icon thingy";
    // Bug_Siemens_PrivateIconNoItem.dcm
    // First thing first let's rewind
    is.seekg(-4, std::ios::cur);
    // FF D8 <=> Start of Image (SOI) marker
    // FF E0 <=> APP0 Reserved for Application Use
    if ( Table.GetTag() == Tag(0xd8ff,0xe0ff) )
      {
      Table = BasicOffsetTable(); // clear up stuff
      //Table.SetByteValue( "", 0 );
      Fragment frag;
      if( FillFragmentWithJPEG( frag, is ) )
        {
        Fragments.push_back( frag );
        }
      return is;
      }
    else
      {
      throw "Catch me if you can";
      //assert(0);
      }
    }
  return is;
}

template <typename TSwap>
std::istream& ReadValue(std::istream &is, bool /*readvalues*/)
{
  const Tag seqDelItem(0xfffe,0xe0dd);
  // not used for now...
  Fragment frag;
  try
    {
    while( frag.Read<TSwap>(is) && frag.GetTag() != seqDelItem )
      {
      //gdcmDebugMacro( "Frag: " << frag );
      Fragments.push_back( frag );
      }
    assert( frag.GetTag() == seqDelItem && frag.GetVL() == 0 );
    }
  catch(Exception &ex)
    {
    (void)ex;
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    // that's ok ! In all cases the whole file was read, because
    // Fragment::Read only fail on eof() reached 1.
    // SIEMENS-JPEG-CorruptFrag.dcm is more difficult to deal with, we have a
    // partial fragment, read we decide to add it anyway to the stack of
    // fragments (eof was reached so we need to clear error bit)
    if( frag.GetTag() == Tag(0xfffe,0xe000)  )
      {
      gdcmWarningMacro( "Pixel Data Fragment could be corrupted. Use file at own risk" );
      Fragments.push_back( frag );
      is.clear(); // clear the error bit
      }
    // 2. GENESIS_SIGNA-JPEG-CorruptFrag.dcm
    else if ( frag.GetTag() == Tag(0xddff,0x00e0) )
      {
      assert( Fragments.size() == 1 );
      const ByteValue *bv = Fragments[0].GetByteValue();
      assert( (unsigned char)bv->GetPointer()[ bv->GetLength() - 1 ] == 0xfe );
      // Yes this is an extra copy, this is a bug anyway, go fix YOUR code
      Fragments[0].SetByteValue( bv->GetPointer(), bv->GetLength() - 1 );
      gdcmWarningMacro( "JPEG Fragment length was declared with an extra byte"
        " at the end: stripped !" );
      is.clear(); // clear the error bit
      }
    // 3. LEICA/WSI
    else if ( (frag.GetTag().GetGroup() == 0x00ff)
      && ((frag.GetTag().GetElement() & 0x00ff) == 0xe0) )
      {
      // Looks like there is a mess with offset and odd byte array
      // We are going first to backtrack one byte back, and then use a
      // ReadBacktrack function which in turn may backtrack up to 10 bytes
      // backward. This appears to be working on a set of DICOM/WSI files from
      // LEICA
      gdcmWarningMacro( "Trying to fix the even-but-odd value length bug #1" );
      assert( Fragments.size() );
      const size_t lastf = Fragments.size() - 1;
      const ByteValue *bv = Fragments[ lastf ].GetByteValue();
      const char *a = bv->GetPointer();
      gdcmAssertAlwaysMacro( (unsigned char)a[ bv->GetLength() - 1 ] == 0xfe );
      Fragments[ lastf ].SetByteValue( bv->GetPointer(), bv->GetLength() - 1 );
      is.seekg( -9, std::ios::cur );
      assert( is.good() );
      while( frag.ReadBacktrack<TSwap>(is) && frag.GetTag() != seqDelItem )
        {
        gdcmDebugMacro( "Frag: " << frag );
        Fragments.push_back( frag );
        }
      assert( frag.GetTag() == seqDelItem && frag.GetVL() == 0 );
      }
    // 4. LEICA/WSI (bis)
    else if ( frag.GetTag().GetGroup() == 0xe000 )
      {
      // Looks like there is a mess with offset and odd byte array
      // We are going first to backtrack one byte back, and then use a
      // ReadBacktrack function which in turn may backtrack up to 10 bytes
      // backward. This appears to be working on a set of DICOM/WSI files from
      // LEICA
      gdcmWarningMacro( "Trying to fix the even-but-odd value length bug #2" );
      assert( Fragments.size() );
      const size_t lastf = Fragments.size() - 1;
      const ByteValue *bv = Fragments[ lastf ].GetByteValue();
      const char *a = bv->GetPointer();
      gdcmAssertAlwaysMacro( (unsigned char)a[ bv->GetLength() - 2 ] == 0xfe );
      Fragments[ lastf ].SetByteValue( bv->GetPointer(), bv->GetLength() - 2 );
      is.seekg( -10, std::ios::cur );
      assert( is.good() );
      while( frag.ReadBacktrack<TSwap>(is) && frag.GetTag() != seqDelItem )
        {
        gdcmDebugMacro( "Frag: " << frag );
        Fragments.push_back( frag );
        }
      assert( frag.GetTag() == seqDelItem && frag.GetVL() == 0 );
      }
    // 5. LEICA/WSI (ter)
    else if ( (frag.GetTag().GetGroup() & 0x00ff) == 0x00e0
    && (frag.GetTag().GetElement() & 0xff00) == 0x0000 )
      {
      // Looks like there is a mess with offset and odd byte array
      // We are going first to backtrack one byte back, and then use a
      // ReadBacktrack function which in turn may backtrack up to 10 bytes
      // backward. This appears to be working on a set of DICOM/WSI files from
      // LEICA
      gdcmWarningMacro( "Trying to fix the even-but-odd value length bug #3" );
      assert( Fragments.size() );
      const size_t lastf = Fragments.size() - 1;
      const ByteValue *bv = Fragments[ lastf ].GetByteValue();
      const char *a = bv->GetPointer();
      gdcmAssertAlwaysMacro( (unsigned char)a[ bv->GetLength() - 3 ] == 0xfe );
      Fragments[ lastf ].SetByteValue( bv->GetPointer(), bv->GetLength() - 3 );
      is.seekg( -11, std::ios::cur );
      assert( is.good() );
      while( frag.ReadBacktrack<TSwap>(is) && frag.GetTag() != seqDelItem )
        {
        gdcmDebugMacro( "Frag: " << frag );
        Fragments.push_back( frag );
        }
      assert( frag.GetTag() == seqDelItem && frag.GetVL() == 0 );
      }
    else
      {
      // 3. gdcm-JPEG-LossLess3a.dcm: easy case, an extra tag was found
      // instead of terminator (eof is the next char)
      gdcmWarningMacro( "Reading failed at Tag:" << frag.GetTag() << " Index #"
        << Fragments.size() << " Offset " << is.tellg() << ". Use file at own risk."
        << ex.what() );
      }
#endif /* GDCM_SUPPORT_BROKEN_IMPLEMENTATION */
    }

  return is;
}

template <typename TSwap>
std::ostream const &Write(std::ostream &os) const
{
  if( !Table.Write<TSwap>(os) )
    {
    assert(0 && "Should not happen");
    return os;
    }
  for(ConstIterator it = Begin();it != End(); ++it)
    {
    it->Write<TSwap>(os);
    }
  // seq del item is not stored, write it !
  const Tag seqDelItem(0xfffe,0xe0dd);
  seqDelItem.Write<TSwap>(os);
  VL zero = 0;
  zero.Write<TSwap>(os);

  return os;
}

//#if defined(SWIGPYTHON) || defined(SWIGCSHARP) || defined(SWIGJAVA)
  // For now leave it there, this does not make sense in the C++ layer
  // Create a new object
  static SmartPointer<SequenceOfFragments> New()
  {
     return new SequenceOfFragments();
  }
//#endif

protected:
public:
  void Print(std::ostream &os) const {
    os << "SQ L= " << SequenceLengthField << "\n";
    os << "Table:" << Table << "\n";
    for(ConstIterator it = Begin();it != End(); ++it)
      {
      os << "  " << *it << "\n";
      }
    assert( SequenceLengthField.IsUndefined() );
      {
      const Tag seqDelItem(0xfffe,0xe0dd);
      VL zero = 0;
      os << seqDelItem;
      os << "\t" << zero;
      }
  }
  bool operator==(const Value &val) const
    {
    const SequenceOfFragments &sqf = dynamic_cast<const SequenceOfFragments&>(val);
    return Table == sqf.Table &&
      SequenceLengthField == sqf.SequenceLengthField &&
      Fragments == sqf.Fragments;
    }

private:
  BasicOffsetTable Table;
  VL SequenceLengthField;
  /// \brief Vector of Sequence Fragments
  FragmentVector Fragments;

private:
  bool FillFragmentWithJPEG( Fragment & frag, std::istream & is );
};

/**
 * \example DecompressJPEGFile.cs
 * This is a C# example on how to use gdcm::SequenceOfFragments
 */

} // end namespace gdcm

#endif //GDCMSEQUENCEOFFRAGMENTS_H
