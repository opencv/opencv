/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMBYTEVALUE_H
#define GDCMBYTEVALUE_H

#include "gdcmValue.h"
#include "gdcmTrace.h"
#include "gdcmVL.h"

#include <vector>
#include <iterator>
#include <iomanip>

//#include <stdlib.h> // abort

namespace gdcm
{
/**
 * \brief Class to represent binary value (array of bytes)
 * \note
 */
class GDCM_EXPORT ByteValue : public Value
{
public:
  ByteValue(const char* array = 0, VL const &vl = 0):
    Internal(array, array+vl),Length(vl) {
      if( vl.IsOdd() )
        {
        gdcmDebugMacro( "Odd length" );
        Internal.resize(vl+1);
        Length++;
        }
  }

  /// \warning casting to uint32_t
  ByteValue(std::vector<char> &v):Internal(v),Length((uint32_t)v.size()) {}
  //ByteValue(std::ostringstream const &os) {
  //  (void)os;
  //   assert(0); // TODO
  //}
  ~ByteValue() {
    Internal.clear();
  }

  // When 'dumping' dicom file we still have some information from
  // Either the VR: eg LO (private tag)
  void PrintASCII(std::ostream &os, VL maxlength ) const;

  void PrintHex(std::ostream &os, VL maxlength) const;

  // Either from Element Number (== 0x0000)
  void PrintGroupLength(std::ostream &os) {
    assert( Length == 2 );
    (void)os;
  }

  bool IsEmpty() const {
#if 0
    if( Internal.empty() ) assert( Length == 0 );
    return Internal.empty();
#else
  return Length == 0;
#endif
  }
  VL GetLength() const { return Length; }

  VL ComputeLength() const { return Length + Length % 2; }
  // Does a reallocation
  void SetLength(VL vl) {
    VL l(vl);
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    // CompressedLossy.dcm
    if( l.IsUndefined() ) throw Exception( "Impossible" );
    if ( l.IsOdd() ) {
      gdcmDebugMacro(
        "BUGGY HEADER: Your dicom contain odd length value field." );
      ++l;
      }
#else
    assert( !l.IsUndefined() && !l.IsOdd() );
#endif
    // I cannot use reserve for now. I need to implement:
    // STL - vector<> and istream
    // http://groups.google.com/group/comp.lang.c++/msg/37ec052ed8283e74
//#define SHORT_READ_HACK
    try
      {
#ifdef SHORT_READ_HACK
    if( l <= 0xff )
#endif
      Internal.resize(l);
      //Internal.reserve(l);
      }
    catch(...)
      {
      //throw Exception("Impossible to allocate: " << l << " bytes." );
      throw Exception("Impossible to allocate" );
      }
    // Keep the exact length
    Length = vl;
  }

  operator const std::vector<char>& () const { return Internal; }

  ByteValue &operator=(const ByteValue &val) {
    Internal = val.Internal;
    Length = val.Length;
    return *this;
    }

  bool operator==(const ByteValue &val) const {
    if( Length != val.Length )
      return false;
    if( Internal == val.Internal )
      return true;
    return false;
    }
  bool operator==(const Value &val) const
    {
    const ByteValue &bv = dynamic_cast<const ByteValue&>(val);
    return Length == bv.Length && Internal == bv.Internal;
    }

  void Append(ByteValue const & bv);

  void Clear() {
    Internal.clear();
  }
  // Use that only if you understand what you are doing
  const char *GetPointer() const {
    if(!Internal.empty()) return &Internal[0];
    return 0;
  }
  void Fill(char c) {
    //if( Internal.empty() ) return;
    std::vector<char>::iterator it = Internal.begin();
    for(; it != Internal.end(); ++it) *it = c;
  }
  bool GetBuffer(char *buffer, unsigned long length) const;
  bool WriteBuffer(std::ostream &os) const {
    if( Length ) {
      //assert( Internal.size() <= Length );
      assert( !(Internal.size() % 2) );
      os.write(&Internal[0], Internal.size() );
      }
    return true;
  }

  template <typename TSwap, typename TType>
  std::istream &Read(std::istream &is, bool readvalues = true) {
    // If Length is odd we have detected that in SetLength
    // and calling std::vector::resize make sure to allocate *AND*
    // initialize values to 0 so we are sure to have a \0 at the end
    // even in this case
    if(Length)
      {
      if( readvalues )
        {
        is.read(&Internal[0], Length);
        assert( Internal.size() == Length || Internal.size() == Length + 1 );
        TSwap::SwapArray((TType*)&Internal[0], Internal.size() / sizeof(TType) );
        }
      else
        {
        is.seekg(Length, std::ios::cur);
        }
      }
    return is;
  }

  template <typename TSwap>
  std::istream &Read(std::istream &is) {
    return Read<TSwap,uint8_t>(is);
  }


  template <typename TSwap, typename TType>
  std::ostream const &Write(std::ostream &os) const {
    assert( !(Internal.size() % 2) );
    if( !Internal.empty() ) {
      //os.write(&Internal[0], Internal.size());
      std::vector<char> copy = Internal;
      TSwap::SwapArray((TType*)&copy[0], Internal.size() / sizeof(TType) );
      os.write(&copy[0], copy.size());
      }
    return os;
  }

  template <typename TSwap>
  std::ostream const &Write(std::ostream &os) const {
    return Write<TSwap,uint8_t>(os);
  }

  /**
   * \brief  Checks whether a 'ByteValue' is printable or not (in order
   *         to avoid corrupting the terminal of invocation when printing)
   *         I don't think this function is working since it does not handle
   *         UNICODE or character set...
   */
  bool IsPrintable(VL length) const {
    assert( length <= Length );
    for(unsigned int i=0; i<length; i++)
      {
      if ( i == (length-1) && Internal[i] == '\0') continue;
      if ( !( isprint((unsigned char)Internal[i]) || isspace((unsigned char)Internal[i]) ) )
        {
        //gdcmWarningMacro( "Cannot print :" << i );
        return false;
        }
      }
    return true;
    }

  /**To Print Values in Native DICOM format **/
  void PrintPNXML(std::ostream &os) const;
  void PrintASCIIXML(std::ostream &os) const;
  void PrintHexXML(std::ostream &os) const;
protected:
  void Print(std::ostream &os) const {
  // This is perfectly valid to have a Length = 0 , so we cannot check
  // the length for printing
  if( !Internal.empty() )
    {
    if( IsPrintable(Length) )
      {
      // WARNING: Internal.end() != Internal.begin()+Length
      std::vector<char>::size_type length = Length;
      if( Internal.back() == 0 ) --length;
      std::copy(Internal.begin(), Internal.begin()+length,
        std::ostream_iterator<char>(os));
      }
    else
      os << "Loaded:" << Internal.size();
    }
  else
    {
    //os << "Not Loaded";
    os << "(no value available)";
    }
  }
/*
//Introduce check for invalid XML characters
friend std::ostream& operator<<(std::ostream &os,const char c);
*/

  void SetLengthOnly(VL vl) {
    Length = vl;
  }

private:
  std::vector<char> Internal;

  // WARNING Length IS NOT Internal.size() some *featured* DICOM
  // implementation define odd length, we always load them as even number
  // of byte, so we need to keep the right Length
  VL Length;
};

} // end namespace gdcm

#endif //GDCMBYTEVALUE_H
