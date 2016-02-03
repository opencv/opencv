/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSTRING_H
#define GDCMSTRING_H

#include "gdcmTypes.h"
#include "gdcmStaticAssert.h"

namespace gdcm
{

/**
 * \brief String
 *
 * \note TDelimiter template parameter is used to separate multiple String (VM1 >)
 *      TMaxLength is only a hint. Noone actually respect the max length
 *      TPadChar is the string padding (0 or space)
 */
template <char TDelimiter = '\\', unsigned int TMaxLength = 64, char TPadChar = ' '>
class /*GDCM_EXPORT*/ String : public std::string /* PLEASE do not export me */
{
  // UI wants \0 for pad character, while ASCII ones wants space char... do not allow anything else
  GDCM_STATIC_ASSERT( TPadChar == ' ' || TPadChar == 0 );

public:
  // typedef are not inherited:
  typedef std::string::value_type             value_type;
  typedef std::string::pointer                pointer;
  typedef std::string::reference              reference;
  typedef std::string::const_reference        const_reference;
  typedef std::string::size_type              size_type;
  typedef std::string::difference_type        difference_type;
  typedef std::string::iterator               iterator;
  typedef std::string::const_iterator         const_iterator;
  typedef std::string::reverse_iterator       reverse_iterator;
  typedef std::string::const_reverse_iterator const_reverse_iterator;

  /// String constructors.
  String(): std::string() {}
  String(const value_type* s): std::string(s)
  {
  if( size() % 2 )
    {
    push_back( TPadChar );
    }
  }
  String(const value_type* s, size_type n): std::string(s, n)
  {
  // We are being passed a const char* pointer, so s[n] == 0 (garanteed!)
  if( n % 2 )
    {
    push_back( TPadChar );
    }
  }
  String(const std::string& s, size_type pos=0, size_type n=npos):
    std::string(s, pos, n)
  {
  // FIXME: some users might already have padded the string 's' with a trailing \0...
  if( size() % 2 )
    {
    push_back( TPadChar );
    }
  }

  /// WARNING: Trailing \0 might be lost in this operation:
  operator const char *() const { return this->c_str(); }

  /// return if string is valid
  bool IsValid() const {
    // Check Length:
    size_type l = size();
    if( l > TMaxLength ) return false;
    return true;
  }

  gdcm::String<TDelimiter, TMaxLength, TPadChar> Truncate() const {
    if( IsValid() ) return *this;
    std::string str = *this; // copy
    str.resize( TMaxLength );
    return str;
  }

  /// Trim function is required to return a std::string object, otherwise we
  /// could not create a gdcm::String object with an odd number of bytes...
  std::string Trim() const {
    std::string str = *this; // copy
    std::string::size_type pos1 = str.find_first_not_of(' ');
    std::string::size_type pos2 = str.find_last_not_of(' ');
    str = str.substr( (pos1 == std::string::npos) ? 0 : pos1,
      (pos2 == std::string::npos) ? (str.size() - 1) : (pos2 - pos1 + 1));
    return str;
  }

  static std::string Trim(const char *input) {
    if( !input ) return "";
    std::string str = input;
    std::string::size_type pos1 = str.find_first_not_of(' ');
    std::string::size_type pos2 = str.find_last_not_of(' ');
    str = str.substr( (pos1 == std::string::npos) ? 0 : pos1,
      (pos2 == std::string::npos) ? (str.size() - 1) : (pos2 - pos1 + 1));
    return str;
  }

};
template <char TDelimiter, unsigned int TMaxLength, char TPadChar>
inline std::istream& operator>>(std::istream &is, String<TDelimiter,TMaxLength,TPadChar> &ms)
{
  if(is)
    {
    std::getline(is, ms, TDelimiter);
    // no such thing as std::get where the delim char would be left, so I need to manually add it back...
    // hopefully this is the right thing to do (no overhead)
    if( !is.eof() ) is.putback( TDelimiter );
    }
  return is;
}
//template <char TDelimiter = EOF, unsigned int TMaxLength = 64, char TPadChar = ' '>
//String String::Trim() const
//{
//  String s;
//  return s;
//}

} // end namespace gdcm

#endif //GDCMSTRING_H
