/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPARSEEXCEPTION_H
#define GDCMPARSEEXCEPTION_H

#include "gdcmException.h"
#include "gdcmDataElement.h"

// Disable clang warning "dynamic exception specifications are deprecated".
// We need to be C++03 and C++11 compatible, and if we remove the 'throw()'
// specifier we'll get an error in C++03 by not matching the superclass.
#if defined(__clang__) && defined(__has_warning)
# if __has_warning("-Wdeprecated")
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wdeprecated"
# endif
#endif

namespace gdcm
{
/**
 * \brief ParseException Standard exception handling object.
 *
 */
class ParseException : public Exception
{
public:
  ParseException()
  {
  }
  virtual ~ParseException() throw() {}

  /** Assignment operator. */
  ParseException &operator= ( const ParseException &orig )
    {
    (void)orig;
    //TODO
    return *this;
    }

  /** Equivalence operator. */
/*  virtual bool operator==( const ParseException &orig )
  {
    return true;
  }*/

/*
  // Multiple calls to what ??
  const char* what() const throw()
    {
    static std::string strwhat;
    std::ostringstream oswhat;
    oswhat << File << ":" << Line << ":\n";
    oswhat << Description;
    strwhat = oswhat.str();
    return strwhat.c_str();
    }
*/
  void SetLastElement(DataElement& de)
    {
    LastElement = de;
    }
  const DataElement& GetLastElement() const { return LastElement; }

private:
  // Store last parsed element before error:
  DataElement LastElement;
};

} // end namespace gdcm

// Undo warning suppression.
#if defined(__clang__) && defined(__has_warning)
# if __has_warning("-Wdeprecated")
#  pragma clang diagnostic pop
# endif
#endif

#endif
