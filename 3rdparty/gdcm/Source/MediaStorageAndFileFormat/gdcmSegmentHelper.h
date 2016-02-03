/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSEGMENTHELPER_H
#define GDCMSEGMENTHELPER_H

#include <string>

namespace gdcm
{

namespace SegmentHelper
{

/**
  * \brief  This structure defines a basic coded entry with all of its attributes.
  *
  * \see  PS 3.3 section 8.8.
  */
struct BasicCodedEntry
{
  /**
    * \brief Constructor.
    */
  BasicCodedEntry():
    CV(""),
    CSD(""),
    CSV(""),
    CM("")
  {}

  /**
    * \brief constructor which defines type 1 attributes.
    */
  BasicCodedEntry(const char * a_CV,
                  const char * a_CSD,
                  const char * a_CM):
    CV(a_CV),
    CSD(a_CSD),
    CSV(""),
    CM(a_CM)
  {}

  /**
    * \brief constructor which defines attributes.
    */
  BasicCodedEntry(const char * a_CV,
                  const char * a_CSD,
                  const char * a_CSV,
                  const char * a_CM):
    CV(a_CV),
    CSD(a_CSD),
    CSV(a_CSV),
    CM(a_CM)
  {}

  /**
    * \brief  Check if each attibutes of the basic coded entry is defined.
    *
    * \param  checkOptionalAttributes Check also type 1C attributes.
    */
  bool IsEmpty(const bool checkOptionalAttributes = false) const;


  //**      Members     **//
  // 0008 0100 1   Code Value
  std::string CV;   /// Code Value attribute
  // 0008 0102 1   Coding Scheme Designator
  std::string CSD;  /// Coding Scheme Designator attribute
  // 0008 0103 1C  Coding Scheme Version
  std::string CSV;  /// Coding Scheme Version attribute
  // 0008 0104 1   Code Meaning
  std::string CM;   /// Code Meaning attribute
};

} // end of SegmentHelper namespace

} // end of gdcm namespace

#endif // GDCMSEGMENTHELPER_H
