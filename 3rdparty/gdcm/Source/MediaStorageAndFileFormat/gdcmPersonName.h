/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMPERSONNAME_H
#define GDCMPERSONNAME_H

#include "gdcmTypes.h"
#include <vector>
#include <algorithm> // std::min
#include <string.h> // strlen

namespace gdcm
{

/**
 * \brief PersonName class
 */
class GDCM_EXPORT PersonName
{
public:
  static const unsigned int MaxNumberOfComponents = 5;
  static const unsigned int MaxLength = 64;
  char Component[MaxNumberOfComponents][MaxLength+1];
  static const char Separator = '^';
  static const char Padding   = ' ';

  unsigned int GetNumberOfComponents() const {
    unsigned int r = 0;
    for(unsigned int i = 0; i < 5; ++i) {
      if( *Component[i] != '\0' ) r = i;
      }
    return r+1;
  }
  unsigned int GetMaxLength() const { return MaxLength; };
  void SetBlob(const std::vector<char>& v) {
  (void)v;
    //assert(0); //TODO
  }
  void SetComponents(const char *comp1 = "",
    const char *comp2 = "",
    const char *comp3 = "",
    const char *comp4 = "",
    const char *comp5 = "") {
    const char *components[5] = { comp1, comp2, comp3, comp4, comp5 };
    SetComponents( components );
  }
  void SetComponents(const char *components[]) {
    for(unsigned int i = 0; i < 5; ++i) {
      //strncpy(Component[i], components[i], std::min( (unsigned int)strlen(components[i]), GetMaxLength() ) );
      assert( strlen(components[i]) < GetMaxLength() );
      strcpy(Component[i], components[i]);
      assert( strlen(Component[i]) < GetMaxLength() );
      }
  }
  void Print(std::ostream &os) const
  {
    //os << "Family Name Complex: " << Component[0] << std::endl;
    //os << "Given  Name Complex: " << Component[1] << std::endl;
    //os << "Middle Name        : " << Component[2] << std::endl;
    //os << "Name Suffix        : " << Component[3] << std::endl;
    //os << "Name Prefix        : " << Component[4] << std::endl;
    os << Component[0] << '^';
    os << Component[1] << '^';
    os << Component[2] << '^';
    os << Component[3] << '^';
    os << Component[4];
  }
};

} // end namespace gdcm

#endif //GDCMPERSONNAME_H
