/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTAGPATH_H
#define GDCMTAGPATH_H

#include "gdcmTag.h"

#include <vector>

namespace gdcm
{

/**
 * \brief class to handle a path of tag.
 *
 * Any Resemblance to Existing XPath is Purely Coincidental
 * ftp://medical.nema.org/medical/dicom/supps/sup118_pc.pdf
 */
class GDCM_EXPORT TagPath
{
public:
  TagPath();
  ~TagPath();
  void Print(std::ostream &) const;

  /// "/0018,0018/"...
  /// No space allowed, comma is use to separate tag group
  /// from tag element and slash is used to separate tag
  /// return false if invalid
  bool ConstructFromString(const char *path);

  /// Return if path is valid or not
  static bool IsValid(const char *path);

  /// Construct from a list of tags
  bool ConstructFromTagList(Tag const *l, unsigned int n);

  bool Push(Tag const & t);
  bool Push(unsigned int itemnum);

private:
  std::vector<Tag> Path;
};

} // end namespace gdcm

#endif //GDCMTAGPATH_H
