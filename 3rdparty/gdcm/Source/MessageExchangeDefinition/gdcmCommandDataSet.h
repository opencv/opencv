/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCOMMANDDATASET_H
#define GDCMCOMMANDDATASET_H

#include "gdcmDataSet.h"
#include "gdcmDataElement.h"

namespace gdcm
{
/**
 * \brief Class to represent a Command DataSet
 *
 * \see DataSet
 */
class GDCM_EXPORT CommandDataSet : public DataSet
{
public:
  CommandDataSet() {}
 ~CommandDataSet() {}

  friend std::ostream &operator<<(std::ostream &_os, const CommandDataSet &_val);

  // FIXME: no virtual function means: duplicate code...
  void Insert(const DataElement& de) {
    if( de.GetTag().GetGroup() == 0x0000 )
      {
      InsertDataElement( de );
      }
    else
      {
      gdcmErrorMacro( "Cannot add element with group != 0x0000 in the command dataset : " << de );
      }
  }
  void Replace(const DataElement& de) {
    Remove(de.GetTag());
    Insert(de);
  }

  /// Read
  std::istream &Read(std::istream &is);

  /// Write
  std::ostream &Write(std::ostream &os) const;

protected:
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const CommandDataSet &val)
{
  val.Print( os );
  return os;
}

} // end namespace gdcm

#endif //GDCMFILEMETAINFORMATION_H
