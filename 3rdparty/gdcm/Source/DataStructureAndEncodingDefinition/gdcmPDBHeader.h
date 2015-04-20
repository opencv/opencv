/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPDBHEADER_H
#define GDCMPDBHEADER_H

#include "gdcmTypes.h"
#include "gdcmDataSet.h"
#include "gdcmPDBElement.h"

namespace gdcm
{

/*
 * Everything done in this code is for the sole purpose of writing interoperable
 * software under Sect. 1201 (f) Reverse Engineering exception of the DMCA.
 * If you believe anything in this code violates any law or any of your rights,
 * please contact us (gdcm-developers@lists.sourceforge.net) so that we can
 * find a solution.
 */
//-----------------------------------------------------------------------------

class DataElement;
class PrivateTag;
/**
 * \brief Class for PDBHeader
 *
 * GEMS MR Image have an Attribute (0025,1b,GEMS_SERS_01) which store the
 * Acquisition parameter of the MR Image. It is compressed and can therefore
 * not be used as is. This class de-encapsulated the Protocol Data Block and
 * allow users to query element by name.
 *
 * \warning
 * Everything you do with this code is at your own risk, since decoding process
 * was not written from specification documents.
 *
 * \warning: the API of this class might change.
 *
 * \see CSAHeader
 */
class GDCM_EXPORT PDBHeader
{
  friend std::ostream& operator<<(std::ostream &_os, const PDBHeader &d);
public :
  PDBHeader() {}
  ~PDBHeader() {}

  /// Load the PDB Header from a DataElement of a DataSet
  bool LoadFromDataElement(DataElement const &de);

  /// Print
  void Print(std::ostream &os) const;

  /// Return the Private Tag where the PDB header is stored within a DICOM DataSet
  static const PrivateTag & GetPDBInfoTag();

  /// Lookup in the PDB header if a PDB element match the name 'name':
  /// \warning Case Sensitive
  const PDBElement &GetPDBElementByName(const char *name);

  /// Return true if the PDB element matching name is found or not
  bool FindPDBElementByName(const char *name);

protected:
  const PDBElement& GetPDBEEnd() const;

private:
  int readprotocoldatablock(const char *input, size_t inputlen, bool verbose);
  std::vector<PDBElement> InternalPDBDataSet;
  static PDBElement PDBEEnd;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const PDBHeader &d)
{
  d.Print( os );
  return os;
}

} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMPDBHEADER_H
