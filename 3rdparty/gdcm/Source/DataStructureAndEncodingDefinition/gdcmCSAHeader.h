/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCSAHEADER_H
#define GDCMCSAHEADER_H

#include "gdcmTypes.h"
#include "gdcmDataSet.h"
#include "gdcmCSAElement.h"

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
 * \brief Class for CSAHeader
 * \details SIEMENS store private information in tag (0x0029,0x10,"SIEMENS CSA
 * HEADER") this class is meant for user wishing to access values stored within
 * this private attribute.
 * There are basically two main 'format' for this attribute : SV10/NOMAGIC and
 * DATASET_FORMAT SV10 and NOMAGIC are from a user prospective identical, see
 * CSAHeader.xml for possible name / value stored in this format.
 * DATASET_FORMAT is in fact simply just another DICOM dataset (implicit) with
 * -currently unknown- value. This can be only be printed for now.
 *
 * \warning
 * Everything you do with this code is at your own risk, since decoding process
 * was not written from specification documents.
 *
 * \warning the API of this class might change.
 *
 * \todo
 * MrEvaProtocol in 29,1020 contains ^M that would be nice to get rid of on UNIX system...
 *
 * \see PDBHeader
 *
 * External references:
 * 5.1.3.2.4.1 MEDCOM History Information
 * and 5.1.4.3 CSA Non-Image Module
 * in
 * http://tamsinfo.toshiba.com/docrequest/pdf/E.Soft_v2.0.pdf
 */
class GDCM_EXPORT CSAHeader
{
  friend std::ostream& operator<<(std::ostream &_os, const CSAHeader &d);
public :
  CSAHeader():InternalDataSet(),InternalType(UNKNOWN),InterfileData(0) {};
  ~CSAHeader() {};

  /// Divers format of CSAHeader as found 'in the wild'
  typedef enum {
    UNKNOWN = 0,
    SV10,
    NOMAGIC,
    DATASET_FORMAT,
    INTERFILE,
    ZEROED_OUT
  } CSAHeaderType;

  template <typename TSwap>
  std::istream &Read(std::istream &is);

  template <typename TSwap>
  const std::ostream &Write(std::ostream &os) const;

  /// Decode the CSAHeader from element 'de'
  bool LoadFromDataElement(DataElement const &de);

  /// Print the CSAHeader (use only if Format == SV10 or NOMAGIC)
  void Print(std::ostream &os) const;

  /// Return the DataSet output (use only if Format == DATASET_FORMAT )
  const DataSet& GetDataSet() const { return InternalDataSet; }

  /// Return the string output (use only if Format == Interfile)
  const char * GetInterfile() const { return InterfileData; }

  /// return the format of the CSAHeader
  /// SV10 and NOMAGIC are equivalent.
  CSAHeaderType GetFormat() const;

  /// Return the private tag used by SIEMENS to store the CSA Image Header
  /// This is: PrivateTag(0x0029,0x0010,"SIEMENS CSA HEADER");
  static const PrivateTag & GetCSAImageHeaderInfoTag();

  /// Return the private tag used by SIEMENS to store the CSA Series Header
  /// This is: PrivateTag(0x0029,0x0020,"SIEMENS CSA HEADER");
  static const PrivateTag & GetCSASeriesHeaderInfoTag();

  /// Return the private tag used by SIEMENS to store the CSA Data Info
  /// This is: PrivateTag(0x0029,0x0010,"SIEMENS CSA NON-IMAGE");
  static const PrivateTag & GetCSADataInfo();

  /// Return the CSAElement corresponding to name 'name'
  /// \warning Case Sensitive
  const CSAElement &GetCSAElementByName(const char *name);

  /// Return true if the CSA element matching 'name' is found or not
  /// \warning Case Sensitive
  bool FindCSAElementByName(const char *name);

protected:
  const CSAElement& GetCSAEEnd() const;

private:
  std::set<CSAElement> InternalCSADataSet;
  DataSet InternalDataSet;
  CSAHeaderType InternalType;
  Tag DataElementTag;
  static CSAElement CSAEEnd;
  const char *InterfileData;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const CSAHeader &d)
{
  d.Print( os );
  return os;
}

} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMCSAHEADER_H
