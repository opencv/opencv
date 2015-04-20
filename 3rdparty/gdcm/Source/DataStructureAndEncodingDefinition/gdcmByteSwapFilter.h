/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMBYTESWAPFILTER_H
#define GDCMBYTESWAPFILTER_H

#include "gdcmDataSet.h"

namespace gdcm
{

/**
 * \brief ByteSwapFilter
 * In place byte-swapping of a dataset
 * FIXME: FL status ??
 */
class GDCM_EXPORT ByteSwapFilter
{
public:
  ByteSwapFilter(DataSet& ds):DS(ds),ByteSwapTag(false) {}
  ~ByteSwapFilter();

  bool ByteSwap();
  void SetByteSwapTag(bool b) { ByteSwapTag = b; }

private:
  DataSet &DS;
  bool ByteSwapTag;

  ByteSwapFilter& operator=(const ByteSwapFilter &);
};

} // end namespace gdcm

#endif //GDCMBYTESWAPFILTER_H
