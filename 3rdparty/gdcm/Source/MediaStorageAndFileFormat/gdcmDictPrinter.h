/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDICTPRINTER_H
#define GDCMDICTPRINTER_H

#include "gdcmPrinter.h"

namespace gdcm
{

/**
 * \brief DictPrinter class
 */
// It's a sink there is no output
class GDCM_EXPORT DictPrinter : public Printer
{
public:
  DictPrinter();
  ~DictPrinter();

  void Print(std::ostream& os);

protected:
  void PrintDataElement2(std::ostream& os, const DataSet &ds, const DataElement &ide);
  void PrintDataSet2(std::ostream& os, const DataSet &ds);
};

} // end namespace gdcm

#endif //GDCMDICTPRINTER_H
