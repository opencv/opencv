/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMJSON_H
#define GDCMJSON_H

/*
See Sup 166 (QIDO-RS)
http://www.dclunie.com/dicom-status/status.html#Supplement166
*/

#include "gdcmFile.h"
#include "gdcmDataElement.h"

namespace gdcm
{

class JSONInternal;
class GDCM_EXPORT JSON
{
public:
  JSON();
  ~JSON();

  bool GetPrettyPrint() const;
  void SetPrettyPrint(bool onoff);
  void PrettyPrintOn();
  void PrettyPrintOff();

  bool Code(DataSet const & in, std::ostream & os);
  bool Decode(std::istream & is, DataSet & out);

private:
  JSONInternal *Internals;
};

} // end namespace gdcm

#endif //GDCMXMLPRINTER_H
