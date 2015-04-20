/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPYTHONFILTER_H
#define GDCMPYTHONFILTER_H

#include <Python.h>

#include "gdcmDataElement.h"
#include "gdcmDicts.h"
#include "gdcmFile.h"

namespace gdcm
{

/**
 * \brief PythonFilter
 * PythonFilter is the class that make gdcm2.x looks more like gdcm1 and transform the binary blob
 * contained in a DataElement into a string, typically this is a nice feature to have for wrapped language
 */
class GDCM_EXPORT PythonFilter
{
public:
  PythonFilter();
  ~PythonFilter();

  void UseDictAlways(bool use) {}

  // Allow user to pass in there own dicts
  void SetDicts(const Dicts &dicts);

  // Convert to string the ByteValue contained in a DataElement
  PyObject *ToPyObject(const Tag& t) const;

  void SetFile(const File& f) { F = f; }
  File &GetFile() { return *F; }
  const File &GetFile() const { return *F; }

private:
  SmartPointer<File> F;
};

} // end namespace gdcm

#endif //GDCMPYTHONFILTER_H
