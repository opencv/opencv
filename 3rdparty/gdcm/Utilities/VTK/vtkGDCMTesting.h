/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkGDCMTesting - GDCM Testing
// .SECTION Description
// GDCM Testing

// .SECTION See Also
// vtkTesting

#ifndef VTKGDCMTESTING_H
#define VTKGDCMTESTING_H

#include "vtkObject.h"

class VTK_EXPORT vtkGDCMTesting : public vtkObject
{
public:
  static vtkGDCMTesting *New();
  vtkTypeRevisionMacro(vtkGDCMTesting,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  static const char *GetVTKDataRoot();
  static const char *GetGDCMDataRoot();

//BTX
  typedef const char* const (*MD5MetaImagesType)[3];
  static const char * const * GetMD5MetaImage(unsigned int file);
//ETX
  static unsigned int GetNumberOfMD5MetaImages();

  static const char * GetMHDMD5FromFile(const char *filepath);
  static const char * GetRAWMD5FromFile(const char *filepath);

protected:
  vtkGDCMTesting();
  ~vtkGDCMTesting();

private:
  vtkGDCMTesting(const vtkGDCMTesting&);  // Not implemented.
  void operator=(const vtkGDCMTesting&);  // Not implemented.
};

#endif
