/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMMedicalImageProperties.h"
#include "vtkObjectFactory.h"

#include "gdcmFile.h"

//----------------------------------------------------------------------------
vtkCxxRevisionMacro(vtkGDCMMedicalImageProperties, "1.21")
vtkStandardNewMacro(vtkGDCMMedicalImageProperties)

class vtkGDCMMedicalImagePropertiesInternals
{
public:
  std::vector< gdcm::SmartPointer<gdcm::File> > Files;
};

//----------------------------------------------------------------------------
vtkGDCMMedicalImageProperties::vtkGDCMMedicalImageProperties()
{
  this->Internals = new vtkGDCMMedicalImagePropertiesInternals;
}

//----------------------------------------------------------------------------
vtkGDCMMedicalImageProperties::~vtkGDCMMedicalImageProperties()
{
  if (this->Internals)
    {
    delete this->Internals;
    this->Internals = NULL;
    }
  this->Clear();
}

//----------------------------------------------------------------------------
void vtkGDCMMedicalImageProperties::Clear()
{
  this->Superclass::Clear();
}

//----------------------------------------------------------------------------
void vtkGDCMMedicalImageProperties::PushBackFile(gdcm::File const &f)
{
  this->Internals->Files.push_back( f );
  size_t i = this->Internals->Files.size();
  gdcm::DataSet &ds = this->Internals->Files[ i - 1 ]->GetDataSet();
  ds.Remove( gdcm::Tag( 0x7fe0, 0x0010 ) );
}

//----------------------------------------------------------------------------
gdcm::File const & vtkGDCMMedicalImageProperties::GetFile(unsigned int t)
{
  return *this->Internals->Files[ t ];
}

//----------------------------------------------------------------------------
void vtkGDCMMedicalImageProperties::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
