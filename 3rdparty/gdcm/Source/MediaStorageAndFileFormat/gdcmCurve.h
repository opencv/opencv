/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCURVE_H
#define GDCMCURVE_H

#include "gdcmTypes.h"
#include "gdcmObject.h"

#include <vector>

namespace gdcm
{

class CurveInternal;
class ByteValue;
class DataSet;
class DataElement;
/**
 * \brief Curve class to handle element 50xx,3000 Curve Data
 *  WARNING: This is deprecated and lastly defined in PS 3.3 - 2004
 *
 *  Examples:
 *  - GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm
 *  - GE_DLX-8-MONO2-Multiframe.dcm
 *  - gdcmSampleData/Philips_Medical_Images/integris_HV_5000/xa_integris.dcm
 *  - TOSHIBA-CurveData[1-3].dcm
 */
class GDCM_EXPORT Curve : public Object
{
public:
  Curve();
  ~Curve();
  void Print(std::ostream &) const;

  void GetAsPoints(float *array) const;

  static unsigned int GetNumberOfCurves(DataSet const & ds);

  // Update curve data from dataelmenet de:
  void Update(const DataElement & de);

  void SetGroup(unsigned short group);
  unsigned short GetGroup() const;
  void SetDimensions(unsigned short dimensions);
  unsigned short GetDimensions() const;
  void SetNumberOfPoints(unsigned short numberofpoints);
  unsigned short GetNumberOfPoints() const;
  void SetTypeOfData(const char *typeofdata);
  const char *GetTypeOfData() const;
  // See PS 3.3 - 2004 - C.10.2.1.1 Type of data
  const char *GetTypeOfDataDescription() const;
  void SetCurveDescription(const char *curvedescription);
  void SetDataValueRepresentation(unsigned short datavaluerepresentation);
  unsigned short GetDataValueRepresentation() const;
  void SetCurveDataDescriptor(const uint16_t * values, size_t num);
  std::vector<unsigned short> const &GetCurveDataDescriptor() const;
  void SetCoordinateStartValue( unsigned short v );
  void SetCoordinateStepValue( unsigned short v );

  void SetCurve(const char *array, unsigned int length);

  bool IsEmpty() const;

  void Decode(std::istream &is, std::ostream &os);

  Curve(Curve const &ov);
private:
  double ComputeValueFromStartAndStep(unsigned int idx) const;
  CurveInternal *Internal;
};

} // end namespace gdcm

#endif //GDCMCURVE_H
