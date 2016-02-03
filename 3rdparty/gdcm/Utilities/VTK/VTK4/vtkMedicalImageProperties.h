/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*=========================================================================

  Portions of this file are subject to the VTK Toolkit Version 3 copyright.

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMedicalImageProperties.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkMedicalImageProperties - some medical image properties.
// .SECTION Description
// vtkMedicalImageProperties is a helper class that can be used by medical
// image readers and applications to encapsulate medical image/acquisition
// properties. Later on, this should probably be extended to add
// any user-defined property.
// .SECTION See Also
// vtkMedicalImageReader2

#ifndef VTKMEDICALIMAGEPROPERTIES_H
#define VTKMEDICALIMAGEPROPERTIES_H

#ifdef __vtkMedicalImageProperties_h
#error Something went terribly wrong
#endif

#include "vtkObject.h"

class vtkMedicalImagePropertiesInternals;

class VTK_IO_EXPORT vtkMedicalImageProperties : public vtkObject
{
public:
  static vtkMedicalImageProperties *New();
  vtkTypeRevisionMacro(vtkMedicalImageProperties,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Convenience method to reset all fields to an emptry string/value
  virtual void Clear();

  // Description:
  // Patient name
  // For ex: DICOM (0010,0010) = DOE,JOHN
  vtkSetStringMacro(PatientName);
  vtkGetStringMacro(PatientName);

  // Description:
  // Patient ID
  // For ex: DICOM (0010,0020) = 1933197
  vtkSetStringMacro(PatientID);
  vtkGetStringMacro(PatientID);

  // Description:
  // Patient age
  // Format: nnnD, nnW, nnnM or nnnY (eventually nnD, nnW, nnY)
  //         with D (day), M (month), W (week), Y (year)
  // For ex: DICOM (0010,1010) = 031Y
  vtkSetStringMacro(PatientAge);
  vtkGetStringMacro(PatientAge);

  // Description:
  // Take as input a string in VR=AS (DICOM PS3.5) and extract either
  // different fields namely: year month week day
  // Return 0 on error, 1 on success
  // One can test fields if they are different from -1 upon success
  static int GetAgeAsFields(const char *age, int &year, int &month, int &week, int &day);

  // For Tcl:
  // From C++ use GetPatientAge + GetAgeAsField
  // Those function parse a DICOM string, and return the value of the number expressed
  // this is either expressed in year, month or days. Thus if a string is expressed in years
  // GetPatientAgeDay/GetPatientAgeWeek/GetPatientAgeMonth will return 0
  int GetPatientAgeYear();
  int GetPatientAgeMonth();
  int GetPatientAgeWeek();
  int GetPatientAgeDay();

  // Description:
  // Patient sex
  // For ex: DICOM (0010,0040) = M
  vtkSetStringMacro(PatientSex);
  vtkGetStringMacro(PatientSex);

  // Description:
  // Patient birth date
  // Format: yyyymmdd
  // For ex: DICOM (0010,0030) = 19680427
  vtkSetStringMacro(PatientBirthDate);
  vtkGetStringMacro(PatientBirthDate);

  // For Tcl:
  // From C++ use GetPatientBirthDate + GetDateAsFields
  int GetPatientBirthDateYear();
  int GetPatientBirthDateMonth();
  int GetPatientBirthDateDay();

  // Description:
  // Study Date
  // Format: yyyymmdd
  // For ex: DICOM (0008,0020) = 20030617
  vtkSetStringMacro(StudyDate);
  vtkGetStringMacro(StudyDate);

  // Description:
  // Acquisition Date
  // Format: yyyymmdd
  // For ex: DICOM (0008,0022) = 20030617
  vtkSetStringMacro(AcquisitionDate);
  vtkGetStringMacro(AcquisitionDate);

  // For Tcl:
  // From C++ use GetAcquisitionDate + GetDateAsFields
  int GetAcquisitionDateYear();
  int GetAcquisitionDateMonth();
  int GetAcquisitionDateDay();

  // Description:
  // Study Time
  // Format: hhmmss.frac (any trailing component(s) can be ommited)
  // For ex: DICOM (0008,0030) = 162552.0705 or 230012, or 0012
  vtkSetStringMacro(StudyTime);
  vtkGetStringMacro(StudyTime);

  // Description:
  // Acquisition time
  // Format: hhmmss.frac (any trailing component(s) can be ommited)
  // For ex: DICOM (0008,0032) = 162552.0705 or 230012, or 0012
  vtkSetStringMacro(AcquisitionTime);
  vtkGetStringMacro(AcquisitionTime);

  // Description:
  // Image Date aka Content Date
  // Format: yyyymmdd
  // For ex: DICOM (0008,0023) = 20030617
  vtkSetStringMacro(ImageDate);
  vtkGetStringMacro(ImageDate);

  // For Tcl:
  // From C++ use GetImageDate + GetDateAsFields
  int GetImageDateYear();
  int GetImageDateMonth();
  int GetImageDateDay();

  // Description:
  // Take as input a string in ISO 8601 date (YYYY/MM/DD) and extract the
  // different fields namely: year month day
  // Return 0 on error, 1 on success
  static int GetDateAsFields(const char *date, int &year, int &month, int &day);

  // Description:
  // Take as input a string in ISO 8601 date (YYYY/MM/DD) and construct a
  // locale date based on the different fields (see GetDateAsFields to extract
  // different fields)
  // Return 0 on error, 1 on success
  static int GetDateAsLocale(const char *date, char *locale);

  // Description:
  // Image Time
  // Format: hhmmss.frac (any trailing component(s) can be ommited)
  // For ex: DICOM (0008,0033) = 162552.0705 or 230012, or 0012
  vtkSetStringMacro(ImageTime);
  vtkGetStringMacro(ImageTime);

  // Description:
  // Image number
  // For ex: DICOM (0020,0013) = 1
  vtkSetStringMacro(ImageNumber);
  vtkGetStringMacro(ImageNumber);

  // Description:
  // Series number
  // For ex: DICOM (0020,0011) = 902
  vtkSetStringMacro(SeriesNumber);
  vtkGetStringMacro(SeriesNumber);

  // Description:
  // Series Description
  // User provided description of the Series
  // For ex: DICOM (0008,103e) = SCOUT
  vtkSetStringMacro(SeriesDescription);
  vtkGetStringMacro(SeriesDescription);

  // Description:
  // Study ID
  // For ex: DICOM (0020,0010) = 37481
  vtkSetStringMacro(StudyID);
  vtkGetStringMacro(StudyID);

  // Description:
  // Study description
  // For ex: DICOM (0008,1030) = BRAIN/C-SP/FACIAL
  vtkSetStringMacro(StudyDescription);
  vtkGetStringMacro(StudyDescription);

  // Description:
  // Modality
  // For ex: DICOM (0008,0060)= CT
  vtkSetStringMacro(Modality);
  vtkGetStringMacro(Modality);

  // Description:
  // Manufacturer
  // For ex: DICOM (0008,0070) = Siemens
  vtkSetStringMacro(Manufacturer);
  vtkGetStringMacro(Manufacturer);

  // Description:
  // Manufacturer's Model Name
  // For ex: DICOM (0008,1090) = LightSpeed QX/i
  vtkSetStringMacro(ManufacturerModelName);
  vtkGetStringMacro(ManufacturerModelName);

  // Description:
  // Station Name
  // For ex: DICOM (0008,1010) = LSPD_OC8
  vtkSetStringMacro(StationName);
  vtkGetStringMacro(StationName);

  // Description:
  // Institution Name
  // For ex: DICOM (0008,0080) = FooCity Medical Center
  vtkSetStringMacro(InstitutionName);
  vtkGetStringMacro(InstitutionName);

  // Description:
  // Convolution Kernel (or algorithm used to reconstruct the data)
  // For ex: DICOM (0018,1210) = Bone
  vtkSetStringMacro(ConvolutionKernel);
  vtkGetStringMacro(ConvolutionKernel);

  // Description:
  // Slice Thickness (Nominal reconstructed slice thickness, in mm)
  // For ex: DICOM (0018,0050) = 0.273438
  vtkSetStringMacro(SliceThickness);
  vtkGetStringMacro(SliceThickness);
  virtual double GetSliceThicknessAsDouble();

  // Description:
  // Peak kilo voltage output of the (x-ray) generator used
  // For ex: DICOM (0018,0060) = 120
  vtkSetStringMacro(KVP);
  vtkGetStringMacro(KVP);

  // Description:
  // Gantry/Detector tilt (Nominal angle of tilt in degrees of the scanning
  // gantry.)
  // For ex: DICOM (0018,1120) = 15
  vtkSetStringMacro(GantryTilt);
  vtkGetStringMacro(GantryTilt);
  virtual double GetGantryTiltAsDouble();

  // Description:
  // Echo Time
  // (Time in ms between the middle of the excitation pulse and the peak of
  // the echo produced)
  // For ex: DICOM (0018,0081) = 105
  vtkSetStringMacro(EchoTime);
  vtkGetStringMacro(EchoTime);

  // Description:
  // Echo Train Length
  // (Number of lines in k-space acquired per excitation per image)
  // For ex: DICOM (0018,0091) = 35
  vtkSetStringMacro(EchoTrainLength);
  vtkGetStringMacro(EchoTrainLength);

  // Description:
  // Repetition Time
  // The period of time in msec between the beginning of a pulse sequence and
  // the beginning of the succeeding (essentially identical) pulse sequence.
  // For ex: DICOM (0018,0080) = 2040
  vtkSetStringMacro(RepetitionTime);
  vtkGetStringMacro(RepetitionTime);

  // Description:
  // Exposure time (time of x-ray exposure in msec)
  // For ex: DICOM (0018,1150) = 5
  vtkSetStringMacro(ExposureTime);
  vtkGetStringMacro(ExposureTime);

  // Description:
  // X-ray tube current (in mA)
  // For ex: DICOM (0018,1151) = 400
  vtkSetStringMacro(XRayTubeCurrent);
  vtkGetStringMacro(XRayTubeCurrent);

  // Description:
  // Exposure (The exposure expressed in mAs, for example calculated
  // from Exposure Time and X-ray Tube Current)
  // For ex: DICOM (0018,1152) = 114
  vtkSetStringMacro(Exposure);
  vtkGetStringMacro(Exposure);

  // Interface to allow insertion of user define values, for instance in DICOM one would want to
  // store the Protocol Name (0018,1030), in this case one would do:
  // AddUserDefinedValue( "Protocol Name", "T1W/SE/1024" );
  void AddUserDefinedValue(const char *name, const char *value);
  // Get a particular user value
  const char *GetUserDefinedValue(const char *name);
  // Get the number of user defined values
  unsigned int GetNumberOfUserDefinedValues();
  // Get a name/value by index
  const char *GetUserDefinedNameByIndex(unsigned int idx);
  const char *GetUserDefinedValueByIndex(unsigned int idx);

  // Description:
  // Copy the contents of p to this instance.
  virtual void DeepCopy(vtkMedicalImageProperties *p);

  // Description:
  // Add/Remove/Query the window/level presets that may have been associated
  // to a medical image. Window is also known as 'width', level is also known
  // as 'center'. The same window/level pair can not be added twice.
  // As a convenience, a comment (aka Explanation) can be associated to a preset.
  // For ex: DICOM Window Center (0028,1050) = 00045\000470
  //         DICOM Window Width  (0028,1051) = 0106\03412
  //         DICOM Window Center Width Explanation (0028,1055) = WINDOW1\WINDOW2
  virtual void AddWindowLevelPreset(double w, double l);
  virtual void RemoveWindowLevelPreset(double w, double l);
  virtual void RemoveAllWindowLevelPresets();
  virtual int GetNumberOfWindowLevelPresets();
  virtual int HasWindowLevelPreset(double w, double l);
  virtual int GetNthWindowLevelPreset(int idx, double *w, double *l);
  virtual double* GetNthWindowLevelPreset(int idx);
  virtual void SetNthWindowLevelPresetComment(int idx, const char *comment);
  virtual const char* GetNthWindowLevelPresetComment(int idx);

  // Description:
  // Mapping from a sliceidx within a volumeidx into a DICOM Instance UID
  // Some DICOM reader can populate this structure so that later on from a slice index
  // in a vtkImageData volume we can backtrack and find out which 2d slice it was coming from
  const char *GetInstanceUIDFromSliceID(int volumeidx, int sliceid);
  void SetInstanceUIDFromSliceID(int volumeidx, int sliceid, const char *uid);

  // Description:
  // Provides the inverse mapping. Returns -1 if a slice for this uid is
  // not found.
  int GetSliceIDFromInstanceUID(int &volumeidx, const char *uid);

//BTX
  typedef enum {
    AXIAL = 0,
    CORONAL,
    SAGITTAL
  } OrientationType;
//ETX
  int GetOrientationType(int volumeidx);
  void SetOrientationType(int volumeidx, int orientation);
  static const char *GetStringFromOrientationType(unsigned int type);

protected:
  vtkMedicalImageProperties();
  ~vtkMedicalImageProperties();

  char *StudyDate;
  char *AcquisitionDate;
  char *StudyTime;
  char *AcquisitionTime;
  char *ConvolutionKernel;
  char *EchoTime;
  char *EchoTrainLength;
  char *Exposure;
  char *ExposureTime;
  char *GantryTilt;
  char *ImageDate;
  char *ImageNumber;
  char *ImageTime;
  char *InstitutionName;
  char *KVP;
  char *ManufacturerModelName;
  char *Manufacturer;
  char *Modality;
  char *PatientAge;
  char *PatientBirthDate;
  char *PatientID;
  char *PatientName;
  char *PatientSex;
  char *RepetitionTime;
  char *SeriesDescription;
  char *SeriesNumber;
  char *SliceThickness;
  char *StationName;
  char *StudyDescription;
  char *StudyID;
  char *XRayTubeCurrent;

  // Description:
  // PIMPL Encapsulation for STL containers
  //BTX
  vtkMedicalImagePropertiesInternals *Internals;
  //ETX

private:
  vtkMedicalImageProperties(const vtkMedicalImageProperties&); // Not implemented.
  void operator=(const vtkMedicalImageProperties&); // Not implemented.
};

#endif
