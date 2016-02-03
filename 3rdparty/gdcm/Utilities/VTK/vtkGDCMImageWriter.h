/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkGDCMImageWriter - write DICOM files
// .SECTION Description
// vtkGDCMImageWriter is a sink object that write DICOM files
// this writer is single threaded (see vtkGDCMThreadedImageReader2 for multi-thread)
//
// .SECTION Warning: vtkLookupTable from the vtkImageData object taken into account
// only if ImageFormat is set to VTK_LOOKUP_TABLE
//
// .SECTION NOTE We are not using the usual API SetFilePrefix / SetFilePattern,
// but instead a list of filenames: see SetFileNames and class gdcm::FilenameGenerator
//
// .SECTION Warning
// You need to specify the correct ImageFormat (taken from the reader)
// You need to explicitly specify the DirectionCosines (taken from the reader)
// Since VTK 5.4 vtkMedicalImageProperties has its own DirectionCosine (no 's')
// user need to make sure the vtkMatrix4x4 is compatible with the 6-vector DirectionCosine.
//
// .SECTION NOTE Shift/Scale are global to all DICOM frames (=files) written
// as 2D slice, therefore the shift/scale operation might not be optimized for
// all slices. This is not recommended for image with a large dynamic range.
//
// .SECTION See Also
// vtkImageWriter vtkMedicalImageProperties vtkGDCMImageReader

#ifndef VTKGDCMIMAGEWRITER_H
#define VTKGDCMIMAGEWRITER_H

#include "vtkImageWriter.h"

class vtkLookupTable;
class vtkMedicalImageProperties;
class vtkMatrix4x4;
class vtkStringArray;
class VTK_EXPORT vtkGDCMImageWriter : public vtkImageWriter
{
public:
  static vtkGDCMImageWriter *New();
  vtkTypeRevisionMacro(vtkGDCMImageWriter,vtkImageWriter);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Pass in the vtkmedicalimageproperties object for medical information
  // to be mapped to DICOM attributes.
  vtkGetObjectMacro(MedicalImageProperties, vtkMedicalImageProperties);
  virtual void SetMedicalImageProperties(vtkMedicalImageProperties*);

  // Description:
  // Pass in the list of filename to be used to write out the DICOM file(s)
  virtual void SetFileNames(vtkStringArray*);
  vtkGetObjectMacro(FileNames, vtkStringArray);

  // Description:
  // Set/Get whether or not the image was compressed using a lossy compression algorithm
  vtkGetMacro(LossyFlag,int);
  vtkSetMacro(LossyFlag,int);
  vtkBooleanMacro(LossyFlag,int);

  // I need that...
  virtual void Write();

  // Description:
  // Get the entension for this file format.
  virtual const char* GetFileExtensions() {
    return ".dcm .DCM"; }

  // Description:
  // Get the name of this file format.
  virtual const char* GetDescriptiveName() {
    return "DICOM"; }

  // Description:
  // You need to manually specify the direction the image is in to write a valid DICOM file
  // since vtkImageData do not contains one (eg. MR Image Storage, CT Image Storage...)
  virtual void SetDirectionCosines(vtkMatrix4x4 *matrix);
  vtkGetObjectMacro(DirectionCosines, vtkMatrix4x4);
  virtual void SetDirectionCosinesFromImageOrientationPatient(const double dircos[6]);

  // Description:
  // Modality LUT
  vtkSetMacro(Shift, double);
  vtkGetMacro(Shift, double);
  vtkSetMacro(Scale, double);
  vtkGetMacro(Scale, double);

  // Description:
  // See vtkGDCMImageReader for list of ImageFormat
  vtkGetMacro(ImageFormat,int);
  vtkSetMacro(ImageFormat,int);

  // Description:
  // Set/Get whether the data comes from the file starting in the lower left
  // corner or upper left corner.
  vtkBooleanMacro(FileLowerLeft, int);
  vtkGetMacro(FileLowerLeft, int);
  vtkSetMacro(FileLowerLeft, int);

  // Description:
  // For color image (more than a single comp) you can specify the planar configuration you prefer
  vtkSetMacro(PlanarConfiguration,int);
  vtkGetMacro(PlanarConfiguration,int);

  // Description:
  // Set/Get specific StudyUID / SeriesUID
  vtkSetStringMacro(StudyUID);
  vtkGetStringMacro(StudyUID);
  vtkSetStringMacro(SeriesUID);
  vtkGetStringMacro(SeriesUID);

//BTX
  enum CompressionTypes {
    NO_COMPRESSION = 0,   // raw (default)
    JPEG_COMPRESSION,     // JPEG
    JPEG2000_COMPRESSION, // J2K
    JPEGLS_COMPRESSION,   // JPEG-LS
    RLE_COMPRESSION       // RLE
  };
//ETX
  // Set/Get the compression type
  vtkSetMacro(CompressionType, int);
  vtkGetMacro(CompressionType, int);

  //void SetCompressionTypeFromString(const char *);
  //const char *GetCompressionTypeAsString();

protected:
  vtkGDCMImageWriter();
  ~vtkGDCMImageWriter();

#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
  int FillInputPortInformation(int port, vtkInformation *info);
  int RequestInformation(
    vtkInformation *request,
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector);
  int RequestUpdateExtent(
    vtkInformation *request,
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector);
  int RequestData(
    vtkInformation *request,
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector);
#else
  void WriteSlice(vtkImageData *data);
#endif /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/
  int WriteGDCMData(vtkImageData *data, int timeStep);

protected:
  virtual /*const*/ char *GetFileName();

private:
  vtkGDCMImageWriter(const vtkGDCMImageWriter&);  // Not implemented.
  void operator=(const vtkGDCMImageWriter&);  // Not implemented.

  // VTK structs:
  //vtkLookupTable *LookupTable;
  vtkMedicalImageProperties *MedicalImageProperties;
  char *StudyUID;
  char *SeriesUID;

  int DataUpdateExtent[6];
  int ImageFormat;

  vtkStringArray *FileNames;
  vtkMatrix4x4 *DirectionCosines;

  double Shift;
  double Scale;
  int FileLowerLeft;
  int PlanarConfiguration;
  int LossyFlag;
  int CompressionType;
};

#endif
