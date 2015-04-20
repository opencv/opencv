/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkGDCMImageReader2 - read DICOM Image files (Pixel Data)
// .SECTION Description
// vtkGDCMImageReader2 is a source object that reads some DICOM files
// this reader is single threaded.
// .SECTION Implementation note: when FileLowerLeft is set to on the image is not flipped
// upside down as VTK would expect, use this option only if you know what you are doing.
// .SECTION Implementation note: when reading a series of 2D slices, user is
// expected to provide an ordered list of filenames. No sorting will be applied afterward.
// .SECTION Implementation note: Although 99% of the time the Zspacing as read
// from a tag in a 2D DICOM file should be correct, there has been reports that this
// value can be missing, or incorrect, in which case users are advised to override this
// value using the return value from gdcm::IPPSorter::GetZSpacing() and set it via
// vtkImageChangeInformation on the reader itself.
// .SECTION TODO
// This reader does not handle a series of 3D images, only a single 3D (multi frame) or a
// list of 2D files are supported for now.
// .SECTION TODO
// Did not implement SetFilePattern / SetFilePrefix API, move it to protected section for now.
// .SECTION BUG
// Overlay are assumed to have the same extent as image. Right now if overlay origin is not
// 0,0 the overlay will have an offset...
// Only the very first overlay is loaded at the VTK level, for now (even if there are more than one in the file)
// .SECTION DataOrigin
// When the reader is instanciated with FileLowerLeftOn the DataOrigin and Image Position (Patient) are
// identical. But when FileLowerLeft is Off, we have to reorder the Y-line of the image, and thus the DataOrigin
// is then translated to the other side of the image.
// .SECTION Spacing
// When reading a 3D volume, the spacing along the Z dimension might be negative (so as to respect up-side-down)
// as specified in the Image Orientation (Patient) tag. When Z-spacing is 0, this means the multi-frame object
// contains image which do not represent uniform volume.
// .SECTION Warning
// When using vtkGDCMPolyDataReader in conjonction with vtkGDCMImageReader2
// it is *required* that FileLowerLeft is set to ON as coordinate system
// would be inconsistent in between the two data structures.
// .SECTION Color Space mapping:
// * VTK_LUMINANCE         <-> MONOCHROME2
// * VTK_LUMINANCE_ALPHA   <-> Not supported
// * VTK_RGB               <-> RGB
// * VTK_RGBA              <-> ARGB (deprecated, DICOM 2008)
// * VTK_INVERSE_LUMINANCE <-> MONOCHROME1
// * VTK_LOOKUP_TABLE      <-> PALETTE COLOR
// * VTK_YBR               <-> YBR_FULL
//
// For detailed information on color space transformation and true lossless transformation see:
// http://gdcm.sourceforge.net/wiki/index.php/Color_Space_Transformations

// .SECTION See Also
// vtkMedicalImageReader2 vtkMedicalImageProperties vtkGDCMPolyDataReader vtkGDCMImageWriter
// vtkDICOMImageReader

#ifndef VTKGDCMIMAGEREADER2_H
#define VTKGDCMIMAGEREADER2_H

#include "vtkMedicalImageReader2.h"
#include "vtkImageData.h"

class vtkPolyData;

// vtkSystemIncludes.h defines:
// #define VTK_LUMINANCE       1
// #define VTK_LUMINANCE_ALPHA 2
// #define VTK_RGB             3
// #define VTK_RGBA            4
#ifndef VTK_INVERSE_LUMINANCE
#define VTK_INVERSE_LUMINANCE 5
#endif
#ifndef VTK_LOOKUP_TABLE
#define VTK_LOOKUP_TABLE 6
#endif
#ifndef VTK_YBR
#define VTK_YBR 7
#endif
#ifndef VTK_CMYK
#define VTK_CMYK 8
#endif

//BTX
namespace gdcm { class ImageReader; }
//ETX
class vtkMatrix4x4;
class VTK_EXPORT vtkGDCMImageReader2 : public vtkMedicalImageReader2
{
public:
  static vtkGDCMImageReader2 *New();
  vtkTypeRevisionMacro(vtkGDCMImageReader2,vtkMedicalImageReader2);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description: is the given file name a DICOM file containing an image ?
  virtual int CanReadFile(const char* fname);

  // Description:
  // Valid extensions
  virtual const char* GetFileExtensions()
    {
    // I would like to get rid of ACR/NEMA/IMA so only allow dcm extension for now
    return ".dcm .DCM";
    }

  // Description:
  // A descriptive name for this format
  virtual const char* GetDescriptiveName()
    {
    return "DICOM";
    }

  // Description:
  // Get the Image Position (Patient) as stored in the DICOM file
  // This is a read-only data member
  vtkGetObjectMacro(DirectionCosines, vtkMatrix4x4);

  virtual void SetMedicalImageProperties(vtkMedicalImageProperties *pd);

  // Description:
  // Specifically request to load the overlay into the gdcm-VTK layer (gdcm always loads them when found).
  // If no overlay is found in the image, then the vtkImageData for the overlay will be empty.
  vtkGetMacro(LoadOverlays,int);
  vtkSetMacro(LoadOverlays,int);
  vtkBooleanMacro(LoadOverlays,int);

  // Description:
  // Set/Get whether or not to load the Icon as vtkImageData (if found in the DICOM file)
  vtkGetMacro(LoadIconImage,int);
  vtkSetMacro(LoadIconImage,int);
  vtkBooleanMacro(LoadIconImage,int);

  // Description:
  // Set/Get whether or not the image was compressed using a lossy compression algorithm
  vtkGetMacro(LossyFlag,int);
  vtkSetMacro(LossyFlag,int);
  vtkBooleanMacro(LossyFlag,int);

  // Description:
  // Read only: number of overlays as found in this image (multiple overlays per slice is allowed)
  // Only valid when LoadOverlays is true
  vtkGetMacro(NumberOfOverlays,int);

  // Description:
  // Read only: number of icon image (there can only be zero or one icon per file)
  // Only valid when LoadIconImage is true
  vtkGetMacro(NumberOfIconImages,int);

  // Description:
  // Get Overlay/IconImage
  // Remember to ALWAYS use those methods in your code, as the internal number for the output port
  // is not garantee to remain the same, as features are added to the reader
  vtkAlgorithmOutput* GetOverlayPort(int index);
  vtkAlgorithmOutput* GetIconImagePort();
  vtkImageData* GetOverlay(int i);
  vtkImageData* GetIconImage();

  // Description:
  // Load image with its associated Lookup Table
  vtkGetMacro(ApplyLookupTable,int);
  vtkSetMacro(ApplyLookupTable,int);
  vtkBooleanMacro(ApplyLookupTable,int);

  // Description:
  // Load image as YBR
  vtkGetMacro(ApplyYBRToRGB,int)
  vtkSetMacro(ApplyYBRToRGB,int)
  vtkBooleanMacro(ApplyYBRToRGB,int);

  // Description:
  // Return VTK_LUMINANCE, VTK_INVERSE_LUMINANCE, VTK_RGB, VTK_RGBA, VTK_LOOKUP_TABLE, VTK_YBR or VTK_CMYK
  // or 0 when ImageFormat is not handled.
  // Warning: For color image, PlanarConfiguration need to be taken into account.
  vtkGetMacro(ImageFormat,int);

  // Description:
  // Return the Planar Configuration. This simply means that the internal DICOM image was stored
  // using a particular planar configuration (most of the time: 0)
  // For monochrome image, PlanarConfiguration is always 0
  vtkGetMacro(PlanarConfiguration,int);

  // Description:
  // Return the 'raw' information stored in the DICOM file:
  // In case of a series of multiple files, only the first file is considered. The Image Orientation (Patient)
  // is garantee to remain the same, and image Image Position (Patient) in other slice can be computed
  // using the ZSpacing (3rd dimension)
  // (0020,0032) DS [87.774866\-182.908510\168.629671]       #  32, 3 ImagePositionPatient
  // (0020,0037) DS [0.001479\0.999989\-0.004376\-0.002039\-0.004372\-0.999988] #  58, 6 ImageOrientationPatient
  vtkGetVector3Macro(ImagePositionPatient,double);
  vtkGetVector6Macro(ImageOrientationPatient,double);

  // Description:
  // Set/Get the first Curve Data:
  vtkGetObjectMacro(Curve,vtkPolyData);
  virtual void SetCurve(vtkPolyData *pd);

  // Description:
  // \DEPRECATED:
  // Modality LUT
  // Value returned by GetShift/GetScale might be innacurate since Shift/Scale could be
  // varying along the Series read. Therefore user are advices not to use those functions
  // anymore
  vtkGetMacro(Shift,double);
  vtkGetMacro(Scale,double);

protected:
  vtkGDCMImageReader2();
  ~vtkGDCMImageReader2();

  vtkSetVector6Macro(ImageOrientationPatient,double);

//BTX
  void FillMedicalImageInformation(const gdcm::ImageReader &reader);
//ETX
  int RequestInformationCompat();
  int RequestDataCompat();

  int ProcessRequest(vtkInformation* request,
                                 vtkInformationVector** inputVector,
                                 vtkInformationVector* outputVector);
  int RequestInformation(vtkInformation *request,
                         vtkInformationVector **inputVector,
                         vtkInformationVector *outputVector);
  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector);

protected:
  vtkMatrix4x4 *DirectionCosines;
  int LoadOverlays;
  int NumberOfOverlays;
  int LoadIconImage;
  int NumberOfIconImages;
  int IconImageDataExtent[6];
  double ImagePositionPatient[3];
  double ImageOrientationPatient[6];
  vtkPolyData *Curve;

  int ImageFormat;
  // the following 3, should remain optional
  int ApplyInverseVideo;
  int ApplyLookupTable;
  int ApplyYBRToRGB;
  // I think that planar configuration need to always be applied as far as VTK is concerned
  int ApplyPlanarConfiguration;
  int ApplyShiftScale;

  int LoadSingleFile(const char *filename, char *pointer, unsigned long &outlen);

  double Shift;
  double Scale;
  int IconDataScalarType;
  int IconNumberOfScalarComponents;
  int PlanarConfiguration;
  int LossyFlag;
  int ForceRescale;

protected:
  // TODO / FIXME
  void SetFilePrefix(const char *) {}
  vtkGetStringMacro(FilePrefix);
  void SetFilePattern(const char *) {}
  vtkGetStringMacro(FilePattern);

private:
  vtkGDCMImageReader2(const vtkGDCMImageReader2&);  // Not implemented.
  void operator=(const vtkGDCMImageReader2&);  // Not implemented.
};
#endif
