/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMIMAGEHELPER_H
#define GDCMIMAGEHELPER_H

#include "gdcmTypes.h"
#include "gdcmTag.h"
#include <vector>
#include "gdcmPixelFormat.h"
#include "gdcmPhotometricInterpretation.h"
#include "gdcmSmartPointer.h"
#include "gdcmLookupTable.h"

namespace gdcm
{

class MediaStorage;
class DataSet;
class File;
class Image;
class ByteValue;
/**
 * \brief ImageHelper (internal class, not intended for user level)
 *
 * \details
 * Helper for writing World images in DICOM. DICOM has a 'template' approach to image where
 * MR Image Storage are distinct object from Enhanced MR Image Storage. For example the
 * Pixel Spacing in one object is not at the same position (ie Tag) as in the other
 * this class is the central (read: fragile) place where all the dispatching is done from
 * a unified view of a world image (typically VTK or ITK point of view) down to the low
 * level DICOM point of view.
 *
 * \warning: do not expect the API of this class to be maintained at any point, since as
 * Modalities are added the API might have to be augmented or behavior changed to cope
 * with new modalities.
 */
class GDCM_EXPORT ImageHelper
{
public:
  /// GDCM 1.x compatibility issue:
  /// when using ReWrite an MR Image Storage would be rewritten with a Rescale Slope/Intercept
  /// while the standard would prohibit this (Philips Medical System is still doing that)
  /// Unless explicitely set elsewhere by the standard, it will use value from 0028,1052 / 0028,1053
  /// for the Rescale Slope & Rescale Intercept values
  static void SetForceRescaleInterceptSlope(bool);
  static bool GetForceRescaleInterceptSlope();

  /// GDCM 1.x compatibility issue:
  /// When using ReWrite an MR Image Storage would be rewritten as Secondary Capture Object while
  /// still having a Pixel Spacing tag (0028,0030). If you have deal with those files, use this
  /// very special flag to handle them
  /// Unless explicitely set elsewhere by the standard, it will use value from 0028,0030 / 0018,0088
  /// for the Pixel Spacing of the Image
  static void SetForcePixelSpacing(bool);
  static bool GetForcePixelSpacing();

  /// This function checks tags (0x0028, 0x0010) and (0x0028, 0x0011) for the
  /// rows and columns of the image in pixels (as opposed to actual distances).
  /// The output is {col , row}
  static std::vector<unsigned int> GetDimensionsValue(const File& f);
  static void SetDimensionsValue(File& f, const Image & img);

  /// This function returns pixel information about an image from its dataset
  /// That includes samples per pixel and bit depth (in that order)
  static PixelFormat GetPixelFormatValue(const File& f);

  /// Set/Get shift/scale from/to a file
  /// \warning this function reads/sets the Slope/Intercept in appropriate
  /// class storage, but also Grid Scaling in RT Dose Storage
  /// Can't take a dataset because the mediastorage of the file must be known
  static std::vector<double> GetRescaleInterceptSlopeValue(File const & f);
  static void SetRescaleInterceptSlopeValue(File & f, const Image & img);

  /// Set/Get Origin (IPP) from/to a file
  static std::vector<double> GetOriginValue(File const & f);
  static void SetOriginValue(DataSet & ds, const Image & img);

  /// Get Direction Cosines (IOP) from/to a file
  /// Requires a file because mediastorage must be known
  static std::vector<double> GetDirectionCosinesValue(File const & f);
  /// Set Direction Cosines (IOP) from/to a file
  /// When IOD does not defines what is IOP (eg. typically Secondary Capture Image Storage)
  /// this call will simply remove the IOP attribute.
  /// Else in case of MR/CT image storage, this call will properly lookup the correct attribute
  /// to store the IOP.
  // FIXME: There is a major issue for image with multiple IOP (eg. Enhanced * Image Storage).
  static void SetDirectionCosinesValue(DataSet & ds, const std::vector<double> & dircos);

  /// Set/Get Spacing from/to a File
  static std::vector<double> GetSpacingValue(File const & f);
  static void SetSpacingValue(DataSet & ds, const std::vector<double> & spacing);

  /// DO NOT USE
  static bool ComputeSpacingFromImagePositionPatient(const std::vector<double> &imageposition, std::vector<double> & spacing);

  static bool GetDirectionCosinesFromDataSet(DataSet const & ds, std::vector<double> & dircos);

  //functions to get more information from a file
  //useful for the stream image reader, which fills in necessary image information
  //distinctly from the reader-style data input
  static PhotometricInterpretation GetPhotometricInterpretationValue(File const& f);
  //returns the configuration of colors in a plane, either RGB RGB RGB or RRR GGG BBB
  static unsigned int GetPlanarConfigurationValue(const File& f);

  //returns the lookup table of an image file
  static SmartPointer<LookupTable> GetLUT(File const& f);

  ///Moved from PixampReader to here.  Generally used for photometric interpretation.
  static const ByteValue* GetPointerFromElement(Tag const &tag, File const& f);

  /// Moved from MediaStorage here, since we need extra info stored in PixelFormat & PhotometricInterpretation
  static MediaStorage ComputeMediaStorageFromModality(const char *modality,
    unsigned int dimension = 2, PixelFormat const & pf = PixelFormat(),
    PhotometricInterpretation const & pi = PhotometricInterpretation(), double rescaleintercept = 0, double rescaleslope = 1 );

protected:
  static Tag GetSpacingTagFromMediaStorage(MediaStorage const &ms);
  static Tag GetZSpacingTagFromMediaStorage(MediaStorage const &ms);

private:
  static bool ForceRescaleInterceptSlope;
  static bool ForcePixelSpacing;
};

} // end namespace gdcm

#endif // GDCMIMAGEHELPER_H
