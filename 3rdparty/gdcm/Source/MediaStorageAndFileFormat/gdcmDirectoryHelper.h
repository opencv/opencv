/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "gdcmDirectory.h"
#include "gdcmDataSet.h"

namespace gdcm
{

/**
 * \brief DirectoryHelper
 * this class is designed to help mitigate some of the commonly performed
 * operations on directories.  namely:
 * 1) the ability to determine the number of series in a directory by what type
 * of series is present
 * 2) the ability to find all ct series in a directory
 * 3) the ability to find all mr series in a directory
 * 4) to load a set of DataSets from a series that's already been sorted by the
 * IPP sorter
 * 5) For rtstruct stuff, you need to know the sopinstanceuid of each z plane,
 * so there's a retrieval function for that
 * 6) then a few other functions for rtstruct writeouts
 */
class GDCM_EXPORT DirectoryHelper
{
public:
  //returns all series UIDs in a given directory that match a particular SOP Instance UID
  static Directory::FilenamesType GetSeriesUIDsBySOPClassUID(const std::string& inDirectory,
    const std::string& inSOPClassUID);

  //specific implementations of the SOPClassUID grabber, so you don't have to
  //remember the SOP Class UIDs of CT or MR images.
  static Directory::FilenamesType GetCTImageSeriesUIDs(const std::string& inDirectory);
  static Directory::FilenamesType GetMRImageSeriesUIDs(const std::string& inDirectory);
  static Directory::FilenamesType GetRTStructSeriesUIDs(const std::string& inDirectory);

  //given a directory and a series UID, provide all filenames with that series UID.
  static Directory::FilenamesType GetFilenamesFromSeriesUIDs(const std::string& inDirectory,
    const std::string& inSeriesUID);

  //given a series UID, load all the images associated with that series UID
  //these images will be IPP sorted, so that they can be used for gathering all
  //the necessary information for generating an RTStruct
  //this function should be called by the writer once, if the writer's dataset
  //vector is empty.  Make sure to have a new writer for new rtstructs.
  static std::vector<DataSet> LoadImageFromFiles(const std::string& inDirectory,
    const std::string& inSeriesUID);

  //When writing RTStructs, each contour will have z position defined.
  //use that z position to determine the SOPInstanceUID for that plane.
  static std::string RetrieveSOPInstanceUIDFromZPosition(double inZPos,
    const std::vector<DataSet>& inDS);

  //When writing RTStructs, the frame of reference is done by planes to start with
  static std::string RetrieveSOPInstanceUIDFromIndex(int inIndex,
   const std::vector<DataSet>& inDS);

  //each plane needs to know the SOPClassUID, and that won't change from image to image
  //so, retrieve this once at the start of writing.
  static std::string GetSOPClassUID(const std::vector<DataSet>& inDS);

  //retrieve the frame of reference from the set of datasets
  static std::string GetFrameOfReference(const std::vector<DataSet>& inDS);

  //both the image and polydata readers use these functions to get std::strings
  static std::string GetStringValueFromTag(const Tag& t, const DataSet& ds);
};

}
