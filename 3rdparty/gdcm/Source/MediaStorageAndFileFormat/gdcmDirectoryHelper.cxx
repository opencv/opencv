
#include "gdcmDirectoryHelper.h"
#include "gdcmScanner.h"
#include "gdcmIPPSorter.h"
#include "gdcmAttribute.h"
#include "gdcmDataElement.h"
#include "gdcmReader.h"


namespace gdcm{
//given an SOPClassUID, get the number of series that are that SOP Class
//that is, the MR images, the CT Images, whatever.  Just have to be sure to give the proper
//SOP Class UID.
//returns an empty vector if nothing's there or if something goes wrong.
Directory::FilenamesType DirectoryHelper::GetSeriesUIDsBySOPClassUID(const std::string& inDirectory,
                                                                       const std::string& inSOPClassUID)
{
  Scanner theScanner;
  Directory theDir;
  theScanner.AddTag(Tag(0x0008, 0x0016));//SOP Class UID
  theScanner.AddTag(Tag(0x0020, 0x000e));//Series UID
  Directory::FilenamesType theReturn;

  try {
    theDir.Load(inDirectory);
    theScanner.Scan(theDir.GetFilenames());

    //now find all series UIDs
    Directory::FilenamesType theSeriesValues = theScanner.GetOrderedValues(Tag(0x0020,0x000e));

    //now count the number of series that are of that given SOPClassUID
    size_t theNumSeries = theSeriesValues.size();
    for (size_t i = 0; i < theNumSeries; i++){
      std::string theFirstFilename =
      theScanner.GetFilenameFromTagToValue(Tag(0x0020,0x000e), theSeriesValues[i].c_str());
      std::string theSOPClassUID = theScanner.GetValue(theFirstFilename.c_str(), Tag(0x0008,0x0016));
      //dicom strings sometimes have trailing spaces; make sure to avoid those
      size_t endpos = theSOPClassUID.find_last_not_of(" "); // Find the first character position from reverse af
      if( std::string::npos != endpos )
        theSOPClassUID = theSOPClassUID.substr( 0, endpos+1 );
      if (theSOPClassUID == inSOPClassUID.c_str()){
        theReturn.push_back(theSeriesValues[i]);
      }
    }
    return theReturn;
  } catch (...){
    Directory::FilenamesType theBlank;
    return theBlank;//something broke during scanning
  }
}


//given the name of a directory, return the list of CT Image UIDs
Directory::FilenamesType DirectoryHelper::GetCTImageSeriesUIDs(const std::string& inDirectory)
{
  return GetSeriesUIDsBySOPClassUID(inDirectory, "1.2.840.10008.5.1.4.1.1.2");
}

//given the name of a directory, return the list of CT Image UIDs
Directory::FilenamesType DirectoryHelper::GetMRImageSeriesUIDs(const std::string& inDirectory)
{
  return GetSeriesUIDsBySOPClassUID(inDirectory, "1.2.840.10008.5.1.4.1.1.4");
}

//given the name of a directory, return the list of CT Image UIDs
Directory::FilenamesType DirectoryHelper::GetRTStructSeriesUIDs(const std::string& inDirectory)
{
  return GetSeriesUIDsBySOPClassUID(inDirectory, "1.2.840.10008.5.1.4.1.1.481.3");
}

//given a directory and a series UID, provide all filenames with that series UID.
Directory::FilenamesType DirectoryHelper::GetFilenamesFromSeriesUIDs(const std::string& inDirectory,
                                                                     const std::string& inSeriesUID)
{
  Scanner theScanner;
  Directory theDir;
  theScanner.AddTag(Tag(0x0020, 0x000e));//Series UID
  Directory::FilenamesType theReturn;

  try {
    theDir.Load(inDirectory);
    theScanner.Scan(theDir.GetFilenames());
    //now find all series UIDs
    Directory::FilenamesType theSeriesValues = theScanner.GetOrderedValues(Tag(0x0020,0x000e));
    //now count the number of series that are of that given SOPClassUID
    size_t theNumSeries = theSeriesValues.size();
    for (size_t i = 0; i < theNumSeries; i++)
      {
      std::string theSeriesUID = theSeriesValues[i];
      //dicom strings sometimes have trailing spaces; make sure to avoid those
      size_t endpos = theSeriesUID.find_last_not_of(" "); // Find the first character position from reverse af
      if( std::string::npos != endpos )
        theSeriesUID = theSeriesUID.substr( 0, endpos+1 );
      if (inSeriesUID == theSeriesUID)
      {
	    Directory::FilenamesType theFilenames =
		    theScanner.GetAllFilenamesFromTagToValue(Tag(0x0020, 0x000e), theSeriesValues[i].c_str());
		  Directory::FilenamesType::const_iterator citor;
		  for (citor = theFilenames.begin(); citor < theFilenames.end(); citor++)
        {
		    theReturn.push_back(*citor);
		    }
    //      theReturn.push_back(theScanner.GetFilenameFromTagToValue(Tag(0x0020,0x000e),
    //        theSeriesValues[i].c_str()));
      }
    }
    return theReturn;
  } catch (...){
    Directory::FilenamesType theBlank;
    return theBlank;//something broke during scanning
  }

}
//the code in GetSeriesUIDsBySOPClassUID will enumerate the CT images in a directory
//This code will retrieve an image by its Series UID.
//this code doesn't return pointers or smart pointers because it's intended to
//be easily wrapped by calling languages that don't know pointers (ie, Java)
//this function is a proof of concept
//for it to really work, it needs to also
std::vector<DataSet> DirectoryHelper::LoadImageFromFiles(const std::string& inDirectory,
                                                           const std::string& inSeriesUID)
{
  Scanner theScanner;
  Directory theDir;
  theScanner.AddTag(Tag(0x0020, 0x000e));//Series UID
  std::vector<DataSet> theReturn;
  std::vector<DataSet> blank;//returned in case of an error

  try {
    theDir.Load(inDirectory);
    theScanner.Scan(theDir.GetFilenames());

    //now find all series UIDs
    Directory::FilenamesType theSeriesValues = theScanner.GetOrderedValues(Tag(0x0020,0x000e));

    //now count the number of series that are of that given SOPClassUID
    size_t theNumSeries = theSeriesValues.size();
    for (size_t i = 0; i < theNumSeries; i++){
      if (inSeriesUID == theSeriesValues[i]){
        //find all files that have that series UID, and then load them via
        //the vtkImageReader
        Directory::FilenamesType theFiles =
        theScanner.GetAllFilenamesFromTagToValue(Tag(0x0020, 0x000e), theSeriesValues[i].c_str());
        IPPSorter sorter;
        sorter.SetComputeZSpacing(true);
        sorter.SetZSpacingTolerance(0.000001);
        if (!sorter.Sort(theFiles)){
          gdcmWarningMacro("Unable to sort Image Files.");
          return blank;
        }
        Directory::FilenamesType theSortedFiles = sorter.GetFilenames();
        for (unsigned long j = 0; j < theSortedFiles.size(); ++j){
          Reader theReader;
          theReader.SetFileName(theSortedFiles[j].c_str());
          theReader.Read();
          theReturn.push_back(theReader.GetFile().GetDataSet());
        }
        return theReturn;
      }
    }
    return blank;
  } catch (...){
    gdcmWarningMacro("Something went wrong reading CT images.");
    return blank;
  }
}


//When writing RTStructs, each contour will have z position defined.
//use that z position to determine the SOPInstanceUID for that plane.
std::string DirectoryHelper::RetrieveSOPInstanceUIDFromZPosition(double inZPos,
                                                const std::vector<DataSet>& inDS)
{
  std::vector<DataSet>::const_iterator itor;
  Tag thePosition(0x0020, 0x0032);
  Tag theSOPInstanceUID(0x0008, 0x0018);
  std::string blank;//return only if there's a problem
  for (itor = inDS.begin(); itor != inDS.end(); itor++)
    {
    if (itor->FindDataElement(thePosition))
      {
      DataElement de = itor->GetDataElement(thePosition);
      Attribute<0x0020,0x0032> at;
      at.SetFromDataElement( de );
      if (fabs(at.GetValue(2) - inZPos)<0.01)
        {
        DataElement de2 = itor->GetDataElement(theSOPInstanceUID);
        const ByteValue* theVal = de2.GetByteValue();
        size_t theValLen = theVal->GetLength();
        std::string theReturn(de2.GetByteValue()->GetPointer(), theValLen);
        return theReturn;
        }
      }
    }
  return blank;
}

//When writing RTStructs, the frame of reference is done by planes to start with
std::string DirectoryHelper::RetrieveSOPInstanceUIDFromIndex(int inIndex,
                                                               const std::vector<DataSet>& inDS)
{

  Tag theSOPInstanceUID(0x0008, 0x0018);
  std::string blank;//return only if there's a problem
  if (inDS[inIndex].FindDataElement(theSOPInstanceUID)){
    DataElement de = inDS[inIndex].GetDataElement(theSOPInstanceUID);
    const ByteValue* theVal = de.GetByteValue();
    size_t theValLen = theVal->GetLength();
    std::string theReturn(de.GetByteValue()->GetPointer(), theValLen);
    return theReturn;
  }
  return blank;
}

//each plane needs to know the SOPClassUID, and that won't change from image to image
//so, retrieve this once at the start of writing.
std::string DirectoryHelper::GetSOPClassUID(const std::vector<DataSet>& inDS)
{
  Tag theSOPClassUID(0x0008, 0x0016);
  std::string blank;//return only if there's a problem
  if (inDS[0].FindDataElement(theSOPClassUID)){
    DataElement de = inDS[0].GetDataElement(theSOPClassUID);
    const ByteValue* theVal = de.GetByteValue();
    size_t theValLen = theVal->GetLength();
    std::string theReturn(de.GetByteValue()->GetPointer(), theValLen);
    return theReturn;
  }
  return blank;
}

//retrieve the frame of reference from the set of datasets
std::string DirectoryHelper::GetFrameOfReference(const std::vector<DataSet>& inDS){

  Tag theSOPClassUID(0x0020, 0x0052);
  std::string blank;//return only if there's a problem
  if (inDS[0].FindDataElement(theSOPClassUID)){
    DataElement de = inDS[0].GetDataElement(theSOPClassUID);
    const ByteValue* theVal = de.GetByteValue();
    size_t theValLen = theVal->GetLength();
    std::string theReturn(de.GetByteValue()->GetPointer(), theValLen);
    return theReturn;
  }
  return blank;
}


//----------------------------------------------------------------------------
//used by the vtkGDCMImageReader and vtkGDCMPolyDataReader. Could be used elsewhere, I suppose.
std::string DirectoryHelper::GetStringValueFromTag(const Tag& t, const DataSet& ds)
{
  std::string buffer;

  if( ds.FindDataElement( t ) )
    {
    const DataElement& de = ds.GetDataElement( t );
    const ByteValue *bv = de.GetByteValue();
    if( bv ) // Can be Type 2
      {
      buffer = std::string( bv->GetPointer(), bv->GetLength() );
      // Will be padded with at least one \0
      }
    }

  // Since return is a const char* the very first \0 will be considered
  return buffer.c_str(); // Yes, I mean .c_str()
}
} // end namespace gdcm
