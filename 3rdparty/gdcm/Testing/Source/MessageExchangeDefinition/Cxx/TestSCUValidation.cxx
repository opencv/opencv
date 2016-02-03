/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmULConnectionManager.h"
#include "gdcmPresentationContextGenerator.h"
#include "gdcmReader.h"
#include "gdcmAttribute.h"
#include "gdcmDataSet.h"
#include "gdcmUIDGenerator.h"
#include "gdcmStringFilter.h"
#include "gdcmWriter.h"

#include "gdcmDirectory.h"
#include "gdcmImageReader.h"
#include "gdcmQueryFactory.h"
#include "gdcmGlobal.h"

const char *AETitle = "ANY";
const char *PeerAETitle = "ANY";
const char *ComputerName = "87.106.65.167"; // www.dicomserver.co.uk
int port = 11112;

gdcm::network::ULConnectionManager *GetConnectionManager(gdcm::BaseRootQuery* theQuery)
{
  gdcm::PresentationContextGenerator generator;
  if( !generator.GenerateFromUID( theQuery->GetAbstractSyntaxUID() ) )
    {
    gdcmErrorMacro( "Failed to generate pres context." );
    return NULL;
    }

  gdcm::network::ULConnectionManager *theManager =
    new gdcm::network::ULConnectionManager();
  if (!theManager->EstablishConnection(AETitle, PeerAETitle, ComputerName, 0,
    (uint16_t)port, 1000, generator.GetPresentationContexts() ))
  {
    throw gdcm::Exception("Failed to establish connection.");
  }
  return theManager;
}

std::vector<gdcm::DataSet> GetPatientInfo(bool validateQuery, bool inStrictQuery)
{
  std::vector<gdcm::DataSet> theDataSets;
  gdcm::BaseRootQuery* theQuery =
    gdcm::QueryFactory::ProduceQuery(gdcm::ePatientRootType, gdcm::eFind,
      gdcm::ePatient);
  theQuery->SetSearchParameter(gdcm::Tag(0x8, 0x52), "PATIENT"); //Query/Retrieval Level
  theQuery->SetSearchParameter(gdcm::Tag(0x10,0x20), ""); //Patient ID
  theQuery->SetSearchParameter(gdcm::Tag(0x10,0x10), "*"); //Patient Name
  if(validateQuery && !theQuery->ValidateQuery(inStrictQuery))
  {
    return theDataSets;
  }

  gdcm::network::ULConnectionManager *theManager = GetConnectionManager( theQuery );
  theDataSets  = theManager->SendFind( theQuery );
  return theDataSets;
}

std::vector<gdcm::DataSet> GetStudyInfo(const char *patientID, bool validateQuery, bool inStrictQuery)
{
  std::vector<gdcm::DataSet> theDataSets;
  gdcm::BaseRootQuery* theQuery =
    gdcm::QueryFactory::ProduceQuery(gdcm::eStudyRootType, gdcm::eFind, gdcm::eStudy);
  theQuery->SetSearchParameter(gdcm::Tag(0x8, 0x52), "STUDY"); //Query/Retrieval Level

  theQuery->SetSearchParameter(gdcm::Tag(0x10,0x20), patientID); //Patient ID
  theQuery->SetSearchParameter(gdcm::Tag(0x20, 0x10), ""); //Study ID
  theQuery->SetSearchParameter(gdcm::Tag(0x20, 0xD), ""); //Study Instance UID
  theQuery->SetSearchParameter(gdcm::Tag(0x20, 0xE), ""); //Series Instance UID
  if(validateQuery && !theQuery->ValidateQuery(inStrictQuery))
  {
    return theDataSets;
  }

  gdcm::network::ULConnectionManager *theManager = GetConnectionManager( theQuery );
  theDataSets  = theManager->SendFind( theQuery );
  return theDataSets;
}

std::vector<gdcm::DataSet> GetSeriesInfo(const char *patientID, const char *studyInstanceUID, bool validateQuery, bool inStrictQuery)
{
  std::vector<gdcm::DataSet> theDataSets;
  gdcm::BaseRootQuery* theQuery =
    gdcm::QueryFactory::ProduceQuery(gdcm::eStudyRootType, gdcm::eFind, gdcm::eSeries);
  theQuery->SetSearchParameter(gdcm::Tag(0x8, 0x52), "SERIES"); //Query/Retrieval Level

  theQuery->SetSearchParameter(gdcm::Tag(0x10,0x20), patientID); //Patient ID
  theQuery->SetSearchParameter(gdcm::Tag(0x20, 0xD), studyInstanceUID); //Study Instance UID
  theQuery->SetSearchParameter(gdcm::Tag(0x20, 0xE), ""); //Series Instance UID
  if(validateQuery && !theQuery->ValidateQuery(inStrictQuery))
  {
    return theDataSets;
  }
  gdcm::network::ULConnectionManager *theManager = GetConnectionManager( theQuery );
  theDataSets  = theManager->SendFind( theQuery );
  return theDataSets;
}

std::vector<gdcm::DataSet> GetImageInfo(const char *patientID,
               const char *studyInstanceUID, const char *seriesInstanceUID, bool validateQuery, bool inStrictQuery)
{
  std::vector<gdcm::DataSet> theDataSets;
  gdcm::BaseRootQuery* theQuery =
    gdcm::QueryFactory::ProduceQuery(gdcm::eStudyRootType, gdcm::eFind, gdcm::eImage);
  theQuery->SetSearchParameter(gdcm::Tag(0x8, 0x52), "SERIES"); //Query/Retrieval Level

  theQuery->SetSearchParameter(gdcm::Tag(0x10,0x20), patientID); //Patient ID
  theQuery->SetSearchParameter(gdcm::Tag(0x20, 0xD), studyInstanceUID); //Study Instance UID
  theQuery->SetSearchParameter(gdcm::Tag(0x20, 0xE), seriesInstanceUID); //Series Instance UID
  theQuery->SetSearchParameter(gdcm::Tag(0x8, 0x18), ""); //SOP Instance UID
  if(validateQuery && !theQuery->ValidateQuery(inStrictQuery))
  {
    return theDataSets;
  }
  gdcm::network::ULConnectionManager *theManager = GetConnectionManager( theQuery );
  theDataSets = theManager->SendFind( theQuery );
  return theDataSets;
}

void PrintDataSets(std::vector<gdcm::DataSet> theDataSets)
{
  std::vector<gdcm::DataSet>::iterator itor;
  for (itor = theDataSets.begin(); itor < theDataSets.end(); itor++)
    itor->Print(std::cout);
}


int TestSCUValidation(int , char *[])
{
  //set this to true to use a strict interpretation of the DICOM standard for query validation
  bool theUseStrictQueries = false;

  //Case 1:
  //Here I want to retrieve Study Information for the known Patient.
  //Here i pass the PatientID as a input and i need to reterive the StudyId,
  //StudyDate and Series Instance UID.
  std::vector<gdcm::DataSet> theDataSets = GetStudyInfo("Z354998", true, theUseStrictQueries);
  PrintDataSets(theDataSets);
  //In the above i validated the constructed Query. This will not allow to add the
  //Series Instance UID as a search parameter for the query. On the result of
  //this i can't get the SeriesInstanceUID from the study level.

  //Case 2:
  //Here I execute the above same CFind Query with out validating.
  //This will send the Query which is having the SeriesInstanceUID tag
  //in search parameter to the CFind. This will executed successfully and
  //returns the SeriesInsanceUID related to the given StudyUID.
  theDataSets = GetStudyInfo("Z354998", false, theUseStrictQueries);
  PrintDataSets(theDataSets);

  //case 3:
  //If i validated the Query Like case 1, i cant get the Series Instance UID from
  //Study level. With out SeriesInstanceUID i can't retrieve the other series Information.
  //In Series Level also i cant add the StudyInstanceUID or other study information
  //as a search parameter. It allows only SeriesInstanceUID, Modality and SeriesNumber
  //as a search parameter.
  theDataSets = GetSeriesInfo("Z354998",
    "1.2.826.0.1.3680043.4.1.19990124221049.2", true, theUseStrictQueries);
  PrintDataSets(theDataSets);

  //case 4:
  //If i execute the above same CFind Query for Get Series with out validating
  //the query, it will return the requested SeriesInstanceUID for Known Study level.
  theDataSets = GetSeriesInfo("Z354998",
    "1.2.826.0.1.3680043.4.1.19990124221049.2", false, theUseStrictQueries);
  PrintDataSets(theDataSets);
  //In StudyLevel I cant get the Series information(Ref:Case 2). In Series Level
  //also i cant get the Series information for the known Study Level(Ref:Case 3).
  //How should i get the Series level information for the known patient and study
  //level information???

  //case 5:
  //For retrieve the Image Level Information (SOP Instance UID) for the known
  //Patient,Study and Series level information, the Image level query is not
  //allowing. It allows only SOP Instance UID and SOP Instance Number as a
  //search Query.
  theDataSets = GetImageInfo("Z354998",
    "1.2.826.0.1.3680043.4.1.19990124221049.2",
    "1.2.826.0.1.3680043.4.1.19990124221049.3", true, theUseStrictQueries);
  PrintDataSets(theDataSets);

  //case 6:
  //Image Level retrieval also give the required information with out
  //validate the generated Query.
  theDataSets = GetImageInfo("Z354998",
    "1.2.826.0.1.3680043.4.1.19990124221049.2",
    "1.2.826.0.1.3680043.4.1.19990124221049.3", false, theUseStrictQueries);
  PrintDataSets(theDataSets);

  //I want the Following things to do in CFind Query.
  // 1. Reterive the Study Level Information for the known Patient.
  // 2. Reterive the Series Level Information for the known Patient and Study.
  // 3. Reterive the Image Level Information for the known Patient, Study and Series.
  return 0;
}
