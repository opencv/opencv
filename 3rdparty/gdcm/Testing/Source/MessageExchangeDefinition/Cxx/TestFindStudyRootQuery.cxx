/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFindStudyRootQuery.h"

#include "gdcmCompositeNetworkFunctions.h"
#include "gdcmTrace.h"

/*
 * STUDY:
 * $ findscu --call GDCM_STORE --aetitle GDCMDASH -P server 11112 -k 8,52="PATIENT" -k 10,20="1*"
 * $ findscu --call GDCM_STORE --aetitle GDCMDASH -S server 11112 -k 8,52="STUDY"   -k 10,20="FOO"
 *
 * SERIES:
 * $ findscu --call GDCM_STORE --aetitle GDCMDASH -S lirispat 11112 -k 8,52="SERIES" -k 20,d="1.2.3" -k 8,60 
 */

int TestFindStudyRootQuery(int , char *[])
{
  //gdcm::Trace::DebugOn();
  gdcm::Trace::WarningOff();

  // STUDY:
  gdcm::ERootType theRoot = gdcm::eStudyRootType;
  gdcm::EQueryLevel theLevel = gdcm::eStudy;

    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( theQuery->ValidateQuery( true ) )
      {
      // No key found is an error
      return 1;
      }
    }
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x10,0x10), "PATIENT" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( theQuery->ValidateQuery( true ) )
      {
      // Patient Id is a Required Key in Study
      return 1;
      }
    }
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0x10), "studyid" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( theQuery->ValidateQuery( true ) )
      {
      // Study Id is a required tag
      return 1;
      }
    }
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x8,0x90), "physician" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( theQuery->ValidateQuery( true ) )
      {
      // ref physician's name is optional
      return 1;
      }
    }
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xd), "studyuid" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( !theQuery->ValidateQuery( true ) )
      {
      // Study UID is the unique tag
      return 1;
      }
    }

  // SERIES:

  theLevel = gdcm::eSeries;

    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( theQuery->ValidateQuery( true ) )
      {
      // No key found is an error
      return 1;
      }
    }
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xd), "1.2.3" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( theQuery->ValidateQuery( true ) )
      {
      // No key at level Series
      return 1;
      }
    }
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xd), "1.2.3" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x8,0x60), "" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( theQuery->ValidateQuery( true ) )
      {
      // missing unique at series level
      return 1;
      }
    }
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xd), "1.2.3" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xe), "4.5.6" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( !theQuery->ValidateQuery( true ) )
      {
      // all unique keys present
      return 1;
      }
    }
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xd), "1.2.3" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xe), "4.5.6" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x8,0x60), "" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( !theQuery->ValidateQuery( true ) )
      {
      // all unique keys present and required is correct level
      return 1;
      }
    }
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xd), "1.2.3" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xe), "4.5.6" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x8,0x20), "" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( theQuery->ValidateQuery( true ) )
      {
      // all unique keys present and required is incorrect level
      return 1;
      }
    }

  // IMAGES:

  theLevel = gdcm::eImage;
    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xd), "1.2.3" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xe), "4.5.6" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x8,0x18), "7.8.9" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( !theQuery->ValidateQuery( true ) )
      {
      // all unique keys present
      return 1;
      }
    }

    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xd), "1.2.3" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xe), "4.5.6" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x8,0x18), "7.8.9" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0x13), "" ) ) ;
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( !theQuery->ValidateQuery( true ) )
      {
      // all unique keys present + required correct level
      return 1;
      }
    }

    {
    std::vector< std::pair<gdcm::Tag, std::string> > keys;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xd), "1.2.3" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0xe), "4.5.6" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x8,0x18), "7.8.9" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0x13), "" ) ) ;
    keys.push_back( std::make_pair( gdcm::Tag(0x20,0x11), "" ) ) ; // series level
    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);
    if( theQuery->ValidateQuery( true ) )
      {
      // all unique keys present + required correct level + one incorrect
      return 1;
      }
    }

  //std::cout << "sucess" << std::endl;
  return 0;
}
