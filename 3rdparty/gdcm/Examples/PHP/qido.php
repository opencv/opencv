<?php
/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

require_once( 'gdcm.php' );

$remote = "www.dicomserver.co.uk";
$portno = 11112;

$scu = new ServiceClassUser();
$scu->SetHostname( $remote );
$scu->SetPort( $portno );
$scu->SetTimeout( 1000 );
//$scu->SetCalledAETitle( "GDCM_STORE" );
$scu->InitializeConnection();
$generator = new PresentationContextGenerator();
$generator->GenerateFromUID( UIDs::VerificationSOPClass );
$scu->SetPresentationContexts( $generator->GetPresentationContexts() );
$scu->StartAssociation();
$scu->SendEcho();
$scu->StopAssociation();

$findds = new DataSet();
$findquery = CompositeNetworkFunctions::ConstructQuery(
      eStudyRootType, eStudy, $findds);

// https://sourceforge.net/p/swig/bugs/1337/
// https://sourceforge.net/p/swig/bugs/1338/
// https://sourceforge.net/p/swig/bugs/1339/

?>
