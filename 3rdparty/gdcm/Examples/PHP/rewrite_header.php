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

/*
 * This simple example will read in a file
 * remove the header (Group 0x2)
 * and write out a file
 */
require_once( 'gdcm.php' );

$reader = new Reader();
$reader->SetFilename( "test.dcm" );
$reader->Read();

$file = $reader->GetFile();

$header = $file->GetHeader();
$header->Clear();

$writer = new Writer();
$writer->SetFilename( "test2.dcm" );
$writer->SetFile( $file );
$writer->Write();

?>
