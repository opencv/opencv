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
 * \author Aleš Pavel
 */
require_once( 'gdcm.php' );

$reader = new Reader();
$reader->SetFilename( "test.dcm" );
$ret=$reader->Read();
if( !$ret )
{
  return 1;
}

$file = $reader->GetFile();
$ano = new Anonymizer();
$ano->SetFile($file);
$ano->RemovePrivateTags();
$ano->RemoveGroupLength();
$t = new Tag(0x10,0x10);
$ano->Replace( $t, "GDCM^PHP^Test^Hello^World" );

$g = new UIDGenerator();
$ano->Replace( new Tag(0x0008,0x0018), $g->Generate() );
$ano->Replace( new Tag(0x0020,0x000d), $g->Generate() );
$ano->Replace( new Tag(0x0020,0x000e), $g->Generate() );
$ano->Replace( new Tag(0x0020,0x0052), $g->Generate() );

$writer = new Writer();
$writer->SetFileName( "test2.dcm" );
$writer->SetFile( $ano->GetFile() );
$ret = $writer->Write();
if( !$ret )
{
  return 1;
}

?>
