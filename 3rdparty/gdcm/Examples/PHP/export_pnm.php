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
 * This simple example will read in an image file
 * and try to write out as a PNM file
 */
require_once( 'gdcm.php' );

$reader = new PixmapReader();
$reader->SetFilename( "test.dcm" );
if( !$reader->Read() )
{
return;
}

$file = $reader->GetFile();
$pixmap = $reader->GetPixmap();

print $pixmap;

$pnm = new PNMCodec();
$pnm->SetDimensions( $pixmap->GetDimensions() );
$pnm->SetPixelFormat( $pixmap->GetPixelFormat() );
$pnm->SetPhotometricInterpretation( $pixmap->GetPhotometricInterpretation() );
$in = $pixmap->GetDataElement();
$outfilename = 'test.pnm';
if( $pnm->Write( $outfilename, $in ) )
{
print "Success";
}

?>
