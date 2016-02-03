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
 */
require_once( 'vtkgdcm.php' );

//$reader = new vtkGDCMImageReader();
$reader = vtkGDCMImageReader::c_New();
$reader->SetFilename( "test.dcm" );
$reader->Update();

$prop = $reader->GetMedicalImageProperties();

$n = $prop->GetNumberOfWindowLevelPresets();
print( "coudou" );
//print( "coucou %d", $n );
if( $n != 0 )
{
// Take the first one by default:
$w = 0;
$l = 0;
$a = $prop->GetNthWindowLevelPreset(0);
print( $a[0] );
//$windowlevel->SetWindow( wl[0] );
//$windowlevel->SetLevel( wl[1] );
}

/*
$renderer = vtkRenderer::c_New();

$windowlevel = vtkImageMapToWindowLevelColors::c_New();
$windowlevel->SetInput( $reader->GetOutput() );

$actor = vtkImageActor::c_New();
$actor->SetInput( $windowlevel->GetOutput() );

$renderer->AddActor( actor );

$renWin = vtkRenderWindow::c_New();
$renWin->OffScreenRenderingOn();
$renWin->AddRenderer($renderer);

$renWin->Render();

$w2if = vtkWindowToImageFilter::c_New();
$w2if->SetInput ( $renWin );

$wr = vtkPNGWriter::c_New();
$wr->SetInput( $w2if->GetOutput() );
$wr->SetFileName ( "offscreenimage.png" );
$wr->Write();
*/

?>
