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
// The output of gdcm::Reader is a gdcm::File

// the dataset is the the set of element we are interested in:
$ds = $file->GetDataSet();
print_r($ds);
$g = c_Global::getInstance();

$dicts = $g->GetDicts();
$pubdict = $dicts->GetPublicDict();

// In this example we will show why using name to lookup attribute can be
// dangerous.
$tPatientName= new Tag(0x0,0x0);
$de1 = $pubdict->GetDictEntryByName("Patient Name", $tPatientName);

printf("Found %s",$tPatientName);

// Indeed the attribute could not be found. Since DICOM 2003, Patient Name
// has become Patient's Name.

$tPatientsName = new Tag();
$de2 = $pubdict->GetDictEntryByName("Patient's Name", $tPatientsName);

printf("Found: %s",$tPatientsName);

// Let's try to read an arbitrary DICOM Attribute:
$tDoseGridScaling=new Tag();
$de3 = $pubdict->GetDictEntryByName("Dose Grid Scaling", $tDoseGridScaling);

printf("Found: %s",$tDoseGridScaling);

if( $ds->FindDataElement( $tDoseGridScaling ) )
{
  $sf= new StringFilter();
  $sf->SetFile($file);
  printf("Attribute Value as String: %s",$sf->ToString( $tDoseGridScaling ));

  // Let's check the name again:
  $pss = $sf->ToStringPair( $tDoseGridScaling );
  printf("Attribute Name Checked: %s", $pss->first);
  printf("Attribute Value (string): %s", $pss->second);

  $dgs = $ds->GetDataElement( $tDoseGridScaling );

  // Let's assume for a moment we knew the tag number:
  $at=new Tag(0x3004,0x000e);
  assert( $at.GetTag() == $tDoseGridScaling );
  $at->SetFromDataSet( $ds );
  // For the sake of long term maintenance, we will not write
  // that this particular attribute is stored as a double. What if
  // a user made a mistake. It is much safer to rely on GDCM internal
  // mechanism to deduce the VR::DS type (represented as a ieee double)
  $v = $at->GetValue();
  printf("DoseGridScaling=%s",$v);
}

?>
