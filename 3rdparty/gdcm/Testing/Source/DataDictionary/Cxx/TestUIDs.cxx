/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmUIDs.h"
#include "gdcmTesting.h"

#include <iostream>

#include <string.h> // strcmp
#include <stdlib.h> // strcmp


// This list was retrieved from:
// http://cardiacatlas.wiki.sourceforge.net/DICOM+Service+Class+Definitions
// Hum...there is not a single difference, exact same number of white space...
// apparently last modifier is 'solidether'
// http://cardiacatlas.wiki.sourceforge.net/page/diff/DICOM+Service+Class+Definitions?v1=209104&v2=209124
// could this be that they copy/paste stuff from gdcm itself ?
// how else could you explain the:
// { "1.2.840.113543.6.6.1.3.10002","Unregistred (?) Philips3D" },


static const char * const sopclassuids[][2] = {
{ "1.2.840.10008.1.1","Verification SOP Class" },
{ "1.2.840.10008.1.2","Implicit VR Little Endian: Default Transfer Syntax for DICOM" },
{ "1.2.840.10008.1.2.1","Explicit VR Little Endian" },
{ "1.2.840.10008.1.2.1.99","Deflated Explicit VR Little Endian" },
{ "1.2.840.10008.1.2.2","Explicit VR Big Endian" },
{ "1.2.840.10008.1.2.4.50","JPEG Baseline (Process 1): Default Transfer Syntax for Lossy JPEG 8 Bit Image Compression" },
{ "1.2.840.10008.1.2.4.51","JPEG Extended (Process 2 & 4): Default Transfer Syntax for Lossy JPEG 12 Bit Image Compression (Process 4 only)" },
{ "1.2.840.10008.1.2.4.52","JPEG Extended (Process 3 & 5)" },
{ "1.2.840.10008.1.2.4.53","JPEG Spectral Selection, Non-Hierarchical (Process 6 & 8)" },
{ "1.2.840.10008.1.2.4.54","JPEG Spectral Selection, Non-Hierarchical (Process 7 & 9)" },
{ "1.2.840.10008.1.2.4.55","JPEG Full Progression, Non-Hierarchical (Process 10 & 12)" },
{ "1.2.840.10008.1.2.4.56","JPEG Full Progression, Non-Hierarchical (Process 11 & 13)" },
{ "1.2.840.10008.1.2.4.57","JPEG Lossless, Non-Hierarchical (Process 14)" },
{ "1.2.840.10008.1.2.4.58","JPEG Lossless, Non-Hierarchical (Process 15)" },
{ "1.2.840.10008.1.2.4.59","JPEG Extended, Hierarchical (Process 16 & 18)" },
{ "1.2.840.10008.1.2.4.60","JPEG Extended, Hierarchical (Process 17 & 19)" },
{ "1.2.840.10008.1.2.4.61","JPEG Spectral Selection, Hierarchical (Process 20 & 22)" },
{ "1.2.840.10008.1.2.4.62","JPEG Spectral Selection, Hierarchical (Process 21 & 23)" },
{ "1.2.840.10008.1.2.4.63","JPEG Full Progression, Hierarchical (Process 24 & 26)" },
{ "1.2.840.10008.1.2.4.64","JPEG Full Progression, Hierarchical (Process 25 & 27)" },
{ "1.2.840.10008.1.2.4.65","JPEG Lossless, Hierarchical (Process 28)" },
{ "1.2.840.10008.1.2.4.66","JPEG Lossless, Hierarchical (Process 29)" },
{ "1.2.840.10008.1.2.4.70","JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1]): Default Transfer Syntax for Lossless JPEG Image Compression" },
{ "1.2.840.10008.1.2.4.80","JPEG-LS Lossless Image Compression" },
{ "1.2.840.10008.1.2.4.81","JPEG-LS Lossy (Near-Lossless) Image Compression" },
{ "1.2.840.10008.1.2.4.90","JPEG 2000 Image Compression (Lossless Only)" },
{ "1.2.840.10008.1.2.4.91","JPEG 2000 Image Compression" },
{ "1.2.840.10008.1.2.4.92","JPEG 2000 Part 2 Multi-component Image Compression (Lossless Only)" },
{ "1.2.840.10008.1.2.4.93","JPEG 2000 Part 2 Multi-component Image Compression" },
{ "1.2.840.10008.1.2.4.94","JPIP Referenced" },
{ "1.2.840.10008.1.2.4.95","JPIP Referenced Deflate" },
{ "1.2.840.10008.1.2.4.100","MPEG2 Main Profile @ Main Level" },
{ "1.2.840.10008.1.2.5","RLE Lossless" },
{ "1.2.840.10008.1.2.6.1","RFC 2557 MIME encapsulation" },
{ "1.2.840.10008.1.2.6.2","XML Encoding" },
{ "1.2.840.10008.1.3.10","Media Storage Directory Storage" },
{ "1.2.840.10008.1.4.1.1","Talairach Brain Atlas Frame of Reference" },
{ "1.2.840.10008.1.4.1.2","SPM2 T1 Frame of Reference" },
{ "1.2.840.10008.1.4.1.3","SPM2 T2 Frame of Reference" },
{ "1.2.840.10008.1.4.1.4","SPM2 PD Frame of Reference" },
{ "1.2.840.10008.1.4.1.5","SPM2 EPI Frame of Reference" },
{ "1.2.840.10008.1.4.1.6","SPM2 FIL T1 Frame of Reference" },
{ "1.2.840.10008.1.4.1.7","SPM2 PET Frame of Reference" },
{ "1.2.840.10008.1.4.1.8","SPM2 TRANSM Frame of Reference" },
{ "1.2.840.10008.1.4.1.9","SPM2 SPECT Frame of Reference" },
{ "1.2.840.10008.1.4.1.10","SPM2 GRAY Frame of Reference" },
{ "1.2.840.10008.1.4.1.11","SPM2 WHITE Frame of Reference" },
{ "1.2.840.10008.1.4.1.12","SPM2 CSF Frame of Reference" },
{ "1.2.840.10008.1.4.1.13","SPM2 BRAINMASK Frame of Reference" },
{ "1.2.840.10008.1.4.1.14","SPM2 AVG305T1 Frame of Reference" },
{ "1.2.840.10008.1.4.1.15","SPM2 AVG152T1 Frame of Reference" },
{ "1.2.840.10008.1.4.1.16","SPM2 AVG152T2 Frame of Reference" },
{ "1.2.840.10008.1.4.1.17","SPM2 AVG152PD Frame of Reference" },
{ "1.2.840.10008.1.4.1.18","SPM2 SINGLESUBJT1 Frame of Reference" },
{ "1.2.840.10008.1.4.2.1","ICBM 452 T1 Frame of Reference" },
{ "1.2.840.10008.1.4.2.2","ICBM Single Subject MRI Frame of Reference" },
{ "1.2.840.10008.1.9","Basic Study Content Notification SOP Class" },
{ "1.2.840.10008.1.20.1","Storage Commitment Push Model SOP Class" },
{ "1.2.840.10008.1.20.1.1","Storage Commitment Push Model SOP Instance" },
{ "1.2.840.10008.1.20.2","Storage Commitment Pull Model SOP Class" },
{ "1.2.840.10008.1.20.2.1","Storage Commitment Pull Model SOP Instance" },
{ "1.2.840.10008.1.40","Procedural Event Logging SOP Class" },
{ "1.2.840.10008.1.40.1","Procedural Event Logging SOP Instance" },
{ "1.2.840.10008.1.42","Substance Administration Logging SOP Class" },
{ "1.2.840.10008.1.42.1","Substance Administration Logging SOP Instance" },
{ "1.2.840.10008.2.6.1","DICOM UID Registry" },
{ "1.2.840.10008.2.16.4","DICOM Controlled Terminology" },
{ "1.2.840.10008.3.1.1.1","DICOM Application Context Name" },
{ "1.2.840.10008.3.1.2.1.1","Detached Patient Management SOP Class" },
{ "1.2.840.10008.3.1.2.1.4","Detached Patient Management Meta SOP Class" },
{ "1.2.840.10008.3.1.2.2.1","Detached Visit Management SOP Class" },
{ "1.2.840.10008.3.1.2.3.1","Detached Study Management SOP Class" },
{ "1.2.840.10008.3.1.2.3.2","Study Component Management SOP Class" },
{ "1.2.840.10008.3.1.2.3.3","Modality Performed Procedure Step SOP Class" },
{ "1.2.840.10008.3.1.2.3.4","Modality Performed Procedure Step Retrieve SOP Class" },
{ "1.2.840.10008.3.1.2.3.5","Modality Performed Procedure Step Notification SOP Class" },
{ "1.2.840.10008.3.1.2.5.1","Detached Results Management SOP Class" },
{ "1.2.840.10008.3.1.2.5.4","Detached Results Management Meta SOP Class" },
{ "1.2.840.10008.3.1.2.5.5","Detached Study Management Meta SOP Class" },
{ "1.2.840.10008.3.1.2.6.1","Detached Interpretation Management SOP Class" },
{ "1.2.840.10008.4.2","Storage Service Class" },
{ "1.2.840.10008.5.1.1.1","Basic Film Session SOP Class" },
{ "1.2.840.10008.5.1.1.2","Basic Film Box SOP Class" },
{ "1.2.840.10008.5.1.1.4","Basic Grayscale Image Box SOP Class" },
{ "1.2.840.10008.5.1.1.4.1","Basic Color Image Box SOP Class" },
{ "1.2.840.10008.5.1.1.4.2","Referenced Image Box SOP Class" },
{ "1.2.840.10008.5.1.1.9","Basic Grayscale Print Management Meta SOP Class" },
{ "1.2.840.10008.5.1.1.9.1","Referenced Grayscale Print Management Meta SOP Class" },
{ "1.2.840.10008.5.1.1.14","Print Job SOP Class" },
{ "1.2.840.10008.5.1.1.15","Basic Annotation Box SOP Class" },
{ "1.2.840.10008.5.1.1.16","Printer SOP Class" },
{ "1.2.840.10008.5.1.1.16.376","Printer Configuration Retrieval SOP Class" },
{ "1.2.840.10008.5.1.1.17","Printer SOP Instance" },
{ "1.2.840.10008.5.1.1.17.376","Printer Configuration Retrieval SOP Instance" },
{ "1.2.840.10008.5.1.1.18","Basic Color Print Management Meta SOP Class" },
{ "1.2.840.10008.5.1.1.18.1","Referenced Color Print Management Meta SOP Class" },
{ "1.2.840.10008.5.1.1.22","VOI LUT Box SOP Class" },
{ "1.2.840.10008.5.1.1.23","Presentation LUT SOP Class" },
{ "1.2.840.10008.5.1.1.24","Image Overlay Box SOP Class" },
{ "1.2.840.10008.5.1.1.24.1","Basic Print Image Overlay Box SOP Class" },
{ "1.2.840.10008.5.1.1.25","Print Queue SOP Instance" },
{ "1.2.840.10008.5.1.1.26","Print Queue Management SOP Class" },
{ "1.2.840.10008.5.1.1.27","Stored Print Storage SOP Class" },
{ "1.2.840.10008.5.1.1.29","Hardcopy Grayscale Image Storage SOP Class" },
{ "1.2.840.10008.5.1.1.30","Hardcopy Color Image Storage SOP Class" },
{ "1.2.840.10008.5.1.1.31","Pull Print Request SOP Class" },
{ "1.2.840.10008.5.1.1.32","Pull Stored Print Management Meta SOP Class" },
{ "1.2.840.10008.5.1.1.33","Media Creation Management SOP Class UID" },
{ "1.2.840.10008.5.1.4.1.1.1","Computed Radiography Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.1.1","Digital X-Ray Image Storage - For Presentation" },
{ "1.2.840.10008.5.1.4.1.1.1.1.1","Digital X-Ray Image Storage - For Processing" },
{ "1.2.840.10008.5.1.4.1.1.1.2","Digital Mammography X-Ray Image Storage - For Presentation" },
{ "1.2.840.10008.5.1.4.1.1.1.2.1","Digital Mammography X-Ray Image Storage - For Processing" },
{ "1.2.840.10008.5.1.4.1.1.1.3","Digital Intra-oral X-Ray Image Storage - For Presentation" },
{ "1.2.840.10008.5.1.4.1.1.1.3.1","Digital Intra-oral X-Ray Image Storage - For Processing" },
{ "1.2.840.10008.5.1.4.1.1.2","CT Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.2.1","Enhanced CT Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.3","Ultrasound Multi-frame Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.3.1","Ultrasound Multi-frame Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.4","MR Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.4.1","Enhanced MR Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.4.2","MR Spectroscopy Storage" },
{ "1.2.840.10008.5.1.4.1.1.5","Nuclear Medicine Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.6","Ultrasound Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.6.1","Ultrasound Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.7","Secondary Capture Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.7.1","Multi-frame Single Bit Secondary Capture Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.7.2","Multi-frame Grayscale Byte Secondary Capture Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.7.3","Multi-frame Grayscale Word Secondary Capture Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.7.4","Multi-frame True Color Secondary Capture Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.8","Standalone Overlay Storage" },
{ "1.2.840.10008.5.1.4.1.1.9","Standalone Curve Storage" },
{ "1.2.840.10008.5.1.4.1.1.9.1","Waveform Storage - Trial" },
{ "1.2.840.10008.5.1.4.1.1.9.1.1","12-lead ECG Waveform Storage" },
{ "1.2.840.10008.5.1.4.1.1.9.1.2","General ECG Waveform Storage" },
{ "1.2.840.10008.5.1.4.1.1.9.1.3","Ambulatory ECG Waveform Storage" },
{ "1.2.840.10008.5.1.4.1.1.9.2.1","Hemodynamic Waveform Storage" },
{ "1.2.840.10008.5.1.4.1.1.9.3.1","Cardiac Electrophysiology Waveform Storage" },
{ "1.2.840.10008.5.1.4.1.1.9.4.1","Basic Voice Audio Waveform Storage" },
{ "1.2.840.10008.5.1.4.1.1.10","Standalone Modality LUT Storage" },
{ "1.2.840.10008.5.1.4.1.1.11","Standalone VOI LUT Storage" },
{ "1.2.840.10008.5.1.4.1.1.11.1","Grayscale Softcopy Presentation State Storage SOP Class" },
{ "1.2.840.10008.5.1.4.1.1.11.2","Color Softcopy Presentation State Storage SOP Class" },
{ "1.2.840.10008.5.1.4.1.1.11.3","Pseudo-Color Softcopy Presentation State Storage SOP Class" },
{ "1.2.840.10008.5.1.4.1.1.11.4","Blending Softcopy Presentation State Storage SOP Class" },
{ "1.2.840.10008.5.1.4.1.1.12.1","X-Ray Angiographic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.12.1.1","Enhanced XA Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.12.2","X-Ray Radiofluoroscopic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.12.2.1","Enhanced XRF Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.13.1.1","X-Ray 3D Angiographic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.13.1.2","X-Ray 3D Craniofacial Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.12.3","X-Ray Angiographic Bi-Plane Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.20","Nuclear Medicine Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.66","Raw Data Storage" },
{ "1.2.840.10008.5.1.4.1.1.66.1","Spatial Registration Storage" },
{ "1.2.840.10008.5.1.4.1.1.66.2","Spatial Fiducials Storage" },
{ "1.2.840.10008.5.1.4.1.1.66.3","Deformable Spatial Registration Storage" },
{ "1.2.840.10008.5.1.4.1.1.66.4","Segmentation Storage" },
{ "1.2.840.10008.5.1.4.1.1.67","Real World Value Mapping Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1","VL Image Storage - Trial" },
{ "1.2.840.10008.5.1.4.1.1.77.2","VL Multi-frame Image Storage - Trial" },
{ "1.2.840.10008.5.1.4.1.1.77.1.1","VL Endoscopic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.1.1","Video Endoscopic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.2","VL Microscopic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.2.1","Video Microscopic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.3","VL Slide-Coordinates Microscopic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.4","VL Photographic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.4.1","Video Photographic Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.5.1","Ophthalmic Photography 8 Bit Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.5.2","Ophthalmic Photography 16 Bit Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.5.3","Stereometric Relationship Storage" },
{ "1.2.840.10008.5.1.4.1.1.77.1.5.4","Ophthalmic Tomography Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.88.1","Text SR Storage - Trial" },
{ "1.2.840.10008.5.1.4.1.1.88.2","Audio SR Storage - Trial" },
{ "1.2.840.10008.5.1.4.1.1.88.3","Detail SR Storage - Trial" },
{ "1.2.840.10008.5.1.4.1.1.88.4","Comprehensive SR Storage - Trial" },
{ "1.2.840.10008.5.1.4.1.1.88.11","Basic Text SR Storage" },
{ "1.2.840.10008.5.1.4.1.1.88.22","Enhanced SR Storage" },
{ "1.2.840.10008.5.1.4.1.1.88.33","Comprehensive SR Storage" },
{ "1.2.840.10008.5.1.4.1.1.88.40","Procedure Log Storage" },
{ "1.2.840.10008.5.1.4.1.1.88.50","Mammography CAD SR Storage" },
{ "1.2.840.10008.5.1.4.1.1.88.59","Key Object Selection Document Storage" },
{ "1.2.840.10008.5.1.4.1.1.88.65","Chest CAD SR Storage" },
{ "1.2.840.10008.5.1.4.1.1.88.67","X-Ray Radiation Dose SR Storage" },
{ "1.2.840.10008.5.1.4.1.1.104.1","Encapsulated PDF Storage" },
{ "1.2.840.10008.5.1.4.1.1.104.2","Encapsulated CDA Storage" },
{ "1.2.840.10008.5.1.4.1.1.128","Positron Emission Tomography Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.129","Standalone PET Curve Storage" },
{ "1.2.840.10008.5.1.4.1.1.481.1","RT Image Storage" },
{ "1.2.840.10008.5.1.4.1.1.481.2","RT Dose Storage" },
{ "1.2.840.10008.5.1.4.1.1.481.3","RT Structure Set Storage" },
{ "1.2.840.10008.5.1.4.1.1.481.4","RT Beams Treatment Record Storage" },
{ "1.2.840.10008.5.1.4.1.1.481.5","RT Plan Storage" },
{ "1.2.840.10008.5.1.4.1.1.481.6","RT Brachy Treatment Record Storage" },
{ "1.2.840.10008.5.1.4.1.1.481.7","RT Treatment Summary Record Storage" },
{ "1.2.840.10008.5.1.4.1.1.481.8","RT Ion Plan Storage" },
{ "1.2.840.10008.5.1.4.1.1.481.9","RT Ion Beams Treatment Record Storage" },
{ "1.2.840.10008.5.1.4.1.2.1.1","Patient Root Query/Retrieve Information Model - FIND" },
{ "1.2.840.10008.5.1.4.1.2.1.2","Patient Root Query/Retrieve Information Model - MOVE" },
{ "1.2.840.10008.5.1.4.1.2.1.3","Patient Root Query/Retrieve Information Model - GET" },
{ "1.2.840.10008.5.1.4.1.2.2.1","Study Root Query/Retrieve Information Model - FIND" },
{ "1.2.840.10008.5.1.4.1.2.2.2","Study Root Query/Retrieve Information Model - MOVE" },
{ "1.2.840.10008.5.1.4.1.2.2.3","Study Root Query/Retrieve Information Model - GET" },
{ "1.2.840.10008.5.1.4.1.2.3.1","Patient/Study Only Query/Retrieve Information Model - FIND" },
{ "1.2.840.10008.5.1.4.1.2.3.2","Patient/Study Only Query/Retrieve Information Model - MOVE" },
{ "1.2.840.10008.5.1.4.1.2.3.3","Patient/Study Only Query/Retrieve Information Model - GET" },
{ "1.2.840.10008.5.1.4.31","Modality Worklist Information Model - FIND" },
{ "1.2.840.10008.5.1.4.32.1","General Purpose Worklist Information Model - FIND" },
{ "1.2.840.10008.5.1.4.32.2","General Purpose Scheduled Procedure Step SOP Class" },
{ "1.2.840.10008.5.1.4.32.3","General Purpose Performed Procedure Step SOP Class" },
{ "1.2.840.10008.5.1.4.32","General Purpose Worklist Management Meta SOP Class" },
{ "1.2.840.10008.5.1.4.33","Instance Availability Notification SOP Class" },
{ "1.2.840.10008.5.1.4.34.1","RT Beams Delivery Instruction Storage (Supplement 74 Frozen Draft)" },
{ "1.2.840.10008.5.1.4.34.2","RT Conventional Machine Verification (Supplement 74 Frozen Draft)" },
{ "1.2.840.10008.5.1.4.34.3","RT Ion Machine Verification (Supplement 74 Frozen Draft)" },
{ "1.2.840.10008.5.1.4.34.4","Unified Worklist and Procedure Step Service Class" },
{ "1.2.840.10008.5.1.4.34.4.1","Unified Procedure Step - Push SOP Class" },
{ "1.2.840.10008.5.1.4.34.4.2","Unified Procedure Step - Watch SOP Class" },
{ "1.2.840.10008.5.1.4.34.4.3","Unified Procedure Step - Pull SOP Class" },
{ "1.2.840.10008.5.1.4.34.4.4","Unified Procedure Step - Event SOP Class" },
{ "1.2.840.10008.5.1.4.34.5","Unified Worklist and Procedure Step SOP Instance" },
{ "1.2.840.10008.5.1.4.37.1","General Relevant Patient Information Query" },
{ "1.2.840.10008.5.1.4.37.2","Breast Imaging Relevant Patient Information Query" },
{ "1.2.840.10008.5.1.4.37.3","Cardiac Relevant Patient Information Query" },
{ "1.2.840.10008.5.1.4.38.1","Hanging Protocol Storage" },
{ "1.2.840.10008.5.1.4.38.2","Hanging Protocol Information Model - FIND" },
{ "1.2.840.10008.5.1.4.38.3","Hanging Protocol Information Model - MOVE" },
{ "1.2.840.10008.5.1.4.41","Product Characteristics Query SOP Class" },
{ "1.2.840.10008.5.1.4.42","Substance Approval Query SOP Class" },
{ "1.2.840.10008.15.0.3.1","dicomDeviceName" },
{ "1.2.840.10008.15.0.3.2","dicomDescription" },
{ "1.2.840.10008.15.0.3.3","dicomManufacturer" },
{ "1.2.840.10008.15.0.3.4","dicomManufacturerModelName" },
{ "1.2.840.10008.15.0.3.5","dicomSoftwareVersion" },
{ "1.2.840.10008.15.0.3.6","dicomVendorData" },
{ "1.2.840.10008.15.0.3.7","dicomAETitle" },
{ "1.2.840.10008.15.0.3.8","dicomNetworkConnectionReference" },
{ "1.2.840.10008.15.0.3.9","dicomApplicationCluster" },
{ "1.2.840.10008.15.0.3.10","dicomAssociationInitiator" },
{ "1.2.840.10008.15.0.3.11","dicomAssociationAcceptor" },
{ "1.2.840.10008.15.0.3.12","dicomHostname" },
{ "1.2.840.10008.15.0.3.13","dicomPort" },
{ "1.2.840.10008.15.0.3.14","dicomSOPClass" },
{ "1.2.840.10008.15.0.3.15","dicomTransferRole" },
{ "1.2.840.10008.15.0.3.16","dicomTransferSyntax" },
{ "1.2.840.10008.15.0.3.17","dicomPrimaryDeviceType" },
{ "1.2.840.10008.15.0.3.18","dicomRelatedDeviceReference" },
{ "1.2.840.10008.15.0.3.19","dicomPreferredCalledAETitle" },
{ "1.2.840.10008.15.0.3.20","dicomTLSCyphersuite" },
{ "1.2.840.10008.15.0.3.21","dicomAuthorizedNodeCertificateReference" },
{ "1.2.840.10008.15.0.3.22","dicomThisNodeCertificateReference" },
{ "1.2.840.10008.15.0.3.23","dicomInstalled" },
{ "1.2.840.10008.15.0.3.24","dicomStationName" },
{ "1.2.840.10008.15.0.3.25","dicomDeviceSerialNumber" },
{ "1.2.840.10008.15.0.3.26","dicomInstitutionName" },
{ "1.2.840.10008.15.0.3.27","dicomInstitutionAddress" },
{ "1.2.840.10008.15.0.3.28","dicomInstitutionDepartmentName" },
{ "1.2.840.10008.15.0.3.29","dicomIssuerOfPatientID" },
{ "1.2.840.10008.15.0.3.30","dicomPreferredCallingAETitle" },
{ "1.2.840.10008.15.0.3.31","dicomSupportedCharacterSet" },
{ "1.2.840.10008.15.0.4.1","dicomConfigurationRoot" },
{ "1.2.840.10008.15.0.4.2","dicomDevicesRoot" },
{ "1.2.840.10008.15.0.4.3","dicomUniqueAETitlesRegistryRoot" },
{ "1.2.840.10008.15.0.4.4","dicomDevice" },
{ "1.2.840.10008.15.0.4.5","dicomNetworkAE" },
{ "1.2.840.10008.15.0.4.6","dicomNetworkConnection" },
{ "1.2.840.10008.15.0.4.7","dicomUniqueAETitle" },
{ "1.2.840.10008.15.0.4.8","dicomTransferCapability" },
{ "1.2.840.113619.4.2","General Electric Magnetic Resonance Image Storage" },
{ "1.2.840.113619.4.3","General Electric Computed Tomography Image Storage" },
{ "1.3.12.2.1107.5.9.1","CSA Non-Image Storage" },
{ "1.2.840.113619.4.26","GE Private 3D Model Storage" },
{ "1.2.840.113619.4.30","GE Advance (PET) Raw Data Storage" },
{ "2.16.840.1.113709.1.5.1","GEPACS_PRIVATE_IMS_INFO Storage" },
{ "1.2.840.113543.6.6.1.3.10002","Unregistred (?) Philips3D" },
{ "1.2.392.200036.9116.7.8.1.1.1","Toshiba Private Data Storage" },
{ "1.2.840.113619.4.27","GE Nuclear Medicine private SOP Class" },
//{ "1.3.46.670589.11.0.0.12.1","Philips Private Gyroscan MR Spectrum" },
{ "1.3.46.670589.11.0.0.12.1","Philips Private MR Spectrum Storage" },
//{ "1.3.46.670589.11.0.0.12.2","Philips Private Gyroscan MR Serie Data" },
{ "1.3.46.670589.11.0.0.12.2","Philips Private MR Series Data Storage" },
{ "1.3.46.670589.2.3.1.1","Philips Private Specialized XA Image" },
{ "1.3.46.670589.2.4.1.1","Philips Private CX Image Storage" },
{ "1.3.46.670589.2.5.1.1","Philips iE33 private 3D Object Storage" },
{ "1.3.46.670589.5.0.1","Philips Private Volume Storage" },
{ "1.3.46.670589.5.0.1.1","Philips Private Volume Image Reference" },
{ "1.3.46.670589.5.0.10","Philips Private MR Synthetic Image Storage" },
{ "1.3.46.670589.5.0.11","Philips Private MR Cardio Analysis Storage" },
{ "1.3.46.670589.5.0.11.1","Philips Private MR Cardio Analysis Data" },
{ "1.3.46.670589.5.0.12","Philips Private CX Synthetic Image Storage" },
{ "1.3.46.670589.5.0.13","Philips Private Perfusion Image Reference" },
{ "1.3.46.670589.5.0.14","Philips Private Perfusion Analysis Data" },
{ "1.3.46.670589.5.0.2","Philips Private 3D Object Storage" },
{ "1.3.46.670589.5.0.2.1","Philips Private 3D Object 2 Storage" },
{ "1.3.46.670589.5.0.3","Philips Private Surface Storage" },
{ "1.3.46.670589.5.0.3.1","Philips Private Surface 2 Storage" },
{ "1.3.46.670589.5.0.4","Philips Private Composite Object Storage" },
{ "1.3.46.670589.5.0.7","Philips Private MR Cardio Profile" },
{ "1.3.46.670589.5.0.8","Philips Private MR Cardio" },
{ "1.3.46.670589.5.0.9","Philips Private CT Synthetic Image Storage" },
{ "1.2.752.24.3.7.6","Sectra Compression (Private Syntax)" },
{ "1.2.752.24.3.7.7","Sectra Compression LS (Private Syntax)" },
{ "1.2.840.113619.5.2","Implicit VR Big Endian DLX (G.E Private)" },
{ NULL, NULL}
};

// Custom list:
static const char * const sopclassuids2[] = {
"1.2.840.10008.1.3.10",
"1.2.840.10008.5.1.4.1.1.1",
"1.2.840.10008.5.1.4.1.1.1.1",
"1.2.840.10008.5.1.4.1.1.11.1",
"1.2.840.10008.5.1.4.1.1.1.2",
"1.2.840.10008.5.1.4.1.1.12.1",
"1.2.840.10008.5.1.4.1.1.12.2",
"1.2.840.10008.5.1.4.1.1.128",
"1.2.840.10008.5.1.4.1.1.2",
"1.2.840.10008.5.1.4.1.1.20",
"1.2.840.10008.5.1.4.1.1.3.1",
"1.2.840.10008.5.1.4.1.1.4",
"1.2.840.10008.5.1.4.1.1.4.1",
"1.2.840.10008.5.1.4.1.1.481.3",
"1.2.840.10008.5.1.4.1.1.5",
"1.2.840.10008.5.1.4.1.1.6",
"1.2.840.10008.5.1.4.1.1.6.1",
"1.2.840.10008.5.1.4.1.1.66",
"1.2.840.10008.5.1.4.1.1.7",
"1.2.840.10008.5.1.4.1.1.88.11",
"1.2.840.10008.5.1.4.1.1.88.22",
"1.2.840.10008.5.1.4.1.1.88.3",
"1.2.840.10008.5.1.4.1.1.88.59",
"1.2.840.10008.5.1.4.1.1.9",
"1.2.840.10008.5.1.4.1.1.9.4.1",
"1.2.840.10008.5.1.4.38.1",
"1.2.840.113619.4.26",
"1.3.12.2.1107.5.9.1",
"1.3.46.670589.11.0.0.12.2",
"1.3.46.670589.5.0.1",
"1.3.46.670589.5.0.10",
"1.3.46.670589.5.0.1.1",
"1.3.46.670589.5.0.11",
"1.3.46.670589.5.0.13",
"1.3.46.670589.5.0.14",
"1.3.46.670589.5.0.2",
"1.3.46.670589.5.0.2.1",
"1.3.46.670589.5.0.3",
"1.3.46.670589.5.0.8",
//"1.3.6.1.4.1.20468.1.10", // invalid
NULL
};

int TestUIDs(int, char *[])
{
  const char* s0 = gdcm::UIDs::GetUIDString( 0 );
  if(s0) return 1;

  // {"1.2.840.10008.5.1.4.1.1.2.1","Enhanced CT Image Storage"},
  // uid_1_2_840_10008_5_1_4_1_1_2_1 = 117, // Enhanced CT Image Storage
  const char* s = gdcm::UIDs::GetUIDString( gdcm::UIDs::uid_1_2_840_10008_5_1_4_1_1_2_1 );
  if(!s) return 1;
  std::cout << s << std::endl;
  const char* n = gdcm::UIDs::GetUIDName( gdcm::UIDs::uid_1_2_840_10008_5_1_4_1_1_2_1 );
  if(!n) return 1;
  std::cout << n << std::endl;
  const char* s1 = gdcm::UIDs::GetUIDString( gdcm::UIDs::EnhancedCTImageStorage );
  if(!s1) return 1;
  std::cout << s1 << std::endl;
  const char* n1 = gdcm::UIDs::GetUIDName( gdcm::UIDs::EnhancedCTImageStorage );
  if(!n1) return 1;
  std::cout << n1 << std::endl;

  gdcm::UIDs uid;
  // valid:
  if( !uid.SetFromUID( "1.2.840.10008.5.1.4.1.1.2.1" ) )
    {
    return 1;
    }
  std::cout << "This is : " << uid.GetName() << std::endl;
  std::cout << "This is : " << uid.GetString() << std::endl;
  std::cout << uid << std::endl;
  // invalid
  if( uid.SetFromUID( "prosper youpla boum c'est le roi du pain d'epices" ) )
    {
    return 1;
    }
  if( uid.GetName() ) return 1;
  if( uid.SetFromUID( "1.2" ) )
    {
    return 1;
    }
  if( uid.GetName() ) return 1;
  if( uid.SetFromUID( "" ) )
    {
    return 1;
    }
  if( uid.GetName() ) return 1;
  // black box:
  if( uid.SetFromUID( NULL ) )
    {
    return 1;
    }
  if( uid.GetName() ) return 1;

  typedef const char* const (*mytype)[2];
  mytype sopclassuid = sopclassuids;
  while( *sopclassuid[0] )
    {
    const char *uid_str = (*sopclassuid)[0];
    const char *name_str = (*sopclassuid)[1];
    //std::cout << uid_str << std::endl;
    if( !uid.SetFromUID( uid_str ) )
      {
      std::cerr << "Invalid UID:" << uid_str << std::endl;
      return 1;
      }
    const char *name = uid.GetName();
    if( !name )
      {
      std::cerr << "problem with: " << uid_str << std::endl;
      return 1;
      }
    if( strcmp( name, name_str) != 0 )
      {
      std::cerr << "Error: " << name << " vs " << name_str << std::endl;
      return 1;
      }

    ++sopclassuid;
    }

  std::cout << "Custom List:" << std::endl;
  const char * const *s2 = sopclassuids2;
  while( *s2 )
    {
    const char *uid_str = *s2;
    if( !uid.SetFromUID( uid_str ) )
      {
      std::cerr << "Invalid UID:" << uid_str << std::endl;
      return 1;
      }

    const char *name = uid.GetName();
    if( !name )
      {
      return 1;
      }
    //std::cout << uid_str << "," << name << std::endl;
    s2++;
    }

  // Print all
  std::cout << "All:" << std::endl;
  for(unsigned int i = 0; i < gdcm::UIDs::GetNumberOfTransferSyntaxStrings(); ++i)
    {
    //const char * const * str_pair = gdcm::UIDs::GetTransferSyntaxString(i);
    uid.SetFromUID( gdcm::UIDs::GetUIDString( i+1 ) );
    //std::cout << uid << std::endl;
    if( !uid.GetName() || !uid.GetString() ) return 1;
    }

  return 0;
}
