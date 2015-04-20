
// GENERATED FILE DO NOT EDIT
// $ xsltproc UIDToC++.xsl Part6.xml > gdcmUIDs.h

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMUIDS_H
#define GDCMUIDS_H

#include "gdcmTypes.h"

namespace gdcm
{

/**
 * \brief all known uids
 */
class GDCM_EXPORT UIDs
{
public:
  typedef enum {
uid_1_2_840_10008_1_1 = 1, // Verification SOP Class
uid_1_2_840_10008_1_2 = 2, // Implicit VR Little Endian: Default Transfer Syntax for DICOM
uid_1_2_840_10008_1_2_1 = 3, // Explicit VR Little Endian
uid_1_2_840_10008_1_2_1_99 = 4, // Deflated Explicit VR Little Endian
uid_1_2_840_10008_1_2_2 = 5, // Explicit VR Big Endian
uid_1_2_840_10008_1_2_4_50 = 6, // JPEG Baseline (Process 1): Default Transfer Syntax for Lossy JPEG 8 Bit Image Compression
uid_1_2_840_10008_1_2_4_51 = 7, // JPEG Extended (Process 2 & 4): Default Transfer Syntax for Lossy JPEG 12 Bit Image Compression (Process 4 only)
uid_1_2_840_10008_1_2_4_52 = 8, // JPEG Extended (Process 3 & 5)
uid_1_2_840_10008_1_2_4_53 = 9, // JPEG Spectral Selection, Non-Hierarchical (Process 6 & 8)
uid_1_2_840_10008_1_2_4_54 = 10, // JPEG Spectral Selection, Non-Hierarchical (Process 7 & 9)
uid_1_2_840_10008_1_2_4_55 = 11, // JPEG Full Progression, Non-Hierarchical (Process 10 & 12)
uid_1_2_840_10008_1_2_4_56 = 12, // JPEG Full Progression, Non-Hierarchical (Process 11 & 13)
uid_1_2_840_10008_1_2_4_57 = 13, // JPEG Lossless, Non-Hierarchical (Process 14)
uid_1_2_840_10008_1_2_4_58 = 14, // JPEG Lossless, Non-Hierarchical (Process 15)
uid_1_2_840_10008_1_2_4_59 = 15, // JPEG Extended, Hierarchical (Process 16 & 18)
uid_1_2_840_10008_1_2_4_60 = 16, // JPEG Extended, Hierarchical (Process 17 & 19)
uid_1_2_840_10008_1_2_4_61 = 17, // JPEG Spectral Selection, Hierarchical (Process 20 & 22)
uid_1_2_840_10008_1_2_4_62 = 18, // JPEG Spectral Selection, Hierarchical (Process 21 & 23)
uid_1_2_840_10008_1_2_4_63 = 19, // JPEG Full Progression, Hierarchical (Process 24 & 26)
uid_1_2_840_10008_1_2_4_64 = 20, // JPEG Full Progression, Hierarchical (Process 25 & 27)
uid_1_2_840_10008_1_2_4_65 = 21, // JPEG Lossless, Hierarchical (Process 28)
uid_1_2_840_10008_1_2_4_66 = 22, // JPEG Lossless, Hierarchical (Process 29)
uid_1_2_840_10008_1_2_4_70 = 23, // JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1]): Default Transfer Syntax for Lossless JPEG Image Compression
uid_1_2_840_10008_1_2_4_80 = 24, // JPEG-LS Lossless Image Compression
uid_1_2_840_10008_1_2_4_81 = 25, // JPEG-LS Lossy (Near-Lossless) Image Compression
uid_1_2_840_10008_1_2_4_90 = 26, // JPEG 2000 Image Compression (Lossless Only)
uid_1_2_840_10008_1_2_4_91 = 27, // JPEG 2000 Image Compression
uid_1_2_840_10008_1_2_4_92 = 28, // JPEG 2000 Part 2 Multi-component Image Compression (Lossless Only)
uid_1_2_840_10008_1_2_4_93 = 29, // JPEG 2000 Part 2 Multi-component Image Compression
uid_1_2_840_10008_1_2_4_94 = 30, // JPIP Referenced
uid_1_2_840_10008_1_2_4_95 = 31, // JPIP Referenced Deflate
uid_1_2_840_10008_1_2_4_100 = 32, // MPEG2 Main Profile @ Main Level
uid_1_2_840_10008_1_2_5 = 33, // RLE Lossless
uid_1_2_840_10008_1_2_6_1 = 34, // RFC 2557 MIME encapsulation
uid_1_2_840_10008_1_2_6_2 = 35, // XML Encoding
uid_1_2_840_10008_1_3_10 = 36, // Media Storage Directory Storage
uid_1_2_840_10008_1_4_1_1 = 37, // Talairach Brain Atlas Frame of Reference
uid_1_2_840_10008_1_4_1_2 = 38, // SPM2 T1 Frame of Reference
uid_1_2_840_10008_1_4_1_3 = 39, // SPM2 T2 Frame of Reference
uid_1_2_840_10008_1_4_1_4 = 40, // SPM2 PD Frame of Reference
uid_1_2_840_10008_1_4_1_5 = 41, // SPM2 EPI Frame of Reference
uid_1_2_840_10008_1_4_1_6 = 42, // SPM2 FIL T1 Frame of Reference
uid_1_2_840_10008_1_4_1_7 = 43, // SPM2 PET Frame of Reference
uid_1_2_840_10008_1_4_1_8 = 44, // SPM2 TRANSM Frame of Reference
uid_1_2_840_10008_1_4_1_9 = 45, // SPM2 SPECT Frame of Reference
uid_1_2_840_10008_1_4_1_10 = 46, // SPM2 GRAY Frame of Reference
uid_1_2_840_10008_1_4_1_11 = 47, // SPM2 WHITE Frame of Reference
uid_1_2_840_10008_1_4_1_12 = 48, // SPM2 CSF Frame of Reference
uid_1_2_840_10008_1_4_1_13 = 49, // SPM2 BRAINMASK Frame of Reference
uid_1_2_840_10008_1_4_1_14 = 50, // SPM2 AVG305T1 Frame of Reference
uid_1_2_840_10008_1_4_1_15 = 51, // SPM2 AVG152T1 Frame of Reference
uid_1_2_840_10008_1_4_1_16 = 52, // SPM2 AVG152T2 Frame of Reference
uid_1_2_840_10008_1_4_1_17 = 53, // SPM2 AVG152PD Frame of Reference
uid_1_2_840_10008_1_4_1_18 = 54, // SPM2 SINGLESUBJT1 Frame of Reference
uid_1_2_840_10008_1_4_2_1 = 55, // ICBM 452 T1 Frame of Reference
uid_1_2_840_10008_1_4_2_2 = 56, // ICBM Single Subject MRI Frame of Reference
uid_1_2_840_10008_1_9 = 57, // Basic Study Content Notification SOP Class
uid_1_2_840_10008_1_20_1 = 58, // Storage Commitment Push Model SOP Class
uid_1_2_840_10008_1_20_1_1 = 59, // Storage Commitment Push Model SOP Instance
uid_1_2_840_10008_1_20_2 = 60, // Storage Commitment Pull Model SOP Class
uid_1_2_840_10008_1_20_2_1 = 61, // Storage Commitment Pull Model SOP Instance
uid_1_2_840_10008_1_40 = 62, // Procedural Event Logging SOP Class
uid_1_2_840_10008_1_40_1 = 63, // Procedural Event Logging SOP Instance
uid_1_2_840_10008_1_42 = 64, // Substance Administration Logging SOP Class
uid_1_2_840_10008_1_42_1 = 65, // Substance Administration Logging SOP Instance
uid_1_2_840_10008_2_6_1 = 66, // DICOM UID Registry
uid_1_2_840_10008_2_16_4 = 67, // DICOM Controlled Terminology
uid_1_2_840_10008_3_1_1_1 = 68, // DICOM Application Context Name
uid_1_2_840_10008_3_1_2_1_1 = 69, // Detached Patient Management SOP Class
uid_1_2_840_10008_3_1_2_1_4 = 70, // Detached Patient Management Meta SOP Class
uid_1_2_840_10008_3_1_2_2_1 = 71, // Detached Visit Management SOP Class
uid_1_2_840_10008_3_1_2_3_1 = 72, // Detached Study Management SOP Class
uid_1_2_840_10008_3_1_2_3_2 = 73, // Study Component Management SOP Class
uid_1_2_840_10008_3_1_2_3_3 = 74, // Modality Performed Procedure Step SOP Class
uid_1_2_840_10008_3_1_2_3_4 = 75, // Modality Performed Procedure Step Retrieve SOP Class
uid_1_2_840_10008_3_1_2_3_5 = 76, // Modality Performed Procedure Step Notification SOP Class
uid_1_2_840_10008_3_1_2_5_1 = 77, // Detached Results Management SOP Class
uid_1_2_840_10008_3_1_2_5_4 = 78, // Detached Results Management Meta SOP Class
uid_1_2_840_10008_3_1_2_5_5 = 79, // Detached Study Management Meta SOP Class
uid_1_2_840_10008_3_1_2_6_1 = 80, // Detached Interpretation Management SOP Class
uid_1_2_840_10008_4_2 = 81, // Storage Service Class
uid_1_2_840_10008_5_1_1_1 = 82, // Basic Film Session SOP Class
uid_1_2_840_10008_5_1_1_2 = 83, // Basic Film Box SOP Class
uid_1_2_840_10008_5_1_1_4 = 84, // Basic Grayscale Image Box SOP Class
uid_1_2_840_10008_5_1_1_4_1 = 85, // Basic Color Image Box SOP Class
uid_1_2_840_10008_5_1_1_4_2 = 86, // Referenced Image Box SOP Class
uid_1_2_840_10008_5_1_1_9 = 87, // Basic Grayscale Print Management Meta SOP Class
uid_1_2_840_10008_5_1_1_9_1 = 88, // Referenced Grayscale Print Management Meta SOP Class
uid_1_2_840_10008_5_1_1_14 = 89, // Print Job SOP Class
uid_1_2_840_10008_5_1_1_15 = 90, // Basic Annotation Box SOP Class
uid_1_2_840_10008_5_1_1_16 = 91, // Printer SOP Class
uid_1_2_840_10008_5_1_1_16_376 = 92, // Printer Configuration Retrieval SOP Class
uid_1_2_840_10008_5_1_1_17 = 93, // Printer SOP Instance
uid_1_2_840_10008_5_1_1_17_376 = 94, // Printer Configuration Retrieval SOP Instance
uid_1_2_840_10008_5_1_1_18 = 95, // Basic Color Print Management Meta SOP Class
uid_1_2_840_10008_5_1_1_18_1 = 96, // Referenced Color Print Management Meta SOP Class
uid_1_2_840_10008_5_1_1_22 = 97, // VOI LUT Box SOP Class
uid_1_2_840_10008_5_1_1_23 = 98, // Presentation LUT SOP Class
uid_1_2_840_10008_5_1_1_24 = 99, // Image Overlay Box SOP Class
uid_1_2_840_10008_5_1_1_24_1 = 100, // Basic Print Image Overlay Box SOP Class
uid_1_2_840_10008_5_1_1_25 = 101, // Print Queue SOP Instance
uid_1_2_840_10008_5_1_1_26 = 102, // Print Queue Management SOP Class
uid_1_2_840_10008_5_1_1_27 = 103, // Stored Print Storage SOP Class
uid_1_2_840_10008_5_1_1_29 = 104, // Hardcopy Grayscale Image Storage SOP Class
uid_1_2_840_10008_5_1_1_30 = 105, // Hardcopy Color Image Storage SOP Class
uid_1_2_840_10008_5_1_1_31 = 106, // Pull Print Request SOP Class
uid_1_2_840_10008_5_1_1_32 = 107, // Pull Stored Print Management Meta SOP Class
uid_1_2_840_10008_5_1_1_33 = 108, // Media Creation Management SOP Class UID
uid_1_2_840_10008_5_1_4_1_1_1 = 109, // Computed Radiography Image Storage
uid_1_2_840_10008_5_1_4_1_1_1_1 = 110, // Digital X-Ray Image Storage - For Presentation
uid_1_2_840_10008_5_1_4_1_1_1_1_1 = 111, // Digital X-Ray Image Storage - For Processing
uid_1_2_840_10008_5_1_4_1_1_1_2 = 112, // Digital Mammography X-Ray Image Storage - For Presentation
uid_1_2_840_10008_5_1_4_1_1_1_2_1 = 113, // Digital Mammography X-Ray Image Storage - For Processing
uid_1_2_840_10008_5_1_4_1_1_1_3 = 114, // Digital Intra-oral X-Ray Image Storage - For Presentation
uid_1_2_840_10008_5_1_4_1_1_1_3_1 = 115, // Digital Intra-oral X-Ray Image Storage - For Processing
uid_1_2_840_10008_5_1_4_1_1_2 = 116, // CT Image Storage
uid_1_2_840_10008_5_1_4_1_1_2_1 = 117, // Enhanced CT Image Storage
uid_1_2_840_10008_5_1_4_1_1_3 = 118, // Ultrasound Multi-frame Image Storage
uid_1_2_840_10008_5_1_4_1_1_3_1 = 119, // Ultrasound Multi-frame Image Storage
uid_1_2_840_10008_5_1_4_1_1_4 = 120, // MR Image Storage
uid_1_2_840_10008_5_1_4_1_1_4_1 = 121, // Enhanced MR Image Storage
uid_1_2_840_10008_5_1_4_1_1_4_2 = 122, // MR Spectroscopy Storage
uid_1_2_840_10008_5_1_4_1_1_5 = 123, // Nuclear Medicine Image Storage
uid_1_2_840_10008_5_1_4_1_1_6 = 124, // Ultrasound Image Storage
uid_1_2_840_10008_5_1_4_1_1_6_1 = 125, // Ultrasound Image Storage
uid_1_2_840_10008_5_1_4_1_1_7 = 126, // Secondary Capture Image Storage
uid_1_2_840_10008_5_1_4_1_1_7_1 = 127, // Multi-frame Single Bit Secondary Capture Image Storage
uid_1_2_840_10008_5_1_4_1_1_7_2 = 128, // Multi-frame Grayscale Byte Secondary Capture Image Storage
uid_1_2_840_10008_5_1_4_1_1_7_3 = 129, // Multi-frame Grayscale Word Secondary Capture Image Storage
uid_1_2_840_10008_5_1_4_1_1_7_4 = 130, // Multi-frame True Color Secondary Capture Image Storage
uid_1_2_840_10008_5_1_4_1_1_8 = 131, // Standalone Overlay Storage
uid_1_2_840_10008_5_1_4_1_1_9 = 132, // Standalone Curve Storage
uid_1_2_840_10008_5_1_4_1_1_9_1 = 133, // Waveform Storage - Trial
uid_1_2_840_10008_5_1_4_1_1_9_1_1 = 134, // 12-lead ECG Waveform Storage
uid_1_2_840_10008_5_1_4_1_1_9_1_2 = 135, // General ECG Waveform Storage
uid_1_2_840_10008_5_1_4_1_1_9_1_3 = 136, // Ambulatory ECG Waveform Storage
uid_1_2_840_10008_5_1_4_1_1_9_2_1 = 137, // Hemodynamic Waveform Storage
uid_1_2_840_10008_5_1_4_1_1_9_3_1 = 138, // Cardiac Electrophysiology Waveform Storage
uid_1_2_840_10008_5_1_4_1_1_9_4_1 = 139, // Basic Voice Audio Waveform Storage
uid_1_2_840_10008_5_1_4_1_1_10 = 140, // Standalone Modality LUT Storage
uid_1_2_840_10008_5_1_4_1_1_11 = 141, // Standalone VOI LUT Storage
uid_1_2_840_10008_5_1_4_1_1_11_1 = 142, // Grayscale Softcopy Presentation State Storage SOP Class
uid_1_2_840_10008_5_1_4_1_1_11_2 = 143, // Color Softcopy Presentation State Storage SOP Class
uid_1_2_840_10008_5_1_4_1_1_11_3 = 144, // Pseudo-Color Softcopy Presentation State Storage SOP Class
uid_1_2_840_10008_5_1_4_1_1_11_4 = 145, // Blending Softcopy Presentation State Storage SOP Class
uid_1_2_840_10008_5_1_4_1_1_12_1 = 146, // X-Ray Angiographic Image Storage
uid_1_2_840_10008_5_1_4_1_1_12_1_1 = 147, // Enhanced XA Image Storage
uid_1_2_840_10008_5_1_4_1_1_12_2 = 148, // X-Ray Radiofluoroscopic Image Storage
uid_1_2_840_10008_5_1_4_1_1_12_2_1 = 149, // Enhanced XRF Image Storage
uid_1_2_840_10008_5_1_4_1_1_13_1_1 = 150, // X-Ray 3D Angiographic Image Storage
uid_1_2_840_10008_5_1_4_1_1_13_1_2 = 151, // X-Ray 3D Craniofacial Image Storage
uid_1_2_840_10008_5_1_4_1_1_12_3 = 152, // X-Ray Angiographic Bi-Plane Image Storage
uid_1_2_840_10008_5_1_4_1_1_20 = 153, // Nuclear Medicine Image Storage
uid_1_2_840_10008_5_1_4_1_1_66 = 154, // Raw Data Storage
uid_1_2_840_10008_5_1_4_1_1_66_1 = 155, // Spatial Registration Storage
uid_1_2_840_10008_5_1_4_1_1_66_2 = 156, // Spatial Fiducials Storage
uid_1_2_840_10008_5_1_4_1_1_66_3 = 157, // Deformable Spatial Registration Storage
uid_1_2_840_10008_5_1_4_1_1_66_4 = 158, // Segmentation Storage
uid_1_2_840_10008_5_1_4_1_1_67 = 159, // Real World Value Mapping Storage
uid_1_2_840_10008_5_1_4_1_1_77_1 = 160, // VL Image Storage - Trial
uid_1_2_840_10008_5_1_4_1_1_77_2 = 161, // VL Multi-frame Image Storage - Trial
uid_1_2_840_10008_5_1_4_1_1_77_1_1 = 162, // VL Endoscopic Image Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_1_1 = 163, // Video Endoscopic Image Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_2 = 164, // VL Microscopic Image Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_2_1 = 165, // Video Microscopic Image Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_3 = 166, // VL Slide-Coordinates Microscopic Image Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_4 = 167, // VL Photographic Image Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_4_1 = 168, // Video Photographic Image Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_5_1 = 169, // Ophthalmic Photography 8 Bit Image Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_5_2 = 170, // Ophthalmic Photography 16 Bit Image Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_5_3 = 171, // Stereometric Relationship Storage
uid_1_2_840_10008_5_1_4_1_1_77_1_5_4 = 172, // Ophthalmic Tomography Image Storage
uid_1_2_840_10008_5_1_4_1_1_88_1 = 173, // Text SR Storage - Trial
uid_1_2_840_10008_5_1_4_1_1_88_2 = 174, // Audio SR Storage - Trial
uid_1_2_840_10008_5_1_4_1_1_88_3 = 175, // Detail SR Storage - Trial
uid_1_2_840_10008_5_1_4_1_1_88_4 = 176, // Comprehensive SR Storage - Trial
uid_1_2_840_10008_5_1_4_1_1_88_11 = 177, // Basic Text SR Storage
uid_1_2_840_10008_5_1_4_1_1_88_22 = 178, // Enhanced SR Storage
uid_1_2_840_10008_5_1_4_1_1_88_33 = 179, // Comprehensive SR Storage
uid_1_2_840_10008_5_1_4_1_1_88_40 = 180, // Procedure Log Storage
uid_1_2_840_10008_5_1_4_1_1_88_50 = 181, // Mammography CAD SR Storage
uid_1_2_840_10008_5_1_4_1_1_88_59 = 182, // Key Object Selection Document Storage
uid_1_2_840_10008_5_1_4_1_1_88_65 = 183, // Chest CAD SR Storage
uid_1_2_840_10008_5_1_4_1_1_88_67 = 184, // X-Ray Radiation Dose SR Storage
uid_1_2_840_10008_5_1_4_1_1_104_1 = 185, // Encapsulated PDF Storage
uid_1_2_840_10008_5_1_4_1_1_104_2 = 186, // Encapsulated CDA Storage
uid_1_2_840_10008_5_1_4_1_1_128 = 187, // Positron Emission Tomography Image Storage
uid_1_2_840_10008_5_1_4_1_1_129 = 188, // Standalone PET Curve Storage
uid_1_2_840_10008_5_1_4_1_1_481_1 = 189, // RT Image Storage
uid_1_2_840_10008_5_1_4_1_1_481_2 = 190, // RT Dose Storage
uid_1_2_840_10008_5_1_4_1_1_481_3 = 191, // RT Structure Set Storage
uid_1_2_840_10008_5_1_4_1_1_481_4 = 192, // RT Beams Treatment Record Storage
uid_1_2_840_10008_5_1_4_1_1_481_5 = 193, // RT Plan Storage
uid_1_2_840_10008_5_1_4_1_1_481_6 = 194, // RT Brachy Treatment Record Storage
uid_1_2_840_10008_5_1_4_1_1_481_7 = 195, // RT Treatment Summary Record Storage
uid_1_2_840_10008_5_1_4_1_1_481_8 = 196, // RT Ion Plan Storage
uid_1_2_840_10008_5_1_4_1_1_481_9 = 197, // RT Ion Beams Treatment Record Storage
uid_1_2_840_10008_5_1_4_1_2_1_1 = 198, // Patient Root Query/Retrieve Information Model - FIND
uid_1_2_840_10008_5_1_4_1_2_1_2 = 199, // Patient Root Query/Retrieve Information Model - MOVE
uid_1_2_840_10008_5_1_4_1_2_1_3 = 200, // Patient Root Query/Retrieve Information Model - GET
uid_1_2_840_10008_5_1_4_1_2_2_1 = 201, // Study Root Query/Retrieve Information Model - FIND
uid_1_2_840_10008_5_1_4_1_2_2_2 = 202, // Study Root Query/Retrieve Information Model - MOVE
uid_1_2_840_10008_5_1_4_1_2_2_3 = 203, // Study Root Query/Retrieve Information Model - GET
uid_1_2_840_10008_5_1_4_1_2_3_1 = 204, // Patient/Study Only Query/Retrieve Information Model - FIND
uid_1_2_840_10008_5_1_4_1_2_3_2 = 205, // Patient/Study Only Query/Retrieve Information Model - MOVE
uid_1_2_840_10008_5_1_4_1_2_3_3 = 206, // Patient/Study Only Query/Retrieve Information Model - GET
uid_1_2_840_10008_5_1_4_31 = 207, // Modality Worklist Information Model - FIND
uid_1_2_840_10008_5_1_4_32_1 = 208, // General Purpose Worklist Information Model - FIND
uid_1_2_840_10008_5_1_4_32_2 = 209, // General Purpose Scheduled Procedure Step SOP Class
uid_1_2_840_10008_5_1_4_32_3 = 210, // General Purpose Performed Procedure Step SOP Class
uid_1_2_840_10008_5_1_4_32 = 211, // General Purpose Worklist Management Meta SOP Class
uid_1_2_840_10008_5_1_4_33 = 212, // Instance Availability Notification SOP Class
uid_1_2_840_10008_5_1_4_34_1 = 213, // RT Beams Delivery Instruction Storage (Supplement 74 Frozen Draft)
uid_1_2_840_10008_5_1_4_34_2 = 214, // RT Conventional Machine Verification (Supplement 74 Frozen Draft)
uid_1_2_840_10008_5_1_4_34_3 = 215, // RT Ion Machine Verification (Supplement 74 Frozen Draft)
uid_1_2_840_10008_5_1_4_34_4 = 216, // Unified Worklist and Procedure Step Service Class
uid_1_2_840_10008_5_1_4_34_4_1 = 217, // Unified Procedure Step - Push SOP Class
uid_1_2_840_10008_5_1_4_34_4_2 = 218, // Unified Procedure Step - Watch SOP Class
uid_1_2_840_10008_5_1_4_34_4_3 = 219, // Unified Procedure Step - Pull SOP Class
uid_1_2_840_10008_5_1_4_34_4_4 = 220, // Unified Procedure Step - Event SOP Class
uid_1_2_840_10008_5_1_4_34_5 = 221, // Unified Worklist and Procedure Step SOP Instance
uid_1_2_840_10008_5_1_4_37_1 = 222, // General Relevant Patient Information Query
uid_1_2_840_10008_5_1_4_37_2 = 223, // Breast Imaging Relevant Patient Information Query
uid_1_2_840_10008_5_1_4_37_3 = 224, // Cardiac Relevant Patient Information Query
uid_1_2_840_10008_5_1_4_38_1 = 225, // Hanging Protocol Storage
uid_1_2_840_10008_5_1_4_38_2 = 226, // Hanging Protocol Information Model - FIND
uid_1_2_840_10008_5_1_4_38_3 = 227, // Hanging Protocol Information Model - MOVE
uid_1_2_840_10008_5_1_4_41 = 228, // Product Characteristics Query SOP Class
uid_1_2_840_10008_5_1_4_42 = 229, // Substance Approval Query SOP Class
uid_1_2_840_10008_15_0_3_1 = 230, // dicomDeviceName
uid_1_2_840_10008_15_0_3_2 = 231, // dicomDescription
uid_1_2_840_10008_15_0_3_3 = 232, // dicomManufacturer
uid_1_2_840_10008_15_0_3_4 = 233, // dicomManufacturerModelName
uid_1_2_840_10008_15_0_3_5 = 234, // dicomSoftwareVersion
uid_1_2_840_10008_15_0_3_6 = 235, // dicomVendorData
uid_1_2_840_10008_15_0_3_7 = 236, // dicomAETitle
uid_1_2_840_10008_15_0_3_8 = 237, // dicomNetworkConnectionReference
uid_1_2_840_10008_15_0_3_9 = 238, // dicomApplicationCluster
uid_1_2_840_10008_15_0_3_10 = 239, // dicomAssociationInitiator
uid_1_2_840_10008_15_0_3_11 = 240, // dicomAssociationAcceptor
uid_1_2_840_10008_15_0_3_12 = 241, // dicomHostname
uid_1_2_840_10008_15_0_3_13 = 242, // dicomPort
uid_1_2_840_10008_15_0_3_14 = 243, // dicomSOPClass
uid_1_2_840_10008_15_0_3_15 = 244, // dicomTransferRole
uid_1_2_840_10008_15_0_3_16 = 245, // dicomTransferSyntax
uid_1_2_840_10008_15_0_3_17 = 246, // dicomPrimaryDeviceType
uid_1_2_840_10008_15_0_3_18 = 247, // dicomRelatedDeviceReference
uid_1_2_840_10008_15_0_3_19 = 248, // dicomPreferredCalledAETitle
uid_1_2_840_10008_15_0_3_20 = 249, // dicomTLSCyphersuite
uid_1_2_840_10008_15_0_3_21 = 250, // dicomAuthorizedNodeCertificateReference
uid_1_2_840_10008_15_0_3_22 = 251, // dicomThisNodeCertificateReference
uid_1_2_840_10008_15_0_3_23 = 252, // dicomInstalled
uid_1_2_840_10008_15_0_3_24 = 253, // dicomStationName
uid_1_2_840_10008_15_0_3_25 = 254, // dicomDeviceSerialNumber
uid_1_2_840_10008_15_0_3_26 = 255, // dicomInstitutionName
uid_1_2_840_10008_15_0_3_27 = 256, // dicomInstitutionAddress
uid_1_2_840_10008_15_0_3_28 = 257, // dicomInstitutionDepartmentName
uid_1_2_840_10008_15_0_3_29 = 258, // dicomIssuerOfPatientID
uid_1_2_840_10008_15_0_3_30 = 259, // dicomPreferredCallingAETitle
uid_1_2_840_10008_15_0_3_31 = 260, // dicomSupportedCharacterSet
uid_1_2_840_10008_15_0_4_1 = 261, // dicomConfigurationRoot
uid_1_2_840_10008_15_0_4_2 = 262, // dicomDevicesRoot
uid_1_2_840_10008_15_0_4_3 = 263, // dicomUniqueAETitlesRegistryRoot
uid_1_2_840_10008_15_0_4_4 = 264, // dicomDevice
uid_1_2_840_10008_15_0_4_5 = 265, // dicomNetworkAE
uid_1_2_840_10008_15_0_4_6 = 266, // dicomNetworkConnection
uid_1_2_840_10008_15_0_4_7 = 267, // dicomUniqueAETitle
uid_1_2_840_10008_15_0_4_8 = 268, // dicomTransferCapability
//frameref_1_2_840_10008_1_4_1_1 = 269, // Talairach Brain Atlas Frame of Reference
//frameref_1_2_840_10008_1_4_1_2 = 270, // SPM2 T1 Frame of Reference
//frameref_1_2_840_10008_1_4_1_3 = 271, // SPM2 T2 Frame of Reference
//frameref_1_2_840_10008_1_4_1_4 = 272, // SPM2 PD Frame of Reference
//frameref_1_2_840_10008_1_4_1_5 = 273, // SPM2 EPI Frame of Reference
//frameref_1_2_840_10008_1_4_1_6 = 274, // SPM2 FIL T1 Frame of Reference
//frameref_1_2_840_10008_1_4_1_7 = 275, // SPM2 PET Frame of Reference
//frameref_1_2_840_10008_1_4_1_8 = 276, // SPM2 TRANSM Frame of Reference
//frameref_1_2_840_10008_1_4_1_9 = 277, // SPM2 SPECT Frame of Reference
//frameref_1_2_840_10008_1_4_1_10 = 278, // SPM2 GRAY Frame of Reference
//frameref_1_2_840_10008_1_4_1_11 = 279, // SPM2 WHITE Frame of Reference
//frameref_1_2_840_10008_1_4_1_12 = 280, // SPM2 CSF Frame of Reference
//frameref_1_2_840_10008_1_4_1_13 = 281, // SPM2 BRAINMASK Frame of Reference
//frameref_1_2_840_10008_1_4_1_14 = 282, // SPM2 AVG305T1 Frame of Reference
//frameref_1_2_840_10008_1_4_1_15 = 283, // SPM2 AVG152T1 Frame of Reference
//frameref_1_2_840_10008_1_4_1_16 = 284, // SPM2 AVG152T2 Frame of Reference
//frameref_1_2_840_10008_1_4_1_17 = 285, // SPM2 AVG152PD Frame of Reference
//frameref_1_2_840_10008_1_4_1_18 = 286, // SPM2 SINGLESUBJT1 Frame of Reference
//frameref_1_2_840_10008_1_4_2_1 = 287, // ICBM 452 T1 Frame of Reference
//frameref_1_2_840_10008_1_4_2_2 = 288, // ICBM Single Subject MRI Frame of Reference
//uid_1_2_840_113619_4_2 = 500,
//uid_1_2_840_113619_4_3 = 501,
//uid_1_3_12_2_1107_5_9_1 = 502,
//uid_1_2_840_113619_4_26 = 503,
//uid_1_2_840_113619_4_30 = 504,
//uid_2_16_840_1_113709_1_5_1 = 505,
//uid_1_2_840_113543_6_6_1_3_10002 = 506,
//uid_1_2_392_200036_9116_7_8_1_1_1 = 507,
//uid_end = 1000
uid_1_2_840_10008_5_1_4_1_1_77_1_6, // VL Whole Slide Microscopy Image Storage
uid_1_2_840_10008_5_1_4_1_1_6_2, // EnhancedUSVolumeStorage
uid_1_2_840_10008_5_1_4_1_1_66_5, // Surface Segmentation Storage
uid_1_2_840_10008_5_1_4_1_1_13_1_3 // Breast Tomosynthesis Image Storage

} TSType;
  typedef enum {
VerificationSOPClass = 1, // Verification SOP Class
ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM = 2, // Implicit VR Little Endian: Default Transfer Syntax for DICOM
ExplicitVRLittleEndian = 3, // Explicit VR Little Endian
DeflatedExplicitVRLittleEndian = 4, // Deflated Explicit VR Little Endian
ExplicitVRBigEndian = 5, // Explicit VR Big Endian
JPEGBaselineProcess1DefaultTransferSyntaxforLossyJPEG8BitImageCompression = 6, // JPEG Baseline (Process 1): Default Transfer Syntax for Lossy JPEG 8 Bit Image Compression
JPEGExtendedProcess24DefaultTransferSyntaxforLossyJPEG12BitImageCompressionProcess4only = 7, // JPEG Extended (Process 2 & 4): Default Transfer Syntax for Lossy JPEG 12 Bit Image Compression (Process 4 only)
JPEGExtendedProcess35Retired = 8, // JPEG Extended (Process 3 & 5)
JPEGSpectralSelectionNonHierarchicalProcess68Retired = 9, // JPEG Spectral Selection, Non-Hierarchical (Process 6 & 8)
JPEGSpectralSelectionNonHierarchicalProcess79Retired = 10, // JPEG Spectral Selection, Non-Hierarchical (Process 7 & 9)
JPEGFullProgressionNonHierarchicalProcess1012Retired = 11, // JPEG Full Progression, Non-Hierarchical (Process 10 & 12)
JPEGFullProgressionNonHierarchicalProcess1113Retired = 12, // JPEG Full Progression, Non-Hierarchical (Process 11 & 13)
JPEGLosslessNonHierarchicalProcess14 = 13, // JPEG Lossless, Non-Hierarchical (Process 14)
JPEGLosslessNonHierarchicalProcess15Retired = 14, // JPEG Lossless, Non-Hierarchical (Process 15)
JPEGExtendedHierarchicalProcess1618Retired = 15, // JPEG Extended, Hierarchical (Process 16 & 18)
JPEGExtendedHierarchicalProcess1719Retired = 16, // JPEG Extended, Hierarchical (Process 17 & 19)
JPEGSpectralSelectionHierarchicalProcess2022Retired = 17, // JPEG Spectral Selection, Hierarchical (Process 20 & 22)
JPEGSpectralSelectionHierarchicalProcess2123Retired = 18, // JPEG Spectral Selection, Hierarchical (Process 21 & 23)
JPEGFullProgressionHierarchicalProcess2426Retired = 19, // JPEG Full Progression, Hierarchical (Process 24 & 26)
JPEGFullProgressionHierarchicalProcess2527Retired = 20, // JPEG Full Progression, Hierarchical (Process 25 & 27)
JPEGLosslessHierarchicalProcess28Retired = 21, // JPEG Lossless, Hierarchical (Process 28)
JPEGLosslessHierarchicalProcess29Retired = 22, // JPEG Lossless, Hierarchical (Process 29)
JPEGLosslessNonHierarchicalFirstOrderPredictionProcess14SelectionValue1DefaultTransferSyntaxforLosslessJPEGImageCompression = 23, // JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1]): Default Transfer Syntax for Lossless JPEG Image Compression
JPEGLSLosslessImageCompression = 24, // JPEG-LS Lossless Image Compression
JPEGLSLossyNearLosslessImageCompression = 25, // JPEG-LS Lossy (Near-Lossless) Image Compression
JPEG2000ImageCompressionLosslessOnly = 26, // JPEG 2000 Image Compression (Lossless Only)
JPEG2000ImageCompression = 27, // JPEG 2000 Image Compression
JPEG2000Part2MulticomponentImageCompressionLosslessOnly = 28, // JPEG 2000 Part 2 Multi-component Image Compression (Lossless Only)
JPEG2000Part2MulticomponentImageCompression = 29, // JPEG 2000 Part 2 Multi-component Image Compression
JPIPReferenced = 30, // JPIP Referenced
JPIPReferencedDeflate = 31, // JPIP Referenced Deflate
MPEG2MainProfileMainLevel = 32, // MPEG2 Main Profile @ Main Level
RLELossless = 33, // RLE Lossless
RFC2557MIMEencapsulation = 34, // RFC 2557 MIME encapsulation
XMLEncoding = 35, // XML Encoding
MediaStorageDirectoryStorage = 36, // Media Storage Directory Storage
TalairachBrainAtlasFrameofReference = 37, // Talairach Brain Atlas Frame of Reference
SPM2T1FrameofReference = 38, // SPM2 T1 Frame of Reference
SPM2T2FrameofReference = 39, // SPM2 T2 Frame of Reference
SPM2PDFrameofReference = 40, // SPM2 PD Frame of Reference
SPM2EPIFrameofReference = 41, // SPM2 EPI Frame of Reference
SPM2FILT1FrameofReference = 42, // SPM2 FIL T1 Frame of Reference
SPM2PETFrameofReference = 43, // SPM2 PET Frame of Reference
SPM2TRANSMFrameofReference = 44, // SPM2 TRANSM Frame of Reference
SPM2SPECTFrameofReference = 45, // SPM2 SPECT Frame of Reference
SPM2GRAYFrameofReference = 46, // SPM2 GRAY Frame of Reference
SPM2WHITEFrameofReference = 47, // SPM2 WHITE Frame of Reference
SPM2CSFFrameofReference = 48, // SPM2 CSF Frame of Reference
SPM2BRAINMASKFrameofReference = 49, // SPM2 BRAINMASK Frame of Reference
SPM2AVG305T1FrameofReference = 50, // SPM2 AVG305T1 Frame of Reference
SPM2AVG152T1FrameofReference = 51, // SPM2 AVG152T1 Frame of Reference
SPM2AVG152T2FrameofReference = 52, // SPM2 AVG152T2 Frame of Reference
SPM2AVG152PDFrameofReference = 53, // SPM2 AVG152PD Frame of Reference
SPM2SINGLESUBJT1FrameofReference = 54, // SPM2 SINGLESUBJT1 Frame of Reference
ICBM452T1FrameofReference = 55, // ICBM 452 T1 Frame of Reference
ICBMSingleSubjectMRIFrameofReference = 56, // ICBM Single Subject MRI Frame of Reference
BasicStudyContentNotificationSOPClassRetired = 57, // Basic Study Content Notification SOP Class
StorageCommitmentPushModelSOPClass = 58, // Storage Commitment Push Model SOP Class
StorageCommitmentPushModelSOPInstance = 59, // Storage Commitment Push Model SOP Instance
StorageCommitmentPullModelSOPClassRetired = 60, // Storage Commitment Pull Model SOP Class
StorageCommitmentPullModelSOPInstanceRetired = 61, // Storage Commitment Pull Model SOP Instance
ProceduralEventLoggingSOPClass = 62, // Procedural Event Logging SOP Class
ProceduralEventLoggingSOPInstance = 63, // Procedural Event Logging SOP Instance
SubstanceAdministrationLoggingSOPClass = 64, // Substance Administration Logging SOP Class
SubstanceAdministrationLoggingSOPInstance = 65, // Substance Administration Logging SOP Instance
DICOMUIDRegistry = 66, // DICOM UID Registry
DICOMControlledTerminology = 67, // DICOM Controlled Terminology
DICOMApplicationContextName = 68, // DICOM Application Context Name
DetachedPatientManagementSOPClassRetired = 69, // Detached Patient Management SOP Class
DetachedPatientManagementMetaSOPClassRetired = 70, // Detached Patient Management Meta SOP Class
DetachedVisitManagementSOPClassRetired = 71, // Detached Visit Management SOP Class
DetachedStudyManagementSOPClassRetired = 72, // Detached Study Management SOP Class
StudyComponentManagementSOPClassRetired = 73, // Study Component Management SOP Class
ModalityPerformedProcedureStepSOPClass = 74, // Modality Performed Procedure Step SOP Class
ModalityPerformedProcedureStepRetrieveSOPClass = 75, // Modality Performed Procedure Step Retrieve SOP Class
ModalityPerformedProcedureStepNotificationSOPClass = 76, // Modality Performed Procedure Step Notification SOP Class
DetachedResultsManagementSOPClassRetired = 77, // Detached Results Management SOP Class
DetachedResultsManagementMetaSOPClassRetired = 78, // Detached Results Management Meta SOP Class
DetachedStudyManagementMetaSOPClassRetired = 79, // Detached Study Management Meta SOP Class
DetachedInterpretationManagementSOPClassRetired = 80, // Detached Interpretation Management SOP Class
StorageServiceClass = 81, // Storage Service Class
BasicFilmSessionSOPClass = 82, // Basic Film Session SOP Class
BasicFilmBoxSOPClass = 83, // Basic Film Box SOP Class
BasicGrayscaleImageBoxSOPClass = 84, // Basic Grayscale Image Box SOP Class
BasicColorImageBoxSOPClass = 85, // Basic Color Image Box SOP Class
ReferencedImageBoxSOPClassRetired = 86, // Referenced Image Box SOP Class
BasicGrayscalePrintManagementMetaSOPClass = 87, // Basic Grayscale Print Management Meta SOP Class
ReferencedGrayscalePrintManagementMetaSOPClassRetired = 88, // Referenced Grayscale Print Management Meta SOP Class
PrintJobSOPClass = 89, // Print Job SOP Class
BasicAnnotationBoxSOPClass = 90, // Basic Annotation Box SOP Class
PrinterSOPClass = 91, // Printer SOP Class
PrinterConfigurationRetrievalSOPClass = 92, // Printer Configuration Retrieval SOP Class
PrinterSOPInstance = 93, // Printer SOP Instance
PrinterConfigurationRetrievalSOPInstance = 94, // Printer Configuration Retrieval SOP Instance
BasicColorPrintManagementMetaSOPClass = 95, // Basic Color Print Management Meta SOP Class
ReferencedColorPrintManagementMetaSOPClassRetired = 96, // Referenced Color Print Management Meta SOP Class
VOILUTBoxSOPClass = 97, // VOI LUT Box SOP Class
PresentationLUTSOPClass = 98, // Presentation LUT SOP Class
ImageOverlayBoxSOPClassRetired = 99, // Image Overlay Box SOP Class
BasicPrintImageOverlayBoxSOPClassRetired = 100, // Basic Print Image Overlay Box SOP Class
PrintQueueSOPInstanceRetired = 101, // Print Queue SOP Instance
PrintQueueManagementSOPClassRetired = 102, // Print Queue Management SOP Class
StoredPrintStorageSOPClassRetired = 103, // Stored Print Storage SOP Class
HardcopyGrayscaleImageStorageSOPClassRetired = 104, // Hardcopy Grayscale Image Storage SOP Class
HardcopyColorImageStorageSOPClassRetired = 105, // Hardcopy Color Image Storage SOP Class
PullPrintRequestSOPClassRetired = 106, // Pull Print Request SOP Class
PullStoredPrintManagementMetaSOPClassRetired = 107, // Pull Stored Print Management Meta SOP Class
MediaCreationManagementSOPClassUID = 108, // Media Creation Management SOP Class UID
ComputedRadiographyImageStorage = 109, // Computed Radiography Image Storage
DigitalXRayImageStorageForPresentation = 110, // Digital X-Ray Image Storage - For Presentation
DigitalXRayImageStorageForProcessing = 111, // Digital X-Ray Image Storage - For Processing
DigitalMammographyXRayImageStorageForPresentation = 112, // Digital Mammography X-Ray Image Storage - For Presentation
DigitalMammographyXRayImageStorageForProcessing = 113, // Digital Mammography X-Ray Image Storage - For Processing
DigitalIntraoralXRayImageStorageForPresentation = 114, // Digital Intra-oral X-Ray Image Storage - For Presentation
DigitalIntraoralXRayImageStorageForProcessing = 115, // Digital Intra-oral X-Ray Image Storage - For Processing
CTImageStorage = 116, // CT Image Storage
EnhancedCTImageStorage = 117, // Enhanced CT Image Storage
UltrasoundMultiframeImageStorageRetired = 118, // Ultrasound Multi-frame Image Storage
UltrasoundMultiframeImageStorage = 119, // Ultrasound Multi-frame Image Storage
MRImageStorage = 120, // MR Image Storage
EnhancedMRImageStorage = 121, // Enhanced MR Image Storage
MRSpectroscopyStorage = 122, // MR Spectroscopy Storage
NuclearMedicineImageStorageRetired = 123, // Nuclear Medicine Image Storage
UltrasoundImageStorageRetired = 124, // Ultrasound Image Storage
UltrasoundImageStorage = 125, // Ultrasound Image Storage
SecondaryCaptureImageStorage = 126, // Secondary Capture Image Storage
MultiframeSingleBitSecondaryCaptureImageStorage = 127, // Multi-frame Single Bit Secondary Capture Image Storage
MultiframeGrayscaleByteSecondaryCaptureImageStorage = 128, // Multi-frame Grayscale Byte Secondary Capture Image Storage
MultiframeGrayscaleWordSecondaryCaptureImageStorage = 129, // Multi-frame Grayscale Word Secondary Capture Image Storage
MultiframeTrueColorSecondaryCaptureImageStorage = 130, // Multi-frame True Color Secondary Capture Image Storage
StandaloneOverlayStorageRetired = 131, // Standalone Overlay Storage
StandaloneCurveStorageRetired = 132, // Standalone Curve Storage
WaveformStorageTrialRetired = 133, // Waveform Storage - Trial
//12leadECGWaveformStorage = 134, // 12-lead ECG Waveform Storage
GeneralECGWaveformStorage = 135, // General ECG Waveform Storage
AmbulatoryECGWaveformStorage = 136, // Ambulatory ECG Waveform Storage
HemodynamicWaveformStorage = 137, // Hemodynamic Waveform Storage
CardiacElectrophysiologyWaveformStorage = 138, // Cardiac Electrophysiology Waveform Storage
BasicVoiceAudioWaveformStorage = 139, // Basic Voice Audio Waveform Storage
StandaloneModalityLUTStorageRetired = 140, // Standalone Modality LUT Storage
StandaloneVOILUTStorageRetired = 141, // Standalone VOI LUT Storage
GrayscaleSoftcopyPresentationStateStorageSOPClass = 142, // Grayscale Softcopy Presentation State Storage SOP Class
ColorSoftcopyPresentationStateStorageSOPClass = 143, // Color Softcopy Presentation State Storage SOP Class
PseudoColorSoftcopyPresentationStateStorageSOPClass = 144, // Pseudo-Color Softcopy Presentation State Storage SOP Class
BlendingSoftcopyPresentationStateStorageSOPClass = 145, // Blending Softcopy Presentation State Storage SOP Class
XRayAngiographicImageStorage = 146, // X-Ray Angiographic Image Storage
EnhancedXAImageStorage = 147, // Enhanced XA Image Storage
XRayRadiofluoroscopicImageStorage = 148, // X-Ray Radiofluoroscopic Image Storage
EnhancedXRFImageStorage = 149, // Enhanced XRF Image Storage
XRay3DAngiographicImageStorage = 150, // X-Ray 3D Angiographic Image Storage
XRay3DCraniofacialImageStorage = 151, // X-Ray 3D Craniofacial Image Storage
XRayAngiographicBiPlaneImageStorageRetired = 152, // X-Ray Angiographic Bi-Plane Image Storage
NuclearMedicineImageStorage = 153, // Nuclear Medicine Image Storage
RawDataStorage = 154, // Raw Data Storage
SpatialRegistrationStorage = 155, // Spatial Registration Storage
SpatialFiducialsStorage = 156, // Spatial Fiducials Storage
DeformableSpatialRegistrationStorage = 157, // Deformable Spatial Registration Storage
SegmentationStorage = 158, // Segmentation Storage
RealWorldValueMappingStorage = 159, // Real World Value Mapping Storage
VLImageStorageTrialRetired = 160, // VL Image Storage - Trial
VLMultiframeImageStorageTrialRetired = 161, // VL Multi-frame Image Storage - Trial
VLEndoscopicImageStorage = 162, // VL Endoscopic Image Storage
VideoEndoscopicImageStorage = 163, // Video Endoscopic Image Storage
VLMicroscopicImageStorage = 164, // VL Microscopic Image Storage
VideoMicroscopicImageStorage = 165, // Video Microscopic Image Storage
VLSlideCoordinatesMicroscopicImageStorage = 166, // VL Slide-Coordinates Microscopic Image Storage
VLPhotographicImageStorage = 167, // VL Photographic Image Storage
VideoPhotographicImageStorage = 168, // Video Photographic Image Storage
OphthalmicPhotography8BitImageStorage = 169, // Ophthalmic Photography 8 Bit Image Storage
OphthalmicPhotography16BitImageStorage = 170, // Ophthalmic Photography 16 Bit Image Storage
StereometricRelationshipStorage = 171, // Stereometric Relationship Storage
OphthalmicTomographyImageStorage = 172, // Ophthalmic Tomography Image Storage
TextSRStorageTrialRetired = 173, // Text SR Storage - Trial
AudioSRStorageTrialRetired = 174, // Audio SR Storage - Trial
DetailSRStorageTrialRetired = 175, // Detail SR Storage - Trial
ComprehensiveSRStorageTrialRetired = 176, // Comprehensive SR Storage - Trial
BasicTextSRStorage = 177, // Basic Text SR Storage
EnhancedSRStorage = 178, // Enhanced SR Storage
ComprehensiveSRStorage = 179, // Comprehensive SR Storage
ProcedureLogStorage = 180, // Procedure Log Storage
MammographyCADSRStorage = 181, // Mammography CAD SR Storage
KeyObjectSelectionDocumentStorage = 182, // Key Object Selection Document Storage
ChestCADSRStorage = 183, // Chest CAD SR Storage
XRayRadiationDoseSRStorage = 184, // X-Ray Radiation Dose SR Storage
EncapsulatedPDFStorage = 185, // Encapsulated PDF Storage
EncapsulatedCDAStorage = 186, // Encapsulated CDA Storage
PositronEmissionTomographyImageStorage = 187, // Positron Emission Tomography Image Storage
StandalonePETCurveStorageRetired = 188, // Standalone PET Curve Storage
RTImageStorage = 189, // RT Image Storage
RTDoseStorage = 190, // RT Dose Storage
RTStructureSetStorage = 191, // RT Structure Set Storage
RTBeamsTreatmentRecordStorage = 192, // RT Beams Treatment Record Storage
RTPlanStorage = 193, // RT Plan Storage
RTBrachyTreatmentRecordStorage = 194, // RT Brachy Treatment Record Storage
RTTreatmentSummaryRecordStorage = 195, // RT Treatment Summary Record Storage
RTIonPlanStorage = 196, // RT Ion Plan Storage
RTIonBeamsTreatmentRecordStorage = 197, // RT Ion Beams Treatment Record Storage
PatientRootQueryRetrieveInformationModelFIND = 198, // Patient Root Query/Retrieve Information Model - FIND
PatientRootQueryRetrieveInformationModelMOVE = 199, // Patient Root Query/Retrieve Information Model - MOVE
PatientRootQueryRetrieveInformationModelGET = 200, // Patient Root Query/Retrieve Information Model - GET
StudyRootQueryRetrieveInformationModelFIND = 201, // Study Root Query/Retrieve Information Model - FIND
StudyRootQueryRetrieveInformationModelMOVE = 202, // Study Root Query/Retrieve Information Model - MOVE
StudyRootQueryRetrieveInformationModelGET = 203, // Study Root Query/Retrieve Information Model - GET
PatientStudyOnlyQueryRetrieveInformationModelFINDRetired = 204, // Patient/Study Only Query/Retrieve Information Model - FIND
PatientStudyOnlyQueryRetrieveInformationModelMOVERetired = 205, // Patient/Study Only Query/Retrieve Information Model - MOVE
PatientStudyOnlyQueryRetrieveInformationModelGETRetired = 206, // Patient/Study Only Query/Retrieve Information Model - GET
ModalityWorklistInformationModelFIND = 207, // Modality Worklist Information Model - FIND
GeneralPurposeWorklistInformationModelFIND = 208, // General Purpose Worklist Information Model - FIND
GeneralPurposeScheduledProcedureStepSOPClass = 209, // General Purpose Scheduled Procedure Step SOP Class
GeneralPurposePerformedProcedureStepSOPClass = 210, // General Purpose Performed Procedure Step SOP Class
GeneralPurposeWorklistManagementMetaSOPClass = 211, // General Purpose Worklist Management Meta SOP Class
InstanceAvailabilityNotificationSOPClass = 212, // Instance Availability Notification SOP Class
RTBeamsDeliveryInstructionStorageSupplement74FrozenDraft = 213, // RT Beams Delivery Instruction Storage (Supplement 74 Frozen Draft)
RTConventionalMachineVerificationSupplement74FrozenDraft = 214, // RT Conventional Machine Verification (Supplement 74 Frozen Draft)
RTIonMachineVerificationSupplement74FrozenDraft = 215, // RT Ion Machine Verification (Supplement 74 Frozen Draft)
UnifiedWorklistandProcedureStepServiceClass = 216, // Unified Worklist and Procedure Step Service Class
UnifiedProcedureStepPushSOPClass = 217, // Unified Procedure Step - Push SOP Class
UnifiedProcedureStepWatchSOPClass = 218, // Unified Procedure Step - Watch SOP Class
UnifiedProcedureStepPullSOPClass = 219, // Unified Procedure Step - Pull SOP Class
UnifiedProcedureStepEventSOPClass = 220, // Unified Procedure Step - Event SOP Class
UnifiedWorklistandProcedureStepSOPInstance = 221, // Unified Worklist and Procedure Step SOP Instance
GeneralRelevantPatientInformationQuery = 222, // General Relevant Patient Information Query
BreastImagingRelevantPatientInformationQuery = 223, // Breast Imaging Relevant Patient Information Query
CardiacRelevantPatientInformationQuery = 224, // Cardiac Relevant Patient Information Query
HangingProtocolStorage = 225, // Hanging Protocol Storage
HangingProtocolInformationModelFIND = 226, // Hanging Protocol Information Model - FIND
HangingProtocolInformationModelMOVE = 227, // Hanging Protocol Information Model - MOVE
ProductCharacteristicsQuerySOPClass = 228, // Product Characteristics Query SOP Class
SubstanceApprovalQuerySOPClass = 229, // Substance Approval Query SOP Class
dicomDeviceName = 230, // dicomDeviceName
dicomDescription = 231, // dicomDescription
dicomManufacturer = 232, // dicomManufacturer
dicomManufacturerModelName = 233, // dicomManufacturerModelName
dicomSoftwareVersion = 234, // dicomSoftwareVersion
dicomVendorData = 235, // dicomVendorData
dicomAETitle = 236, // dicomAETitle
dicomNetworkConnectionReference = 237, // dicomNetworkConnectionReference
dicomApplicationCluster = 238, // dicomApplicationCluster
dicomAssociationInitiator = 239, // dicomAssociationInitiator
dicomAssociationAcceptor = 240, // dicomAssociationAcceptor
dicomHostname = 241, // dicomHostname
dicomPort = 242, // dicomPort
dicomSOPClass = 243, // dicomSOPClass
dicomTransferRole = 244, // dicomTransferRole
dicomTransferSyntax = 245, // dicomTransferSyntax
dicomPrimaryDeviceType = 246, // dicomPrimaryDeviceType
dicomRelatedDeviceReference = 247, // dicomRelatedDeviceReference
dicomPreferredCalledAETitle = 248, // dicomPreferredCalledAETitle
dicomTLSCyphersuite = 249, // dicomTLSCyphersuite
dicomAuthorizedNodeCertificateReference = 250, // dicomAuthorizedNodeCertificateReference
dicomThisNodeCertificateReference = 251, // dicomThisNodeCertificateReference
dicomInstalled = 252, // dicomInstalled
dicomStationName = 253, // dicomStationName
dicomDeviceSerialNumber = 254, // dicomDeviceSerialNumber
dicomInstitutionName = 255, // dicomInstitutionName
dicomInstitutionAddress = 256, // dicomInstitutionAddress
dicomInstitutionDepartmentName = 257, // dicomInstitutionDepartmentName
dicomIssuerOfPatientID = 258, // dicomIssuerOfPatientID
dicomPreferredCallingAETitle = 259, // dicomPreferredCallingAETitle
dicomSupportedCharacterSet = 260, // dicomSupportedCharacterSet
dicomConfigurationRoot = 261, // dicomConfigurationRoot
dicomDevicesRoot = 262, // dicomDevicesRoot
dicomUniqueAETitlesRegistryRoot = 263, // dicomUniqueAETitlesRegistryRoot
dicomDevice = 264, // dicomDevice
dicomNetworkAE = 265, // dicomNetworkAE
dicomNetworkConnection = 266, // dicomNetworkConnection
dicomUniqueAETitle = 267, // dicomUniqueAETitle
dicomTransferCapability = 268, // dicomTransferCapability
//TalairachBrainAtlasFrameofReference = 269, // Talairach Brain Atlas Frame of Reference
//SPM2T1FrameofReference = 270, // SPM2 T1 Frame of Reference
//SPM2T2FrameofReference = 271, // SPM2 T2 Frame of Reference
//SPM2PDFrameofReference = 272, // SPM2 PD Frame of Reference
//SPM2EPIFrameofReference = 273, // SPM2 EPI Frame of Reference
//SPM2FILT1FrameofReference = 274, // SPM2 FIL T1 Frame of Reference
//SPM2PETFrameofReference = 275, // SPM2 PET Frame of Reference
//SPM2TRANSMFrameofReference = 276, // SPM2 TRANSM Frame of Reference
//SPM2SPECTFrameofReference = 277, // SPM2 SPECT Frame of Reference
//SPM2GRAYFrameofReference = 278, // SPM2 GRAY Frame of Reference
//SPM2WHITEFrameofReference = 279, // SPM2 WHITE Frame of Reference
//SPM2CSFFrameofReference = 280, // SPM2 CSF Frame of Reference
//SPM2BRAINMASKFrameofReference = 281, // SPM2 BRAINMASK Frame of Reference
//SPM2AVG305T1FrameofReference = 282, // SPM2 AVG305T1 Frame of Reference
//SPM2AVG152T1FrameofReference = 283, // SPM2 AVG152T1 Frame of Reference
//SPM2AVG152T2FrameofReference = 284, // SPM2 AVG152T2 Frame of Reference
//SPM2AVG152PDFrameofReference = 285, // SPM2 AVG152PD Frame of Reference
//SPM2SINGLESUBJT1FrameofReference = 286, // SPM2 SINGLESUBJT1 Frame of Reference
//ICBM452T1FrameofReference = 287, // ICBM 452 T1 Frame of Reference
//ICBMSingleSubjectMRIFrameofReference = 288, // ICBM Single Subject MRI Frame of Reference
VLWholeSlideMicroscopyImageStorage,
EnhancedUSVolumeStorage,
SurfaceSegmentationStorage,
BreastTomosynthesisImageStorage
} TSName;


  typedef const char* const (*TransferSyntaxStringsType)[2];
  static TransferSyntaxStringsType GetTransferSyntaxStrings();
  static const char * const *GetTransferSyntaxString(unsigned int ts);
  static unsigned int GetNumberOfTransferSyntaxStrings();


  // TODO: Because I would like a dual signature for TSType and TSName, C++ won't let me do it...
  static const char* GetUIDString(/*TSType*/ unsigned int ts);
  static const char* GetUIDName(/*TSType*/ unsigned int ts);

  /// Initialize object from a string (a uid number)
  /// return false on error, and internal state is set to 0
  bool SetFromUID(const char *str);

  /// When object is Initialize function return the well known name associated with uid
  /// return NULL when not initialized
  const char *GetName() const;

  /// When object is Initialize function return the uid
  /// return NULL when not initialized
  const char *GetString() const;

  operator TSType () const { return TSField; }

private:
  TSType TSField;
};
//-----------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &_os, const UIDs &uid)
{
  _os << uid.GetString() << " -> " << uid.GetName();
  return _os;

}

} // end namespace gdcm

#endif //GDCMUIDS_H
