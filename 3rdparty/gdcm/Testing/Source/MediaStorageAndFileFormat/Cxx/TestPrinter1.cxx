/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmReader.h"
#include "gdcmPrinter.h"
#include "gdcmFilename.h"
#include "gdcmTesting.h"

// the following list has been generated using gdcm, git: df760b9d8b3c9b280ad423153c649190f6e21204
// This correspond to the commit just before:
// BUG: an explicit length VR=SQ dataelement would not have been loaded as
// expected and a call to GetSequenceOfItems would fails. Thus downstream filter
// would fail load the SQ as expected. Introducing the more robust interface:
// GetValueAsSQ to solve that issue.
static const char * const printmd5[][2] = {
{ "a19bffac370df32acbf6b4991d1cbafe" , "00191113.dcm" } ,
{ "28f356915bc519137e2555db87e9914e" , "012345.002.050.dcm" } ,
{ "94f8c6ab090bdc11e61625bfc2dd39b7" , "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm" } ,
{ "20c11831c616eb121a405cd73de5cba2" , "05148044-mr-siemens-avanto-syngo.dcm" } ,
{ "ec10dcbf1b13ace8a6c0cc5b24c6c870" , "3E768EB7.dcm" } ,
{ "fde1da68a1707dcc687ddffd57e6b3c3" , "ACUSON-24-YBR_FULL-RLE-b.dcm" } ,
{ "a27c6628d6379783a3be223481d5cba4" , "ACUSON-24-YBR_FULL-RLE.dcm" } ,
{ "9f3aa114b955a812942f815b6b456eaf" , "ALOKA_SSD-8-MONO2-RLE-SQ.dcm" } ,
// MM: Check on Fri Jan 21 08:54:17 CET 2011, for AMIInvalidPrivateDefinedLengthSQasUN.dcm
{ "7e7c192969966d13823aff31482fd66e" , "AMIInvalidPrivateDefinedLengthSQasUN.dcm" } ,
{ "9bd2c821bdc3a53b5a6a09b581ef3b9c" , "BugGDCM2_UndefItemWrongVL.dcm" } ,
{ "6d2af85d2af299c223b684538d42d9e5" , "CR-MONO1-10-chest.dcm" } ,
{ "820d45cefd528e011921ea129bec9084" , "CT_16b_signed-UsedBits13.dcm" } ,
{ "fcf2ca019aa3138188edb18552983733" , "CT-MONO2-12-lomb-an2.acr" } ,
{ "0e4137f63819c706dfa370405d662c79" , "CT-MONO2-16-ankle.dcm" } ,
{ "f00a663066627e7ecdd1a7379137d396" , "CT-MONO2-16-brain.dcm" } ,
{ "87eadd3a3460c13593c9468bc533aa58" , "CT-MONO2-16-chest.dcm" } ,
{ "8b0efbe7cfd72a7d14cef44683fbacab" , "CT-MONO2-16-ort.dcm" } ,
{ "32c8e40e0726bbcc374f20771d37a312" , "CT-MONO2-8-abdo.dcm" } ,
{ "c5ffd15172c43b200e3654d5e3e45f3e" , "CT-SIEMENS-Icone-With-PaletteColor.dcm" } ,
{ "ca0e93a8995fccad9d031e0125ea1a29" , "CT-SIEMENS-MissingPixelDataInIconSQ.dcm" } ,
{ "a46ca467a0df5e1c71c8017b6b6768e7" , "D_CLUNIE_CT1_J2KI.dcm" } ,
{ "d2698545b07ad721faa71fdc1cf4a35d" , "D_CLUNIE_CT1_J2KR.dcm" } ,
{ "78f619a764a94a2c2bdc084d53ad2577" , "D_CLUNIE_CT1_JPLL.dcm" } ,
{ "4aaa4a602d7a1fd80c023acba36d6a84" , "D_CLUNIE_CT1_RLE.dcm" } ,
{ "9ee24f3a3291cd86f0bc78fddc3417e7" , "D_CLUNIE_CT2_JPLL.dcm" } ,
{ "b01b2b28aa3caa945b05865a9c440349" , "D_CLUNIE_CT2_RLE.dcm" } ,
{ "7d6531dc92c53ad25835b75fa5505b66" , "D_CLUNIE_MR1_JPLL.dcm" } ,
{ "e8a970a1c861192331a98162146ee0f9" , "D_CLUNIE_MR1_JPLY.dcm" } ,
{ "6f8b7b128c86c5a670a284e7eac38391" , "D_CLUNIE_MR1_RLE.dcm" } ,
{ "b770b1ca257d7c1c885f69a5a1b58808" , "D_CLUNIE_MR2_JPLL.dcm" } ,
{ "da74aefef4e8ac49dbe1a4dd3b1dcb26" , "D_CLUNIE_MR2_JPLY.dcm" } ,
{ "023eb2a9ecbfd3c04bb148ec339865f0" , "D_CLUNIE_MR2_RLE.dcm" } ,
{ "923d4103e1b063db6ed895fb67395750" , "D_CLUNIE_MR3_JPLL.dcm" } ,
{ "11b9cfc714d6f8eadc7b8b9c871f7b7d" , "D_CLUNIE_MR3_JPLY.dcm" } ,
{ "6773ea516bf525ed1278d23e402a4e78" , "D_CLUNIE_MR3_RLE.dcm" } ,
{ "1646ec013060ef3f261849db705f80f3" , "D_CLUNIE_MR4_JPLL.dcm" } ,
{ "fbbfe4caab2179641bd0de5d8ccf79ae" , "D_CLUNIE_MR4_JPLY.dcm" } ,
{ "fafdb8499580e59c2f6d9047894272ea" , "D_CLUNIE_MR4_RLE.dcm" } ,
{ "bc35454da5d8c0825236ccae83adcb82" , "D_CLUNIE_NM1_JPLL.dcm" } ,
{ "03527dbfa9b0b82fee7d6eb601abf470" , "D_CLUNIE_NM1_JPLY.dcm" } ,
{ "9488a473e2cd1fc981f36643e7dc1b82" , "D_CLUNIE_NM1_RLE.dcm" } ,
{ "d9da8798488b1c825dbd4ef940e386b9" , "D_CLUNIE_RG1_JPLL.dcm" } ,
{ "68a498853d26cce827eefb0f8be3581d" , "D_CLUNIE_RG1_RLE.dcm" } ,
{ "9f652b621320650c6be743b6bafcf539" , "D_CLUNIE_RG2_JPLL.dcm" } ,
{ "1f0e2edba097f4491b7d45fed604729d" , "D_CLUNIE_RG2_JPLY.dcm" } ,
{ "403e0e40507608cdcd61ee461de91226" , "D_CLUNIE_RG2_RLE.dcm" } ,
{ "73ecf348f874e7fb69f5dbfb4637979e" , "D_CLUNIE_RG3_JPLL.dcm" } ,
{ "b97b49d7f0bc3a3c3cb421292d8f13b5" , "D_CLUNIE_RG3_JPLY.dcm" } ,
{ "ec7dcf6857238547f0fb43f51317b05f" , "D_CLUNIE_RG3_RLE.dcm" } ,
{ "af35b9ad269b8a37df3d25389655272e" , "D_CLUNIE_SC1_JPLY.dcm" } ,
{ "50f92b02a370e2858632dfde330a2881" , "D_CLUNIE_SC1_RLE.dcm" } ,
{ "ad377bc5cc50cb1b6312e9035417b2f1" , "D_CLUNIE_US1_RLE.dcm" } ,
{ "e6ea16c1ba4a8e8f97383811fd91288a" , "D_CLUNIE_VL1_RLE.dcm" } ,
{ "291ac2ee3c48a76ae0ab9a095199997c" , "D_CLUNIE_VL2_RLE.dcm" } ,
{ "91c92c4d1330c283012d2bdd3a91dc2f" , "D_CLUNIE_VL3_RLE.dcm" } ,
{ "fcc44b14754a56879e0f3afc8161c1c0" , "D_CLUNIE_VL4_RLE.dcm" } ,
{ "d7ded36c6827f41d61790bed41b963e3" , "D_CLUNIE_VL6_RLE.dcm" } ,
{ "75801e9732698ea89740f935052f28d6" , "D_CLUNIE_XA1_JPLL.dcm" } ,
{ "79d8daf6837a4b5aefaa963b1fb88df1" , "D_CLUNIE_XA1_JPLY.dcm" } ,
{ "e2bda8297745ab7a77ca84242ad4c980" , "D_CLUNIE_XA1_RLE.dcm" } ,
{ "bb52d9f7a9116c87103066dd18a60e68" , "D_CLUNIE_CT1_JLSN.dcm" } ,
{ "8984d6319cd913d0f88fdb61989412f2" , "D_CLUNIE_CT1_JLSL.dcm" } ,
{ "4ef94d44a83066b80b39d254fb2a1c28" , "DCMTK_JPEGExt_12Bits.dcm" } ,
{ "d982dca9cc6c9db94a6233ca76e52f26" , "DermaColorLossLess.dcm" } ,
{ "ee354d9b221360982f15b630e1fdb046" , "DICOMDIR" } ,
{ "aa0556f64fb938935447f3e3b6b67b68" , "dicomdir_Acusson_WithPrivate_WithSR" } ,
{ "9934ab3c8adca82cad0fe8997b6c5cd3" , "DICOMDIR_MR_B_VA12A" } ,
{ "6a4cfd1ddd6eea5538dd7f8cf1ba1e1f" , "DICOMDIR-Philips-EasyVision-4200-Entries" } ,
{ "4254e4123245565e43a86e191acff01b" , "dicomdir_Pms_With_heavy_embedded_sequence" } ,
{ "9afc3c3e9d208292430b45867a9981e1" , "dicomdir_Pms_WithVisit_WithPrivate_WithStudyComponents" } ,
{ "356f30be965bbe4e335a56c6c5fe1928" , "dicomdir_With_embedded_icons" } ,
{ "d174a9f2dea0da9e013eb3180317ebbc" , "DMCPACS_ExplicitImplicit_BogusIOP.dcm" } ,
{ "0d2f1d2902b30e65f43f141057611525" , "DX_GE_FALCON_SNOWY-VOI.dcm" } ,
{ "1cbeb77ea2d6e0171dd38e0d6d5cb0b9" , "DX_J2K_0Padding.dcm" } ,
{ "3493bf0e698798529fde6ef488289879" , "ELSCINT1_JP2vsJ2K.dcm" } ,
{ "8f5581be656bd6f1ab6c9ec94f302284" , "ELSCINT1_LOSSLESS_RICE.dcm" } ,
{ "54f138a1aa6819ec1560a7ed344cde1a" , "ELSCINT1_PMSCT_RLE1.dcm" } ,
{ "177e832e1ccacfeb4c46fec80840cd62" , "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm" } ,
{ "3199cd21166043d619209d7a2073fb56" , "fffc0000UN.dcm" } ,
{ "9b426b02521cbc2f3ee25ab383ca835e" , "FUJI-10-MONO1-ACR_NEMA_2.dcm" } ,
{ "2aba3dacd00a0bc14a06e2d7142b916a" , "gdcm-ACR-LibIDO.acr" } ,
{ "b808c747b59f9d1a91a01491103abea5" , "gdcm-CR-DCMTK-16-NonSamplePerPix.dcm" } ,
{ "e7b5089a2a007221993ee5a7e6d44959" , "gdcm-JPEG-Extended.dcm" } ,
{ "20894653d750b4edf6034258a8dc3cf7" , "gdcm-JPEG-LossLess3a.dcm" } ,
{ "5011206e2a744c8a6f2cedb1ff7e5e11" , "gdcm-JPEG-LossLessThoravision.dcm" } ,
{ "49f3c524267607c09b6a78448f804eb8" , "gdcm-MR-PHILIPS-16-Multi-Seq.dcm" } ,
{ "3ada4145885084c465fc0d2969299428" , "gdcm-MR-PHILIPS-16-NonSquarePixels.dcm" } ,
{ "ac5f5bea451fd81aa2416fbab1a4e8c4" , "gdcm-MR-SIEMENS-16-2.acr" } ,
{ "48c49a7a41a7efea9ea0eadcd89ac9fa" , "gdcm-US-ALOKA-16.dcm" } ,
{ "9e9f42e825db2951519320c2e907d936" , "GE_CT_With_Private_compressed-icon.dcm" } ,
{ "9e126a24f81534e1cd653f16739a6192" , "GE_DLX-8-MONO2-Multiframe.dcm" } ,
{ "6b66f8c38fe96db805e7dedd9a997811" , "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm" } ,
{ "61ca6c5115e6f74565f6f2ca06647444" , "GE_DLX-8-MONO2-PrivateSyntax.dcm" } ,
{ "2ed065d325227d88abfaaa7b6b693b75" , "GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm" } ,
{ "8d398fce426d3a248d8c3f7582e3751d" , "GE_GENESIS-16-MONO2-WrongLengthItem.dcm" } ,
{ "bb5d500a0391a399b035968496b6fd5d" , "GE_LOGIQBook-8-RGB-HugePreview.dcm" } ,
{ "4baf83d23aa101ba8853193478764bda" , "GE_MR_0025xx1bProtocolDataBlock.dcm" } ,
{ "d5efa34d8091e1ad04683eefb41f33c7" , "GE_RHAPSODE-16-MONO2-JPEG-Fragments.dcm" } ,
{ "d41d8cd98f00b204e9800998ecf8427e" , "IM-0001-0066.dcm" } ,
{ "cd085d783924d8a7fa2270ff40c6dc3e" , "ITK_GDCM124_MultiframeSecondaryCaptureInvalid.dcm" } ,
{ "b47e43977843d8ceae0fcd957baf2692" , "JDDICOM_Sample2.dcm" } ,
{ "132cb5de391304c8a4f7653118d72d99" , "JDDICOM_Sample2-dcmdjpeg.dcm" } ,
{ "cf59e9bf2638e29400c32b1eb37582df" , "JPEGDefinedLengthSequenceOfFragments.dcm" } ,
{ "9c0548b6cc474c309686cfc3bdff7723" , "JPEG_LossyYBR.dcm" } ,
{ "aabb6942cee8e8d2ecbd8706e6c10a53" , "KODAK-12-MONO1-Odd_Terminated_Sequence.dcm" } ,
{ "492ba4d8a4b904a15a4f14fe35b31a16" , "KODAK_CompressedIcon.dcm" } ,
{ "9a085a611f192d6939b992be4e02fc8f" , "LEADTOOLS_FLOWERS-16-MONO2-JpegLossless.dcm" } ,
{ "4c884d7e06658c53275c932358e83f50" , "LEADTOOLS_FLOWERS-16-MONO2-RLE.dcm" } ,
{ "b879261a881099f66c6417487b5727e6" , "LEADTOOLS_FLOWERS-16-MONO2-Uncompressed.dcm" } ,
{ "792f30183573b0a228a8d372de1fd3e2" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossless.dcm" } ,
{ "9163d48f9b103792d7cc2401bd880efc" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm" } ,
{ "0b88e68d7629601e885db6126bdec0e5" , "LEADTOOLS_FLOWERS-24-RGB-Uncompressed.dcm" } ,
{ "3cbcc077573136a5cc96bd9cc4c6d42e" , "LEADTOOLS_FLOWERS-8-MONO2-JpegLossy.dcm" } ,
{ "a3074d046c5f382f62e6ee6b3f9561ea" , "LEADTOOLS_FLOWERS-8-MONO2-RLE.dcm" } ,
{ "b31da32c67ac695376cc928150be98d2" , "LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm" } ,
{ "ea9f4285ce92f9db2e349c7132fb849e" , "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" } ,
{ "9bac4c50cd8afce8f9995cefd542760f" , "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" } ,
{ "8fb0042d71596bb45e7387b0f0ed0d6b" , "libido1.0-vol.acr" } ,
{ "05fe8714421e16371d62ae2c280b7107" , "LIBIDO-16-ACR_NEMA-Volume.dcm" } ,
{ "251033db6d79bcfde7a8e17a814713a9" , "LIBIDO-24-ACR_NEMA-Rectangle.dcm" } ,
{ "d71a08e78c1d33c858836f6c1a0f2231" , "LIBIDO-8-ACR_NEMA-Lena_128_128.acr" } ,
{ "e39e4923be2d8bb2fb19c6c8deae216f" , "LJPEG_BuginGDCM12.dcm" } ,
{ "e6bc657d132abebb01a675ade04129f1" , "MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm" } ,
{ "174d5848257864f700c2e32d8441d0c5" , "MAROTECH_CT_JP2Lossy.dcm" } ,
{ "497d6ea45b8f3f7f6a3bf8125dcc43b1" , "MR16BitsAllocated_8BitsStored.dcm" } ,
{ "5b23ccf10ad6358b253a7ec185deb2a9" , "MR-Brucker-CineTagging-NonSquarePixels.dcm" } ,
{ "31246836410a24124acf6bea5a36a942" , "MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm" } ,
{ "c5042379519ac6b25df96b4ada96e9b1" , "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm" } ,
{ "8419613dfb8d4b190ef8a988f8927bce" , "MR-MONO2-12-an2.acr" } ,
{ "b4058b67ec1eb3d3d3acde27d51eb24a" , "MR-MONO2-12-angio-an1.acr" } ,
{ "f782f6ea25928310bd69c3ca5c6a97d2" , "MR-MONO2-12-shoulder.dcm" } ,
{ "7f6bccb00b34a7d277eacaffd2bb0362" , "MR-MONO2-16-head.dcm" } ,
{ "9bd4d79cc59c66b19c21577e12cd6226" , "MR-MONO2-8-16x-heart.dcm" } ,
{ "cd623ac04602182dc66adb505c516341" , "MR_Philips-Intera_BreaksNOSHADOW.dcm" } ,
{ "216e2c1f7f7f27e2a7d0bcefd49ab23c" , "MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm" } ,
{ "abce00bceef1711dca51e26950fc7a61" , "MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm" } ,
{ "1cad78e854ba2eea2d5fbe9c8c1bbcd4" , "MR_Philips_Intera_PrivateSequenceImplicitVR.dcm" } ,
{ "a812bbfd98e706b09ca6f9c9b06d16fc" , "MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm" } ,
{ "5797c9dfe94e4a4bccfbf81ab0aaa957" , "MR-SIEMENS-DICOM-WithOverlays.dcm" } ,
{ "df28467760104edc923d0625a0e2a778" , "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" } ,
{ "bf324a1c91d3a09d9136f54f5910e570" , "MR_SIEMENS_forceLoad29-1010_29-1020.dcm" } ,
{ "acbf80a7c610b44caebd52bc22de74f9" , "MR_Spectroscopy_SIEMENS_OF.dcm" } ,
{ "1a5646a7b05840813c067d8c8dfa507e" , "NM-MONO2-16-13x-heart.dcm" } ,
{ "8d8c3b8eb830b57b162e7c9f2d64b214" , "OsirixFake16BitsStoredFakeSpacing.dcm" } ,
{ "bf793beb0e96f2399e467409e3cf5311" , "OT-MONO2-8-a7.dcm" } ,
{ "7f4cddc9a88f8c5147b89f769eb1cae7" , "OT-PAL-8-face.dcm" } ,
{ "5fce2728bd9457d4fc1224a374829e2f" , "PET-cardio-Multiframe-Papyrus.dcm" } ,
{ "071b840050588d14fde61646e058e1c6" , "PHILIPS_Brilliance_ExtraBytesInOverlay.dcm" } ,
{ "308e32440ff9db54a01aa6aa7ed9d361" , "PHILIPS_GDCM12xBug2.dcm" } ,
{ "396d76f4143e544782ac23df3394b542" , "PHILIPS_GDCM12xBug.dcm" } ,
{ "c38ad661c6a14f8bd88b304755e7ba7b" , "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" } ,
{ "e0a690636c608f312ca550908d4b551f" , "PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm" } ,
{ "03d88060c3bd820b96840599c0099470" , "PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm" } ,
{ "211eee5dbca0899f762c788acc1fb3c2" , "PHILIPS_Intera-16-MONO2-Uncompress.dcm" } ,
{ "4f34474ed72e6a5960fc4691e588f8e0" , "PICKER-16-MONO2-Nested_icon.dcm" } ,
{ "629da04611e097e2cc532d6fe5e6454d" , "PICKER-16-MONO2-No_DicomV3_Preamble.dcm" } ,
{ "3aaca1826b4a4deb9e41ebf2af4fa6b2" , "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" } ,
{ "6dc48ff6f48d3c09db2fe877630ab003" , "RadBWLossLess.dcm" } ,
{ "c2f13224f3f0bc3aa8bbfa6f4a0a23ec" , "rle16loo.dcm" } ,
{ "bd5d9b2f994cfc0c6a95cca8c586533a" , "rle16sti.dcm" } ,
{ "631c5bb2e2f046215999072d13316363" , "SIEMENS-12-Jpeg_Process_2_4-Lossy-a.dcm" } ,
{ "593324f844b730f77bb4a337f51f3b3d" , "SIEMENS_CSA2.dcm" } ,
{ "fa919f3ee7cef3af1f362dd166d53103" , "SIEMENS_GBS_III-16-ACR_NEMA_1.acr" } ,
{ "825580733e8cfaf5d8679348889db40d" , "SIEMENS_GBS_III-16-ACR_NEMA_1-ULis2Bytes.dcm" } ,
{ "ff6bf3c70ef86ca08244317afd7fb98a" , "SIEMENS_ImageLocationUN.dcm" } ,
{ "ac5f5bea451fd81aa2416fbab1a4e8c4" , "SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm" } ,
{ "5202b5e2521b1a5b480317864081beae" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm" } ,
{ "8969371a42ecebc02f316a344a887c81" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm" } ,
{ "c32e8171b87f3825bf6a37c4ed4b627a" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm" } ,
{ "ea85da9fae20a585af6a71467916fa4a" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm" } ,
{ "1e22b1b3ef16b82cac4299279018282a" , "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm" } ,
{ "16f02a1ee88f8753978a3dad9a5968ac" , "SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm" } ,
{ "09fc7fb4d8d2fce59ad12f51da0bddfd" , "SIEMENS_MAGNETOM-12-MONO2-VRUN.dcm" } ,
{ "082dbd9bea567b86cc4415d66ff158e5" , "SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm" } ,
{ "be0b7098b442990aebf3364d747af482" , "SIEMENS-MR-RGB-16Bits.dcm" } ,
{ "a6469a8635a1b08b9642394e8dc6b350" , "SIEMENS_SOMATOM-12-ACR_NEMA-ZeroLengthUs.acr" } ,
{ "5fc4be2074b6186d3993818cd1baeb89" , "SIEMENS_Sonata-12-MONO2-SQ.dcm" } ,
{ "c56ae357244ed3d4203c2b28fe3ab447" , "SIEMENS_Sonata-16-MONO2-Value_Multiplicity.dcm" } ,
{ "7176eb4ec64dab63578eb6826318c87d" , "SignedShortLosslessBug.dcm" } ,
{ "ba40a5d86160fb17474759e7408c5326" , "simpleImageWithIcon.dcm" } ,
{ "cbd59e416dd82dc14df251ea6fccb55b" , "test.acr" } ,
{ "0c2c475f6d21ae0aeadbf16565dcdbc4" , "TG18-CH-2k-01.dcm" } ,
{ "e8cc7ed19eedf9bed9ab60683f3dbfa1" , "THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm" } ,
{ "8c80dcb91b204820e7d57ad0bdb2e945" , "TheralysGDCM120Bug.dcm" } ,
{ "e8819809884c214fe78ee2b227417e5c" , "TOSHIBA_MRT150-16-MONO2-ACR_NEMA_2.dcm" } ,
{ "408d556013117b9e60b610d97f41c211" , "undefined_length_un_vr.dcm" } ,
{ "787ca80c8dd1aa619d5f85610862380b" , "US-GE-4AICL142.dcm" } ,
{ "f1e9c893391e1d458cffa2a1e0904160" , "US-IRAD-NoPreambleStartWith0003.dcm" } ,
{ "d1d9be67c9d066fabc8e1c4ae124f9a0" , "US-IRAD-NoPreambleStartWith0005.dcm" } ,
{ "17222122d388f2c1185884c644c041db" , "US-MONO2-8-8x-execho.dcm" } ,
{ "d58a4930f5b436939d297730b883d5b9" , "US-PAL-8-10x-echo.dcm" } ,
{ "e0552f839d327506c32c4b9ceeb56459" , "US-RGB-8-epicard.dcm" } ,
{ "77bd6ff6dcac8e4fb763253e047dbdd2" , "US-RGB-8-esopecho.dcm" } ,
{ "7a3535f869f4a450b8de3d73a268e713" , "XA-MONO2-8-12x-catheter.dcm" } ,
{ "167af475c7e2f4605544fa1602c34d50" , "IM-0001-0066.CommandTag00.dcm" },
{ "d2cb6962750eb8f92c480e6cc2f4d104" , "GDCMJ2K_TextGBR.dcm" },
{ "2e039bbc7520f809963e051ff5144ccf" , "UnexpectedSequenceDelimiterInFixedLengthSequence.dcm" },
{ "0a90894370ba84dbe31acd1290ff9999" , "NM_Kakadu44_SOTmarkerincons.dcm" },
{ "504f0fae0f9e6e1ed2a03a5c96c0500f" , "PhilipsInteraSeqTermInvLen.dcm" },
{ "2940bd46f097f79012d24f47504c3c8c" , "TOSHIBA_J2K_OpenJPEGv2Regression.dcm" },
{ "696917fea41e83b9980bad82b609162c" , "TOSHIBA_J2K_SIZ0_PixRep1.dcm" },
{ "7ef3da46c43e51cfe2eb82e4d23dd623" , "TOSHIBA_J2K_SIZ1_PixRep0.dcm" },
{ "7f69568a362343fe7ff0266db7b3614f" , "NM-PAL-16-PixRep1.dcm" },
{ "8272b02db757629bf21b2cbea24e6e93" , "MEDILABInvalidCP246_EVRLESQasUN.dcm" },
{ "7483119e9d4facc0cfcc1fb0532fc4a0" , "JPEGInvalidSecondFrag.dcm" },
{ "4be7f56f11ce6090b7139879c4fdc71f" , "SC16BitsAllocated_8BitsStoredJPEG.dcm" },
{ "1060ff4eb8731d4b2152db83fed2eedd" , "SC16BitsAllocated_8BitsStoredJ2K.dcm" },

{ 0 ,0 }
};

int TestPrint(const char *filename, bool verbose= false)
{
  gdcm::Reader r;
  r.SetFileName( filename );
  if( !r.Read() )
    {
    std::cerr << "Could not read: " << filename << std::endl;
    return 1;
    }

  gdcm::Printer print;
  print.SetFile( r.GetFile() );
  std::ostringstream out;
  if( verbose )
    print.Print( std::cout );
  print.Print( out );

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();

  std::string buf = out.str();
  if( buf.find( "GDCM:UNKNOWN" ) != std::string::npos )
    {
    if( strcmp(name, "test.acr" ) != 0
      && strcmp(name, "LIBIDO-8-ACR_NEMA-Lena_128_128.acr" ) != 0
      && strcmp(name, "gdcm-ACR-LibIDO.acr" ) != 0
      && strcmp(name, "SIEMENS_GBS_III-16-ACR_NEMA_1.acr" ) != 0
      && strcmp(name, "LIBIDO-24-ACR_NEMA-Rectangle.dcm" ) != 0
      && strcmp(name, "NM_Kakadu44_SOTmarkerincons.dcm" ) != 0
    )
      {
      std::cerr << "UNKNOWN Attribute with : " << name << std::endl;
      return 1;
      }
    }
  char digest[33];
  gdcm::Testing::ComputeMD5(&buf[0], buf.size(), digest);

  unsigned int i = 0;
  const char *p = printmd5[i][1];
  while( p != 0 )
    {
    if( strcmp( name, p ) == 0 )
      {
      break;
      }
    ++i;
    p = printmd5[i][1];
    }

  const char *refmd5 = printmd5[i][0];
  if( !refmd5 )
    {
    std::cerr << "Problem with : " << name << " missing md5= " << digest << std::endl;
    return 1;
    }
  if( strcmp( refmd5, digest) )
    {
    std::cerr << "Problem with : " << name << " " << refmd5 << " vs " << digest << std::endl;
    return 1;
    }

  return 0;
}


int TestPrinter1(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestPrint(filename, true);
    }

  // else
  int r = 0, i = 0;
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestPrint( filename );
    ++i;
    }

  return r;
}
