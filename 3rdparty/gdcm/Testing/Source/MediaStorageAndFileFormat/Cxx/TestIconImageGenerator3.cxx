/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageReader.h"
#include "gdcmImageWriter.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmTesting.h"
#include "gdcmByteSwap.h"
#include "gdcmIconImageGenerator.h"

static const char * const iconimagearray3[][2] = {
 {"18de6726bc64732057fb57ffa6f63482" , "SIEMENS_SOMATOM-12-ACR_NEMA-ZeroLengthUs.acr" },
 {"e751b44ccfa55354e08ba04fade91687" , "gdcm-MR-SIEMENS-16-2.acr" },
 {"a7ff57ec105e45b72d91ac750953befc" , "test.acr" },
 {"464cc3218d282f534326f2323264f8ca" , "MR-MONO2-12-an2.acr" },
 {"e23711964e5606b7cf8a9bf194e46a27" , "CT-MONO2-12-lomb-an2.acr" },
 {"fc5db4e2e7fca8445342b83799ff16d8" , "LIBIDO-8-ACR_NEMA-Lena_128_128.acr" },
 {"50ab781df5647feb681c6c24a1245d0d" , "gdcm-ACR-LibIDO.acr" },
 {"ce338fe6899778aacfc28414f2d9498b" , "libido1.0-vol.acr" },
 {"7acad6aa0274edfc797492128c2dd2bf" , "SIEMENS_CSA2.dcm" },
 {"231db66c0b1a60cab9e8e879711ebc16" , "gdcm-JPEG-LossLessThoravision.dcm" },
 {"353a62510cef13ce252f5e3a7ece54dc" , "XA-MONO2-8-12x-catheter.dcm" },
 {"b992b65f8beed6adc2224541be339d68" , "gdcm-MR-PHILIPS-16-Multi-Seq.dcm" },
 {"a9b91f5b18b1d7078ce597ec611a2c43" , "PHILIPS_GDCM12xBug.dcm" },
 {"a0b26e46dc8920167cd6b9b5ed57d8df" , "MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm" },
 {"eff839f187f2dfa07204b6550a96d4b6" , "D_CLUNIE_CT1_J2KI.dcm" },
 {"4f94be538667ad8d39abf42ee604bf7d" , "rle16sti.dcm" },
 {"7acd654cb0176b7b6f1982245593e8e6" , "3E768EB7.dcm" },
 {"6fe22e9a0983a07c5b1dde43373c36f3" , "D_CLUNIE_MR2_JPLY.dcm" },
 {"f17269f3d2447ed0b523b41a98b03129" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm" },
 {"311524ab5c9cb14812256c561c78e573" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm" },
 {"35619ea28ae211558d526e63429d5cbb" , "MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm" },
 {"a189258582a49855ae002d381016d02f" , "LEADTOOLS_FLOWERS-16-MONO2-Uncompressed.dcm" },
 {"b9291915007d8fae5c2a67a691c33236" , "D_CLUNIE_MR3_JPLY.dcm" },
 {"e2100ad66e70220df75ee60e9f69acdd" , "D_CLUNIE_VL2_RLE.dcm" },
 {"985dd349809ef7d63feeb511a63b05d1" , "OsirixFake16BitsStoredFakeSpacing.dcm" },
 {"046896d4baee6361e58217bd0a046162" , "GE_RHAPSODE-16-MONO2-JPEG-Fragments.dcm" },
 {"54307dbe02cab5ee23efa8360eb0cbd5" , "MR_SIEMENS_forceLoad29-1010_29-1020.dcm" },
 {"7e8bfa2560d883b45a0f868a81daac4d" , "fffc0000UN.dcm" },
 {"00f59e7e071f7fdf7d2e700df06a1d6b" , "LIBIDO-24-ACR_NEMA-Rectangle.dcm" },
 {"c1d31936337a3c65bba3edf67466a139" , "IM-0001-0066.CommandTag00.dcm" },
 {"6d95a1855031ac5401ec2c8ba7ff8b18" , "D_CLUNIE_NM1_JPLY.dcm" },
 {"046896d4baee6361e58217bd0a046162" , "D_CLUNIE_CT1_J2KR.dcm" },
 {"a439396c24ce353e108bd0308e8a80e2" , "TheralysGDCM120Bug.dcm" },
 {"e5155757353de1ad5c2b68c88fd34901" , "SIEMENS_MAGNETOM-12-MONO2-VRUN.dcm" },
 {"338d45d1ee3ec0202cdfa637c491669a" , "THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm" },
 {"5f90ef6a0621c658cfd2ae1f2bba351e" , "ALOKA_SSD-8-MONO2-RLE-SQ.dcm" },
 {"c881ef5c9a6a643ac7e890279ffc010f" , "GE_GENESIS-16-MONO2-WrongLengthItem.dcm" },
 {"13d97ddbcff7e14a89a6e5c337611881" , "US-RGB-8-esopecho.dcm" },
 {"0464a1b7160b097ea33f25865cce8fc9" , "D_CLUNIE_MR4_RLE.dcm" },
 {"3703d1695f0c02fa7a5cbe7142c8fc6f" , "D_CLUNIE_MR2_JPLL.dcm" },
 {"f0b59feaa0499a68038b627f2df71d93" , "SIEMENS_Sonata-16-MONO2-Value_Multiplicity.dcm" },
 {"a189258582a49855ae002d381016d02f" , "LEADTOOLS_FLOWERS-16-MONO2-JpegLossless.dcm" },
 {"dcce1e8c7b5cdc15d315807b4ec2d34b" , "CT-MONO2-16-ankle.dcm" },
 {"75c3cb93e002c788acc7eff4a93e2b5b" , "TOSHIBA_J2K_SIZ1_PixRep0.dcm" },
 {"4e8c0f3b4154f54ae04756d61fb1c2f5" , "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" },
 {"fc5db4e2e7fca8445342b83799ff16d8" , "simpleImageWithIcon.dcm" },
 {"1c582b9b521c398ed55776bcd615af3e" , "CR-MONO1-10-chest.dcm" },
 {"046896d4baee6361e58217bd0a046162" , "D_CLUNIE_CT1_RLE.dcm" },
 {"f00d34662aeb08d14bcaf230053bfd5f" , "D_CLUNIE_NM1_JPLL.dcm" },
 {"39faf30ee66e8c70dd2e772fcdbb1078" , "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm" },
 {"ce338fe6899778aacfc28414f2d9498b" , "NM-MONO2-16-13x-heart.dcm" },
 {"5a4101fb01068d1b48afcacc9bca9f7e" , "US-IRAD-NoPreambleStartWith0003.dcm" },
 {"3c18da99b3d3d53399d622aabdd96b64" , "MR-MONO2-16-head.dcm" },
 {"4e8c0f3b4154f54ae04756d61fb1c2f5" , "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" },
 {"443c20bd84b91d30037ad1da84568c6e" , "MAROTECH_CT_JP2Lossy.dcm" },
 {"488e407541a6c353f0540049ba36d61c" , "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm" },
 {"50057b97e7c585ba31144867cd40c363" , "US-PAL-8-10x-echo.dcm" },
 {"e189b6b3a44a8d929f654105f9842707" , "MR-SIEMENS-DICOM-WithOverlays.dcm" },
 {"139f2ba3c6f578f883125d090163e730" , "D_CLUNIE_MR1_JPLL.dcm" },
 {"046896d4baee6361e58217bd0a046162" , "D_CLUNIE_CT1_JPLL.dcm" },
 {"bbc89a7b6e41d668f22b6e612cfb13e7" , "AMIInvalidPrivateDefinedLengthSQasUN.dcm" },
 {"35619ea28ae211558d526e63429d5cbb" , "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm" },
 {"8b2eca7c4d23a79a9e6d9437bff66d67" , "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" },
 {"4ef7021ed6522a730693574a37759940" , "D_CLUNIE_SC1_JPLY.dcm" },
 {"ce338fe6899778aacfc28414f2d9498b" , "gdcm-MR-PHILIPS-16-NonSquarePixels.dcm" },
 {"b92f9dcf1acd6dd998539d22a5296f74" , "SIEMENS-MR-RGB-16Bits.dcm" },
 {"07c92897d0cbd052060f8a9d13a6386a" , "US-IRAD-NoPreambleStartWith0005.dcm" },
 {"64b228a9c05e2f1b4a6371abf1ac7e21" , "PICKER-16-MONO2-Nested_icon.dcm" },
 {"41d6ffef9d2edbad51c94a8d476bb1ab" , "gdcm-JPEG-Extended.dcm" },
 {"5d8f3a1415c53f2b8b6ea459b4f3f6c1" , "D_CLUNIE_US1_RLE.dcm" },
 {"d0dcc34bad0aa031269d3101fa044e4b" , "D_CLUNIE_RG2_JPLY.dcm" },
 {"4a56065c6ab80bc0a32b39a143e58d5c" , "PHILIPS_Brilliance_ExtraBytesInOverlay.dcm" },
 {"2888e58bfe4d11b2671f2cd6a35f9b7c" , "MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm" },
 {"00d2f91bbf1f1d5c248f98220f85a860" , "TG18-CH-2k-01.dcm" },
 {"a189258582a49855ae002d381016d02f" , "LEADTOOLS_FLOWERS-16-MONO2-RLE.dcm" },
 {"703d03c5fb65c9493775c7ed6a9aa001" , "OT-PAL-8-face.dcm" },
 {"12919f48891b07e74651efd5a2a52676" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm" },
 {"94e0346dc7594b042c6e67a0f8ebfa71" , "SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm" },
 {"06148d96de1833da4518b6d97fdd4165" , "GDCMJ2K_TextGBR.dcm" },
 {"59f3e40cbdd708b43dd818a953bd5ca0" , "MR16BitsAllocated_8BitsStored.dcm" },
 {"fe4ee9d2fc8f068c7e5b61e8dc90811d" , "GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm" },
 {"046896d4baee6361e58217bd0a046162" , "D_CLUNIE_CT1_JLSL.dcm" },
 {"9f04eeafdcd85d84cc6f8edad761e3a1" , "ELSCINT1_JP2vsJ2K.dcm" },
 {"531bfd1059c6396e9da4c9a4f5436e86" , "D_CLUNIE_XA1_JPLL.dcm" },
 {"12ea2d93daec9fda8f26666dbaf21c30" , "SIEMENS_Sonata-12-MONO2-SQ.dcm" },
 {"acf0d5ea13eb09415ffe3ec2d0800590" , "JPEGDefinedLengthSequenceOfFragments.dcm" },
 {"6b9e6ee648acb3ca92814bc6642d0b96" , "PHILIPS_GDCM12xBug2.dcm" },
 {"ce338fe6899778aacfc28414f2d9498b" , "ITK_GDCM124_MultiframeSecondaryCaptureInvalid.dcm" },
 {"941378c3e000b34264342b1d46ce6dce" , "D_CLUNIE_RG2_JPLL.dcm" },
 {"9efc87377ac56dd01476046bbbd5bf7b" , "FUJI-10-MONO1-ACR_NEMA_2.dcm" },
 {"99b4254ecb0b4075b8005b775b02bea1" , "D_CLUNIE_MR1_JPLY.dcm" },
 {"531bfd1059c6396e9da4c9a4f5436e86" , "D_CLUNIE_XA1_RLE.dcm" },
 {"8058ebaf622a0108d12a387c75de5aae" , "BugGDCM2_UndefItemWrongVL.dcm" },
 {"e751b44ccfa55354e08ba04fade91687" , "SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm" },
 {"c860f1795d0be70b310f5b653b34f06f" , "JDDICOM_Sample2.dcm" },
 {"5fcd3432874ea0f3cbc59a317534a0b0" , "TOSHIBA_MRT150-16-MONO2-ACR_NEMA_2.dcm" },
 {"ce338fe6899778aacfc28414f2d9498b" , "DCMTK_JPEGExt_12Bits.dcm" },
 {"7abf9227edf8f7106198b320c5daa538" , "US-MONO2-8-8x-execho.dcm" },
 {"16fff764b95a60d83e4b6781da292cea" , "UnexpectedSequenceDelimiterInFixedLengthSequence.dcm" },
 {"d600407a5f79393f294518be3bba7fed" , "ACUSON-24-YBR_FULL-RLE-b.dcm" },
 {"eb1c07547d036c3f778700cfef1ad24c" , "gdcm-US-ALOKA-16.dcm" },
 {"9805da917d916632cf3a22b6b9140bce" , "DX_J2K_0Padding.dcm" },
 {"4d6a3590ae425105315a0995f8a7da41" , "D_CLUNIE_CT2_RLE.dcm" },
 {"f1b14732b392da7ea5fba9b47225945c" , "D_CLUNIE_RG1_JPLL.dcm" },
 {"86362d24addeeaaa8b1b4e485ff49ea7" , "D_CLUNIE_VL3_RLE.dcm" },
 {"5d590fa042089bbd62f771e403ad3068" , "NM_Kakadu44_SOTmarkerincons.dcm" },
 {"790f684146906a7f0b9d8c3838066579" , "CT-SIEMENS-Icone-With-PaletteColor.dcm" },
 {"336e557af65d34ae7cb5dcc18582b12b" , "GE_DLX-8-MONO2-PrivateSyntax.dcm" },
 {"848ee1315da8c58064a685e70378dae5" , "CT_16b_signed-UsedBits13.dcm" },
 {"29db994be059e339ab881eb760878a54" , "D_CLUNIE_RG3_JPLY.dcm" },
 {"d1bd49bdf8ce55652f7802f8f8820f97" , "DX_GE_FALCON_SNOWY-VOI.dcm" },
 {"7dcb5cdba4bfc242dbcc0d275ad14970" , "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" },
 {"d66fddd6bc26af655f500cd37de3b135" , "CT-MONO2-16-brain.dcm" },
 {"351d2c3befec588fe7798ae0884908e7" , "D_CLUNIE_VL4_RLE.dcm" },
 {"97e8c927ca201a2f932b5b3ecc6453e5" , "D_CLUNIE_MR3_RLE.dcm" },
 {"bbb1be6461a9c8b1a0fd507b5c19ec20" , "undefined_length_un_vr.dcm" },
 {"ae8fe4406652ca3e6b93d2ade029d94e" , "CT-MONO2-16-ort.dcm" },
 {"ab976b85e44c2332d826067520f6c2e6" , "05148044-mr-siemens-avanto-syngo.dcm" },
 {"8a9c001007c43fff520fada2b75e1e72" , "GE_LOGIQBook-8-RGB-HugePreview.dcm" },
 {"3808ac31fbf6ede5d0a1ca47ff604f8d" , "RadBWLossLess.dcm" },
 {"0ceae99a40ae6ee2ad5cf1fa06a86820" , "KODAK-12-MONO1-Odd_Terminated_Sequence.dcm" },
 {"e8dd9e61c1eb37c4ab54003ef9fd2842" , "CT-MONO2-16-chest.dcm" },
 {"63808ab2b6faa36b6914e01907358eb7" , "PhilipsInteraSeqTermInvLen.dcm" },
 {"3a1f7960887aa18084a6f7347926c8b5" , "D_CLUNIE_MR4_JPLY.dcm" },
 {"91c2703f4a5dec2d780539c29b450d47" , "D_CLUNIE_RG3_JPLL.dcm" },
 {"96f9a1224838a0695b3d2cc8b0b26b97" , "MR_Philips-Intera_BreaksNOSHADOW.dcm" },
 {"c75d2a8866ccdb2a96f7634bcc81adef" , "GE_CT_With_Private_compressed-icon.dcm" },
 {"4c600037ef23dbf2da360db9f8d7e28b" , "TOSHIBA_J2K_OpenJPEGv2Regression.dcm" },
 {"6c004779f7e43673883f04742aae9541" , "D_CLUNIE_CT1_JLSN.dcm" },
 {"e5155757353de1ad5c2b68c88fd34901" , "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm" },
 {"afccf775abd15a6c053c7f134b204762" , "00191113.dcm" },
 {"9670e5f8c163087d75cc4044356359c3" , "MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm" },
 {"ac1abc22d5d52b21c74c9f34b6178cd1" , "SignedShortLosslessBug.dcm" },
 {"0f5efe869ef6b9846707e89fbef18f2c" , "GE_DLX-8-MONO2-Multiframe.dcm" },
 {"6763fff05fa5b5dc2b842a40573b18a3" , "PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm" },
 {"10151c9ee27d40cfd58f0d79e693f38e" , "LJPEG_BuginGDCM12.dcm" },
 {"d1779fd58242dc231aa021cbd9687b04" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm" },
 {"91c2703f4a5dec2d780539c29b450d47" , "D_CLUNIE_RG3_RLE.dcm" },
 {"ab0815dcb84bd5a42972134ea9c0493d" , "CT-SIEMENS-MissingPixelDataInIconSQ.dcm" },
 {"3703d1695f0c02fa7a5cbe7142c8fc6f" , "MR-MONO2-12-shoulder.dcm" },
 {"5a898d46c28350c4f55b9915b9001852" , "LEADTOOLS_FLOWERS-24-RGB-Uncompressed.dcm" },
 {"654e4af3861f475c16bf498efc5df861" , "PICKER-16-MONO2-No_DicomV3_Preamble.dcm" },
 {"885cd1e48a980257534d3c88380ec259" , "SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm" },
 {"dbe025b7a22b829898b6eb2b35580171" , "CT-MONO2-8-abdo.dcm" },
 {"97e8c927ca201a2f932b5b3ecc6453e5" , "D_CLUNIE_MR3_JPLL.dcm" },
 {"5cffc4c262df7534f4008aa49aaef3ae" , "D_CLUNIE_VL6_RLE.dcm" },
 {"a189258582a49855ae002d381016d02f" , "LEADTOOLS_FLOWERS-8-MONO2-RLE.dcm" },
 {"051d73b41bb387005e1dbcc3af7253b3" , "PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm" },
 {"e20e59c88037bbe676e8c67bee2a9540" , "TOSHIBA_J2K_SIZ0_PixRep1.dcm" },
 {"ca3a2e9d6d2813df183a9fd9c9d2e9a1" , "SIEMENS-12-Jpeg_Process_2_4-Lossy-a.dcm" },
 {"f073eb11257017c6422158aa5cfb372c" , "gdcm-JPEG-LossLess3a.dcm" },
 {"9b2dfd69af52cd8625fe1faa73f51493" , "D_CLUNIE_XA1_JPLY.dcm" },
 {"f1b14732b392da7ea5fba9b47225945c" , "D_CLUNIE_RG1_RLE.dcm" },
 {"8b86324f5b7b88f06dffe975a06f1544" , "US-RGB-8-epicard.dcm" },
 {"5a478d37f436a6687a07fd43aaef79ee" , "GE_MR_0025xx1bProtocolDataBlock.dcm" },
 {"43d75fbe29085c5760d6ff4c083a9726" , "rle16loo.dcm" },
 {"cbebb7b25f11a1c39194566b194e84e5" , "DMCPACS_ExplicitImplicit_BogusIOP.dcm" },
 {"4d6a3590ae425105315a0995f8a7da41" , "D_CLUNIE_CT2_JPLL.dcm" },
 {"c860f1795d0be70b310f5b653b34f06f" , "JDDICOM_Sample2-dcmdjpeg.dcm" },
 {"35619ea28ae211558d526e63429d5cbb" , "MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm" },
 {"2f2f0a482fc0c1b8d99db02a57122b22" , "KODAK_CompressedIcon.dcm" },
 {"1b28808b244d86f4b6ec626b4d538680" , "ACUSON-24-YBR_FULL-RLE.dcm" },
 {"3703d1695f0c02fa7a5cbe7142c8fc6f" , "D_CLUNIE_MR2_RLE.dcm" },
 {"22a67114d4ae8e036665e1a1a508ca8d" , "JPEG_LossyYBR.dcm" },
 {"3646c27f6fb60d66ab916934d331d550" , "012345.002.050.dcm" },
 {"941378c3e000b34264342b1d46ce6dce" , "D_CLUNIE_RG2_RLE.dcm" },
 {"0ba3c7de165ffe7fb43733f25de2fdf1" , "MR_Philips_Intera_PrivateSequenceImplicitVR.dcm" },
 {"e344c9a95328db382caf8a7b86c91992" , "SIEMENS_ImageLocationUN.dcm" },
 {"35619ea28ae211558d526e63429d5cbb" , "PHILIPS_Intera-16-MONO2-Uncompress.dcm" },
 {"42aa4fa1ac9421f083927baaf7230cf6" , "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" },
 {"5a898d46c28350c4f55b9915b9001852" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossless.dcm" },
 {"00e15542d31e48490442eb4c336103e8" , "US-GE-4AICL142.dcm" },
 {"2bacf42b57e957883e35451c5ff26f50" , "D_CLUNIE_SC1_RLE.dcm" },
 {"0464a1b7160b097ea33f25865cce8fc9" , "D_CLUNIE_MR4_JPLL.dcm" },
 {"9b801c57776e5f7c4b45f2c6c47795d8" , "MR-Brucker-CineTagging-NonSquarePixels.dcm" },
 {"373a42fc21d411b4c060d2cf2a4e85f4" , "MR-MONO2-8-16x-heart.dcm" },
 {"6e9d26218dce3f54b41efbe442a2154b" , "LEADTOOLS_FLOWERS-8-MONO2-JpegLossy.dcm" },
 {"ccda9f1170c12f443d83990e9777ca8e" , "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm" },
 {"f00d34662aeb08d14bcaf230053bfd5f" , "D_CLUNIE_NM1_RLE.dcm" },
 {"a2b00a80955b1f368dcff51264b9e2e7" , "D_CLUNIE_VL1_RLE.dcm" },
 {"f00ef30b32dd57515c2c0d5c921c6ee1" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm" },
 {"0638ee027ccf94fd7ecca641b6fd402c" , "DermaColorLossLess.dcm" },
 {"5340e9111251343c6a932992241b299e" , "OT-MONO2-8-a7.dcm" },
 {"139f2ba3c6f578f883125d090163e730" , "D_CLUNIE_MR1_RLE.dcm" },
 {"a189258582a49855ae002d381016d02f" , "LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm" },
 {"afccf775abd15a6c053c7f134b204762" , "00191113.dcm" },
 {"f4f35d60b3cc18aaa6d8d92f0cd3708a" , "NM-PAL-16-PixRep1.dcm" },
 {"843f097f2161014cd1b11be5413eace9" , "MEDILABInvalidCP246_EVRLESQasUN.dcm" },
 {"c103d405ae61aa4ed148e17157e5afc1" , "JPEGInvalidSecondFrag.dcm" },

 // sentinel
 { 0, 0 }
};

namespace gdcm
{
int TestIconImageGenerate3(const char *subdir, const char* filename, bool verbose = false)
{
  ImageReader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    const FileMetaInformation &header = reader.GetFile().GetHeader();
    MediaStorage ms = header.GetMediaStorage();
    bool isImage = MediaStorage::IsImage( ms );
    bool pixeldata = reader.GetFile().GetDataSet().FindDataElement( Tag(0x7fe0,0x0010) );
    if( isImage && pixeldata )
      {
      std::cerr << "Failed to read: " << filename << std::endl;
      return 1;
      }
    else
      {
      // not an image give up...
      std::cerr << "Problem with: " << filename << " but that's ok" << std::endl;
      return 0;
      }
    }

  // Create directory first:
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );

  IconImageGenerator iig;
  iig.SetPixmap( reader.GetImage() );
  iig.ConvertRGBToPaletteColor( false );
  const unsigned int idims[2] = { 128, 128};
  //const unsigned int idims[2] = { 552,421 };
  iig.SetOutputDimensions( idims );
  bool b = iig.Generate();

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();

  unsigned int i = 0;
  const char *p = iconimagearray3[i][1];
  while( p != 0 )
    {
    if( strcmp( name, p ) == 0 )
      {
      break;
      }
    ++i;
    p = iconimagearray3[i][1];
    }
  const char *refmd5 = iconimagearray3[i][0];

  if( b )
    {
    const gdcm::IconImage &icon = iig.GetIconImage();
    if( verbose ) icon.Print( std::cout );
    unsigned long len = icon.GetBufferLength();
    std::vector< char > vbuffer;
    vbuffer.resize( len );
    char *buffer = &vbuffer[0];
    bool res2 = icon.GetBuffer(buffer);
    if( !res2 )
      {
      std::cerr << "res2 failure:" << filename << std::endl;
      return 1;
      }
    char digest[33];
    gdcm::Testing::ComputeMD5(buffer, len, digest);
    Image & img = reader.GetImage();
    img.SetIconImage( iig.GetIconImage() );

    ImageWriter writer;
    writer.SetFileName( outfilename.c_str() );
#if 1
    writer.SetImage( img );
#else
    Image &ii = writer.GetImage();
    (Bitmap&)ii = iig.GetIconImage();
#endif
    if( !writer.Write() )
      {
      std::cerr << "Failed to write: " << outfilename << std::endl;
      return 1;
      }

    if( verbose )
      {
    std::cout << "success: " << outfilename << std::endl;
      std::cout << "ref=" << refmd5 << std::endl;
      std::cout << "md5=" << digest << std::endl;
      }

    if( !refmd5 )
      {
      std::cerr << " missing md5= {\"" << digest << "\" , \"" << name << "\" }," << std::endl;
      return 1;
      }
    if( strcmp( refmd5, digest) )
      {
      std::cerr << "Problem with : " << name << " " << refmd5 << " vs " << digest << std::endl;
      return 1;
      }

    }
  else
    {
    assert( refmd5 == 0 );
    std::cerr << "Could not generate Icon for: " << filename << std::endl;
    return 1;
    }

  return 0;
}
}

int TestIconImageGenerator3(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return gdcm::TestIconImageGenerate3(argv[0],filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += gdcm::TestIconImageGenerate3(argv[0], filename );
    ++i;
    }

  return r;
}
