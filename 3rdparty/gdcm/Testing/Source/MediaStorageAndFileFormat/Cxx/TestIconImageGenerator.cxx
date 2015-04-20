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

static const char * const iconimagearray[][2] = {
 {"f4f187737b9646348844804cd4eda259" , "SIEMENS_SOMATOM-12-ACR_NEMA-ZeroLengthUs.acr" },
 {"8659bcbae5479e03925ea51af0d42170" , "gdcm-MR-SIEMENS-16-2.acr" },
 {"f50225bbd7de605fa5b32d47ab4d0f19" , "test.acr" },
 {"890bd3ca8ab288128f520fd2a0c33539" , "MR-MONO2-12-an2.acr" },
 {"8893470f6cbe6314c66763b709177c7a" , "CT-MONO2-12-lomb-an2.acr" },
 {"2134b4b992ded28502f24ab3f21c6432" , "LIBIDO-8-ACR_NEMA-Lena_128_128.acr" },
 {"f9d8a0796ebf5a6aabddb4e133d09770" , "gdcm-ACR-LibIDO.acr" },
 {"68525a2313b5416fb6cf1f6b4fdb8d48" , "libido1.0-vol.acr" },
 {"a6453c0e0abbaad98edd3fd2d76f9d31" , "SIEMENS_CSA2.dcm" },
 {"84b514b71413607571b40dcc73a1e73b" , "gdcm-JPEG-LossLessThoravision.dcm" },
 {"c58f9cbfa1fe616278f963b48b33ee9f" , "XA-MONO2-8-12x-catheter.dcm" },
 {"d383b49843c5226c8f21cd5ee89b6da4" , "gdcm-MR-PHILIPS-16-Multi-Seq.dcm" },
 {"5fd844186b04820e1a55d42e99d81466" , "PHILIPS_GDCM12xBug.dcm" },
 {"5ec31742812dc9d3fd2a4eca8e8b8a48" , "MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm" },
 {"0132ce06e00b199446ebd14d957e6119" , "D_CLUNIE_CT1_J2KI.dcm" },
 {"3f7cae9b920adb3ca4a96ace2c0c91d7" , "rle16sti.dcm" },
 {"aa3c60bbe989c9f3ef5e59243e08af56" , "3E768EB7.dcm" },
 {"e3dae1f82b71857960bad4103524a72b" , "D_CLUNIE_MR2_JPLY.dcm" },
 {"5995fc49c087d64182a7dc2cabedb4b2" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm" },
 {"b7051f50189e469f6fc52d5108080b6d" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm" },
 {"17cd7bdeb852639b214377f0263a82d3" , "MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm" },
 {"365049df79c7491d14476c21083283ee" , "LEADTOOLS_FLOWERS-16-MONO2-Uncompressed.dcm" },
 {"2cc83519b459f49f22d37c56f0362c3a" , "D_CLUNIE_MR3_JPLY.dcm" },
 {"e8b529fbe615b4c540318695913d02e7" , "D_CLUNIE_VL2_RLE.dcm" },
 {"e223c80bb1ce7344559632777881fc98" , "OsirixFake16BitsStoredFakeSpacing.dcm" },
 {"e1b4e64a3a665d9ad74dc8a14cdc882b" , "GE_RHAPSODE-16-MONO2-JPEG-Fragments.dcm" },
 {"cb78acab107d772f3ef2353ffdc32360" , "MR_SIEMENS_forceLoad29-1010_29-1020.dcm" },
 {"4755c9b837968ca19b7a9882d4662c33" , "fffc0000UN.dcm" },
 {"796594769ee3570292de36fdb4509df1" , "LIBIDO-24-ACR_NEMA-Rectangle.dcm" },
 {"ffab5ebdfe60100e71295fe50412a98f" , "IM-0001-0066.CommandTag00.dcm" },
 {"4ce869f259e705d596bd459f7863c0ca" , "D_CLUNIE_NM1_JPLY.dcm" },
 {"e1b4e64a3a665d9ad74dc8a14cdc882b" , "D_CLUNIE_CT1_J2KR.dcm" },
 {"539914be101cd8a0fad88ea3e0827f59" , "TheralysGDCM120Bug.dcm" },
 {"20ac559d51fd5685236979d165178dd6" , "SIEMENS_MAGNETOM-12-MONO2-VRUN.dcm" },
 {"ed6e642ca59d3e90d7db08856330cf00" , "THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm" },
 {"209400d084e2c179c8c3d4973e75c66a" , "ALOKA_SSD-8-MONO2-RLE-SQ.dcm" },
 {"66bfea16c5837f976654e038b119af7a" , "GE_GENESIS-16-MONO2-WrongLengthItem.dcm" },
 {"d4669bd89971c1affd311c7d1d8b20c7" , "US-RGB-8-esopecho.dcm" },
 {"4b827f55ca36895f40bc47d27a0d7ea4" , "D_CLUNIE_MR4_RLE.dcm" },
 {"7d1c251e29dcd79af32c791a6f8e24e9" , "D_CLUNIE_MR2_JPLL.dcm" },
 {"5d642ed928e155551f3431f172e3378d" , "SIEMENS_Sonata-16-MONO2-Value_Multiplicity.dcm" },
 {"365049df79c7491d14476c21083283ee" , "LEADTOOLS_FLOWERS-16-MONO2-JpegLossless.dcm" },
 {"4305b70775ce53503ef93614815961e1" , "CT-MONO2-16-ankle.dcm" },
 {"1a27a48d4efe2999b83bc89d3003d05c" , "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" },
 {"2134b4b992ded28502f24ab3f21c6432" , "simpleImageWithIcon.dcm" },
 {"a13e29a4a0ca1a31338589a45f21a9e2" , "CR-MONO1-10-chest.dcm" },
 {"e1b4e64a3a665d9ad74dc8a14cdc882b" , "D_CLUNIE_CT1_RLE.dcm" },
 {"32931c4b9131f43496ab4a3518527af1" , "D_CLUNIE_NM1_JPLL.dcm" },
 {"efc58ca0ba4b51b511374f8e53f2af4e" , "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm" },
 {"620f0b67a91f7f74151bc5be745b7110" , "NM-MONO2-16-13x-heart.dcm" },
 {"9abf5e3aadcfe2b9c1cfe41782b36704" , "US-IRAD-NoPreambleStartWith0003.dcm" },
 {"f778aac834d5a68acaa97d8faec18e72" , "MR-MONO2-16-head.dcm" },
 {"1a27a48d4efe2999b83bc89d3003d05c" , "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" },
 {"09e18d103bc8c9cf08b1455af2717310" , "MAROTECH_CT_JP2Lossy.dcm" },
 {"bedb38b85f3bdb85c99b62e5b01a3ada" , "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm" },
 {"fb5402472ee00639ef8dc1b09692c299" , "US-PAL-8-10x-echo.dcm" },
 {"cea1ea10e8efc68bfb559bbefd1e4cc1" , "MR-SIEMENS-DICOM-WithOverlays.dcm" },
 {"6246c70687f0deab108c889eacda1b8e" , "D_CLUNIE_MR1_JPLL.dcm" },
 {"e1b4e64a3a665d9ad74dc8a14cdc882b" , "D_CLUNIE_CT1_JPLL.dcm" },
 {"db6293d8adc92a2273b6ae2fee13c2e7" , "AMIInvalidPrivateDefinedLengthSQasUN.dcm" },
 {"17cd7bdeb852639b214377f0263a82d3" , "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm" },
 {"b041b08aa1786195f3f26a75c1b84b85" , "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" },
 {"9172854202523aa6688367efcbfc3bdc" , "D_CLUNIE_SC1_JPLY.dcm" },
 {"d29e7ff251d7d754ce8b5368f61032b0" , "gdcm-MR-PHILIPS-16-NonSquarePixels.dcm" },
 {"8a5d13f6c85b2eebc1c02432d729cd45" , "SIEMENS-MR-RGB-16Bits.dcm" },
 {"3e8618fee9c8baca508eeef8206a340a" , "US-IRAD-NoPreambleStartWith0005.dcm" },
 {"3a7816dc8359a0d8a935313909fcf817" , "PICKER-16-MONO2-Nested_icon.dcm" },
 {"be5490cac843a7325027e578067d4eb3" , "gdcm-JPEG-Extended.dcm" },
 {"c790d0a462907135c1991f10a4846f98" , "D_CLUNIE_US1_RLE.dcm" },
 {"3ac268f0ec6f1005a72db89d182f143c" , "D_CLUNIE_RG2_JPLY.dcm" },
 {"5e848e7bc12f27e598c3348a586def24" , "PHILIPS_Brilliance_ExtraBytesInOverlay.dcm" },
 {"3bbe5ebe2ff7e260d56e40f670e3e335" , "MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm" },
 {"9df1b097f3f7eb4a3725065e361b9213" , "TG18-CH-2k-01.dcm" },
 {"365049df79c7491d14476c21083283ee" , "LEADTOOLS_FLOWERS-16-MONO2-RLE.dcm" },
 {"27daf49ec13f58db1d800fc58378852e" , "OT-PAL-8-face.dcm" },
 {"07fef244d4e14358d453c144770b2a55" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm" },
 {"0c38dfd851a74ea2205a0a6b69c2ddf7" , "SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm" },
 {"620f0b67a91f7f74151bc5be745b7110" , "GDCMJ2K_TextGBR.dcm" },
 {"5158679aaa379a50dec1e4a092d91455" , "MR16BitsAllocated_8BitsStored.dcm" },
 {"22103a2ee208d18f8c4e60dc15e5444a" , "GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm" },
 {"e1b4e64a3a665d9ad74dc8a14cdc882b" , "D_CLUNIE_CT1_JLSL.dcm" },
 {"719505e71eafd643fa2e114acc83496f" , "ELSCINT1_JP2vsJ2K.dcm" },
 {"09dd85a0f964f307f66f2b8727d22efb" , "D_CLUNIE_XA1_JPLL.dcm" },
 {"a5019730f663a81b50d559c8a98da4a6" , "SIEMENS_Sonata-12-MONO2-SQ.dcm" },
 {"ed0f5fc46b2e0bb6550d68686d956ca6" , "JPEGDefinedLengthSequenceOfFragments.dcm" },
 {"5aaee8a979cb11627e56ca5757e5d4e3" , "PHILIPS_GDCM12xBug2.dcm" },
 {"620f0b67a91f7f74151bc5be745b7110" , "ITK_GDCM124_MultiframeSecondaryCaptureInvalid.dcm" },
 {"e34c7c8d406ceec07ff1c8b82b59d2e5" , "D_CLUNIE_RG2_JPLL.dcm" },
 {"7478edd0202fde583138df8dfe2782f9" , "FUJI-10-MONO1-ACR_NEMA_2.dcm" },
 {"c5d8e44435241965c400935d30b0d654" , "D_CLUNIE_MR1_JPLY.dcm" },
 {"09dd85a0f964f307f66f2b8727d22efb" , "D_CLUNIE_XA1_RLE.dcm" },
 {"2783d3ac71623cd136de7eb8af13753b" , "BugGDCM2_UndefItemWrongVL.dcm" },
 {"8659bcbae5479e03925ea51af0d42170" , "SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm" },
 {"477d801b27ef3cb21a8685cedbc9b12e" , "JDDICOM_Sample2.dcm" },
 {"028dcdbc8e309eac282f73cff0751273" , "TOSHIBA_MRT150-16-MONO2-ACR_NEMA_2.dcm" },
 {"7e183442aefeadbfff9f0d79a7064a5d" , "DCMTK_JPEGExt_12Bits.dcm" },
 {"9f04afae98a960de78b31e416bdbe31b" , "US-MONO2-8-8x-execho.dcm" },
 {"7cc0d00a56ec460d4025d4f9cea4e445" , "UnexpectedSequenceDelimiterInFixedLengthSequence.dcm" },
 {"bee28068ae9edef69fef792583b3e20f" , "ACUSON-24-YBR_FULL-RLE-b.dcm" },
 {"b171f7581c9cf15b2d32cfd62e9b6038" , "gdcm-US-ALOKA-16.dcm" },
 {"b6ba84d4868436e0b44b19d75cefe239" , "DX_J2K_0Padding.dcm" },
 {"76b3e5225e1914e61010eb11c50b0d13" , "D_CLUNIE_CT2_RLE.dcm" },
 {"455aa24ef2085a6b57f8b29d4f42b558" , "D_CLUNIE_RG1_JPLL.dcm" },
 {"e0a3fd5917c5a13a16981e1d0346af4d" , "D_CLUNIE_VL3_RLE.dcm" },
 {"fd69a42d6ed9ac779fbdea5a874f27da" , "NM_Kakadu44_SOTmarkerincons.dcm" },
 {"5a622115e307a024aeb6003ad45c87d7" , "CT-SIEMENS-Icone-With-PaletteColor.dcm" },
 {"d687a5770da5430258c9354ef7283fb1" , "GE_DLX-8-MONO2-PrivateSyntax.dcm" },
 {"48843eac44f481073e2349a9b40034e6" , "CT_16b_signed-UsedBits13.dcm" },
 {"7b8f24a39d4d46d6898d0c868e2ead62" , "D_CLUNIE_RG3_JPLY.dcm" },
 {"a1b35156bf0be7f0fe5054e13b6dc11f" , "DX_GE_FALCON_SNOWY-VOI.dcm" },
 {"8745f035e70b8dad2a4b04b855c1974e" , "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" },
 {"608d7fd9e8cf15f95b75a12072a209d9" , "CT-MONO2-16-brain.dcm" },
 {"c2de60330cda14759a117b937a7e5a95" , "D_CLUNIE_VL4_RLE.dcm" },
 {"b4213e55f351e2c9e013050652c8fad4" , "D_CLUNIE_MR3_RLE.dcm" },
 {"cdae2ce2d07ef1880f9eba78e72c7f03" , "undefined_length_un_vr.dcm" },
 {"1e456d9eb136554a3d035d65999ac72f" , "CT-MONO2-16-ort.dcm" },
 {"7022cd07ae0e9311ca2bb13372becee1" , "05148044-mr-siemens-avanto-syngo.dcm" },
 {"5d6aaab5766479ec6e27de4eaa4a6438" , "GE_LOGIQBook-8-RGB-HugePreview.dcm" },
 {"d8b9952cb72ea315ae4b81038ca7498f" , "RadBWLossLess.dcm" },
 {"b7b5b0a05dde85a672529e896ffd34cc" , "KODAK-12-MONO1-Odd_Terminated_Sequence.dcm" },
 {"c1b31ebeb79ba29263e59254af17eb88" , "CT-MONO2-16-chest.dcm" },
 {"4085155ff0a36230f55dfd17c07b016f" , "PhilipsInteraSeqTermInvLen.dcm" },
 {"4c80f49834a2f8cec23ff8c7c0e80613" , "D_CLUNIE_MR4_JPLY.dcm" },
 {"dc04449fe2bdd8c5ff26f2b6b4f3a1aa" , "D_CLUNIE_RG3_JPLL.dcm" },
 {"01d9c3bac1a49f874eb70640d9187657" , "MR_Philips-Intera_BreaksNOSHADOW.dcm" },
 {"f86eb566e712fa57bf5bdd13af9233c6" , "GE_CT_With_Private_compressed-icon.dcm" },
 {"b2ed24f5bad1fe7fe1ceb62a1a37bf03" , "D_CLUNIE_CT1_JLSN.dcm" },
 {"20ac559d51fd5685236979d165178dd6" , "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm" },
 {"cee47e57f6b1aaace74bb813e33a74eb" , "00191113.dcm" },
 {"645a354b8bfbbda1c3e01a79c583eca0" , "MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm" },
 {"f8ef8fa89c0415c54ee8350db9bafc59" , "SignedShortLosslessBug.dcm" },
 {"9274ed6683e657a8e6c95278cb93a971" , "GE_DLX-8-MONO2-Multiframe.dcm" },
 {"cb7ccda2e8aed6d56df6350207c1d122" , "PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm" },
 {"788c5bf06000799bd630c70ce0418430" , "LJPEG_BuginGDCM12.dcm" },
 {"610b9ac0f1aac082eb1c0b140a85d2de" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm" },
 {"dc04449fe2bdd8c5ff26f2b6b4f3a1aa" , "D_CLUNIE_RG3_RLE.dcm" },
 {"587e56aa881d645ee67ac3afe317b11d" , "CT-SIEMENS-MissingPixelDataInIconSQ.dcm" },
 {"7d1c251e29dcd79af32c791a6f8e24e9" , "MR-MONO2-12-shoulder.dcm" },
 {"0b818ac3983d76cfdc535a46058a1a53" , "LEADTOOLS_FLOWERS-24-RGB-Uncompressed.dcm" },
 {"ba8fb87069401e40d9c0dd5bf0f2bb06" , "PICKER-16-MONO2-No_DicomV3_Preamble.dcm" },
 {"634fca1b1087cfb2dd4c74a567daf302" , "SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm" },
 {"ba0654f3e7aae75bca8fc24705ddb78a" , "CT-MONO2-8-abdo.dcm" },
 {"b4213e55f351e2c9e013050652c8fad4" , "D_CLUNIE_MR3_JPLL.dcm" },
 {"03f25fb3e6e8ae53b9565553e2026a66" , "D_CLUNIE_VL6_RLE.dcm" },
 {"365049df79c7491d14476c21083283ee" , "LEADTOOLS_FLOWERS-8-MONO2-RLE.dcm" },
 {"99b13adb3932f9fcf50c0aa936b6e367" , "PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm" },
 {"1f0795064ee1d19534780b7a4e92d5db" , "SIEMENS-12-Jpeg_Process_2_4-Lossy-a.dcm" },
 {"de42da38e961881e7a68f4d9337007cd" , "gdcm-JPEG-LossLess3a.dcm" },
 {"bec532ee78edb47bd0894dcbb2a7d1f5" , "D_CLUNIE_XA1_JPLY.dcm" },
 {"455aa24ef2085a6b57f8b29d4f42b558" , "D_CLUNIE_RG1_RLE.dcm" },
 {"e77b93eb953f2a21fa75d257da0514fb" , "US-RGB-8-epicard.dcm" },
 {"ddc56ce334439f64953f1b11c82aeeb9" , "GE_MR_0025xx1bProtocolDataBlock.dcm" },
 {"b9f48f54e75b7f1994cfe1a7152d9ab5" , "rle16loo.dcm" },
 {"50bf4b4c98228d2f11fb0d8c4b49e3cc" , "DMCPACS_ExplicitImplicit_BogusIOP.dcm" },
 {"76b3e5225e1914e61010eb11c50b0d13" , "D_CLUNIE_CT2_JPLL.dcm" },
 {"477d801b27ef3cb21a8685cedbc9b12e" , "JDDICOM_Sample2-dcmdjpeg.dcm" },
 {"17cd7bdeb852639b214377f0263a82d3" , "MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm" },
 {"19cd0ebffc7541a5ec944d25200923ad" , "KODAK_CompressedIcon.dcm" },
 {"486199008daf27f167efee9469fffd52" , "ACUSON-24-YBR_FULL-RLE.dcm" },
 {"7d1c251e29dcd79af32c791a6f8e24e9" , "D_CLUNIE_MR2_RLE.dcm" },
 {"b53c440c32a7bd20d24cc1997bd7c9e6" , "JPEG_LossyYBR.dcm" },
 {"4c54ea0d88336020167a5cb9437a1dec" , "012345.002.050.dcm" },
 {"e34c7c8d406ceec07ff1c8b82b59d2e5" , "D_CLUNIE_RG2_RLE.dcm" },
 {"b209bb00a5bb920ebfb52b50132bd3cb" , "MR_Philips_Intera_PrivateSequenceImplicitVR.dcm" },
 {"261065dc52f086beac57ed71977b40ec" , "SIEMENS_ImageLocationUN.dcm" },
 {"17cd7bdeb852639b214377f0263a82d3" , "PHILIPS_Intera-16-MONO2-Uncompress.dcm" },
 {"60e754c9dfc02bba3aee11e67653d844" , "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" },
 {"0b818ac3983d76cfdc535a46058a1a53" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossless.dcm" },
 {"91b4da2ca9fd378ca404580c84c62984" , "US-GE-4AICL142.dcm" },
 {"5b24d4a43d7c72090eabb88a06b56f15" , "D_CLUNIE_SC1_RLE.dcm" },
 {"4b827f55ca36895f40bc47d27a0d7ea4" , "D_CLUNIE_MR4_JPLL.dcm" },
 {"95ccb6a943e1ca82f3be78ceaa72b9d4" , "MR-Brucker-CineTagging-NonSquarePixels.dcm" },
 {"552dd953ebd0575d124e44df0218b0ea" , "MR-MONO2-8-16x-heart.dcm" },
 {"e73b404ca6d9b4b7ba54f46d2662deca" , "LEADTOOLS_FLOWERS-8-MONO2-JpegLossy.dcm" },
 {"8a35a388332dbb40018ee6814ab27994" , "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm" },
 {"32931c4b9131f43496ab4a3518527af1" , "D_CLUNIE_NM1_RLE.dcm" },
 {"20e2f6cc2b60ae26adfdd3b3ee0e1915" , "D_CLUNIE_VL1_RLE.dcm" },
 {"0b1cb469990edd0cabfa34c8b6cb427d" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm" },
 {"620f0b67a91f7f74151bc5be745b7110" , "DermaColorLossLess.dcm" },
 {"b9395dcfbb50e1af31147f80e8e0c1e7" , "OT-MONO2-8-a7.dcm" },
 {"6246c70687f0deab108c889eacda1b8e" , "D_CLUNIE_MR1_RLE.dcm" },
 {"365049df79c7491d14476c21083283ee" , "LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm" },
 {"cee47e57f6b1aaace74bb813e33a74eb" , "00191113.dcm" },
 {"07b6f4e2fba920d28e2f29dbb037b640" , "TOSHIBA_J2K_SIZ1_PixRep0.dcm" },
 {"206713a8ff161546d664d1285f99f90b" , "TOSHIBA_J2K_OpenJPEGv2Regression.dcm" },
 {"dc70ae52689ebf8f5003320e1c7f216b" , "TOSHIBA_J2K_SIZ0_PixRep1.dcm" },
 {"3475cb96e0308cb84502be1c1531b588" , "NM-PAL-16-PixRep1.dcm" },
 {"bcac830aae8652a88f4c4aa6d9e7e116" , "MEDILABInvalidCP246_EVRLESQasUN.dcm" },
 {"5b8db1c8d42870f68bfd908529c58710" , "JPEGInvalidSecondFrag.dcm" },
 {"37f43037574e53a0c27151443e27613f" , "SC16BitsAllocated_8BitsStoredJ2K.dcm" },
 {"620f0b67a91f7f74151bc5be745b7110" , "SC16BitsAllocated_8BitsStoredJPEG.dcm" },

 // sentinel
 { 0, 0 }
};

namespace gdcm
{
int TestIconImageGenerate(const char *subdir, const char* filename, bool verbose = false)
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
  const unsigned int idims[2] = { 64, 64 };
  iig.SetOutputDimensions( idims );
  bool b = iig.Generate();

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();

  unsigned int i = 0;
  const char *p = iconimagearray[i][1];
  while( p != 0 )
    {
    if( strcmp( name, p ) == 0 )
      {
      break;
      }
    ++i;
    p = iconimagearray[i][1];
    }
  const char *refmd5 = iconimagearray[i][0];

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

int TestIconImageGenerator(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return gdcm::TestIconImageGenerate(argv[0],filename, true);
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
    r += gdcm::TestIconImageGenerate(argv[0], filename );
    ++i;
    }

  return r;
}
