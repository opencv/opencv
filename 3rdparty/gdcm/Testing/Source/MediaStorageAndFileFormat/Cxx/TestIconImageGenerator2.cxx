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

static const char * const iconimagearray2[][2] = {
 {"7e097b4a57af6a823bd692e37b131b1c" , "SIEMENS_SOMATOM-12-ACR_NEMA-ZeroLengthUs.acr" },
 {"7f2638cea652bf3452ada158d2ea67c4" , "gdcm-MR-SIEMENS-16-2.acr" },
 {"da0922a02f8e763ec878089617e0bc4c" , "test.acr" },
 {"e75ee90f23db1952a593393cdfaaf99f" , "MR-MONO2-12-an2.acr" },
 {"c9c46afd186531850af9576cf54c800b" , "CT-MONO2-12-lomb-an2.acr" },
 {"feb10d9a28e5166a3baf9ec6ef4f460a" , "LIBIDO-8-ACR_NEMA-Lena_128_128.acr" },
 {"f9d8a0796ebf5a6aabddb4e133d09770" , "gdcm-ACR-LibIDO.acr" },
 {"a1bce65e11c5fd2e26f7b38ebfdd1c53" , "libido1.0-vol.acr" },
 {"9edea7caea2b85aad952d35c64f1e092" , "SIEMENS_CSA2.dcm" },
 {"011d1f2abd9e27a2dc5d013bd4848104" , "gdcm-JPEG-LossLessThoravision.dcm" },
 {"ddb7e67d119eb2ce14731cd9007e36cc" , "XA-MONO2-8-12x-catheter.dcm" },
 {"1041bae356da40d2113210ff2adef923" , "gdcm-MR-PHILIPS-16-Multi-Seq.dcm" },
 {"a3a6b3bf75ccdc91f82effc60d005688" , "PHILIPS_GDCM12xBug.dcm" },
 {"a247e2ca32279956079c9a87403bd157" , "MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm" },
 {"48b19e53bffc8feab8671fd23c869028" , "D_CLUNIE_CT1_J2KI.dcm" },
 {"3f7cae9b920adb3ca4a96ace2c0c91d7" , "rle16sti.dcm" },
 {"f706f740496e445ee59141b15f3baeb4" , "3E768EB7.dcm" },
 {"b38c113fffa4925f6f06a928b357f6a1" , "D_CLUNIE_MR2_JPLY.dcm" },
 {"c3d3a218fcec778476090bce4c8b3201" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm" },
 {"882a7327bab05d88571dcf29f902adf3" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm" },
 {"7944f0f8eb06466099fe6cd792ae8bfa" , "MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm" },
 {"af193ff3143f299826d55e00696b3218" , "LEADTOOLS_FLOWERS-16-MONO2-Uncompressed.dcm" },
 {"b9b499634bdaff3be7710722f75d9b5d" , "D_CLUNIE_MR3_JPLY.dcm" },
 {"e8b529fbe615b4c540318695913d02e7" , "D_CLUNIE_VL2_RLE.dcm" },
 {"8393213bd2c340a8cbd639f45a6b497e" , "OsirixFake16BitsStoredFakeSpacing.dcm" },
 {"205a3ba253b61bf5ebb374523c366ccb" , "GE_RHAPSODE-16-MONO2-JPEG-Fragments.dcm" },
 {"c265bf6c3bf4417987ff81b92ca4ca6c" , "MR_SIEMENS_forceLoad29-1010_29-1020.dcm" },
 {"fa7da45d1a8eb16b092880ceff5a50e5" , "fffc0000UN.dcm" },
 {"796594769ee3570292de36fdb4509df1" , "LIBIDO-24-ACR_NEMA-Rectangle.dcm" },
 {"32b10fd4f13cfb4d74e1145f166e9dae" , "IM-0001-0066.CommandTag00.dcm" },
 {"619f8c1650962cddf7695e39f43e8c49" , "D_CLUNIE_NM1_JPLY.dcm" },
 {"205a3ba253b61bf5ebb374523c366ccb" , "D_CLUNIE_CT1_J2KR.dcm" },
 {"30a530ea4df623cbec31f752637e234b" , "TheralysGDCM120Bug.dcm" },
 {"3120b59e3635a912c6a60897f734b2ff" , "SIEMENS_MAGNETOM-12-MONO2-VRUN.dcm" },
 {"5bdb2335ceb2dac83c2248d6ff8e0a49" , "THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm" },
 {"30ff6655c5eb98ad34cd20836418a5a7" , "ALOKA_SSD-8-MONO2-RLE-SQ.dcm" },
 {"992e4ef2b01f9d79514d7ab22c354be9" , "GE_GENESIS-16-MONO2-WrongLengthItem.dcm" },
 {"d4669bd89971c1affd311c7d1d8b20c7" , "US-RGB-8-esopecho.dcm" },
 {"85124dd05ab0567aecaf896373f780da" , "D_CLUNIE_MR4_RLE.dcm" },
 {"07964ed19883cb96460d1407795f4306" , "D_CLUNIE_MR2_JPLL.dcm" },
 {"096547f2834609ae4dbc115169806fc4" , "SIEMENS_Sonata-16-MONO2-Value_Multiplicity.dcm" },
 {"af193ff3143f299826d55e00696b3218" , "LEADTOOLS_FLOWERS-16-MONO2-JpegLossless.dcm" },
 {"da7002947725a950c3226cd23aa8d718" , "CT-MONO2-16-ankle.dcm" },
 {"1a27a48d4efe2999b83bc89d3003d05c" , "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" },
 {"feb10d9a28e5166a3baf9ec6ef4f460a" , "simpleImageWithIcon.dcm" },
 {"aeb28d2cfb5376e2bb1ac4aac0b4807c" , "CR-MONO1-10-chest.dcm" },
 {"205a3ba253b61bf5ebb374523c366ccb" , "D_CLUNIE_CT1_RLE.dcm" },
 {"4d469f85f51b3b81bdf38ebba23a68e7" , "D_CLUNIE_NM1_JPLL.dcm" },
 {"8ffdae8852e9388369935d65b8afa408" , "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm" },
 {"abe1950da247d18196fe5bea54cb26f1" , "NM-MONO2-16-13x-heart.dcm" },
 {"6bd67bbd6e3e65808db8598fe0913f86" , "US-IRAD-NoPreambleStartWith0003.dcm" },
 {"99672e03104c9176595bc0002cd8edf8" , "MR-MONO2-16-head.dcm" },
 {"1a27a48d4efe2999b83bc89d3003d05c" , "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" },
 {"af8491953e762c12feb93a80a1575edf" , "MAROTECH_CT_JP2Lossy.dcm" },
 {"932281859be22ffae33a2669b34cc2e6" , "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm" },
 {"fb5402472ee00639ef8dc1b09692c299" , "US-PAL-8-10x-echo.dcm" },
 {"038acb53133733a9f2c54528b8029511" , "MR-SIEMENS-DICOM-WithOverlays.dcm" },
 {"890d0a5afdbf77398d857b523f86e83a" , "D_CLUNIE_MR1_JPLL.dcm" },
 {"205a3ba253b61bf5ebb374523c366ccb" , "D_CLUNIE_CT1_JPLL.dcm" },
 {"7e971371b523cc1db53f3fc7f2ac5441" , "AMIInvalidPrivateDefinedLengthSQasUN.dcm" },
 {"7944f0f8eb06466099fe6cd792ae8bfa" , "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm" },
 {"3443eba48f11dd9aadb4632c8c896b39" , "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" },
 {"b72e483ad977e18b8016c7cc916c4bbd" , "D_CLUNIE_SC1_JPLY.dcm" },
 {"582df1a47c5992cc4f37496dab8c5130" , "gdcm-MR-PHILIPS-16-NonSquarePixels.dcm" },
 {"8a5d13f6c85b2eebc1c02432d729cd45" , "SIEMENS-MR-RGB-16Bits.dcm" },
 {"aacd6f6257e068cf53cdec5c7e172513" , "US-IRAD-NoPreambleStartWith0005.dcm" },
 {"6a9ecd676ce47b0fda111a804b58b054" , "PICKER-16-MONO2-Nested_icon.dcm" },
 {"9b720fd0aa079bc2f3f9545bd0e207c6" , "gdcm-JPEG-Extended.dcm" },
 {"c790d0a462907135c1991f10a4846f98" , "D_CLUNIE_US1_RLE.dcm" },
 {"4689c053642dcee7a2a9d0475c1b6528" , "D_CLUNIE_RG2_JPLY.dcm" },
 {"b3b8a60e7b7b20c029eec1d5006e2b9c" , "PHILIPS_Brilliance_ExtraBytesInOverlay.dcm" },
 {"91533007e5e07353012dea33149d920f" , "MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm" },
 {"c8902cc14f285f2687866f60e45c7d86" , "TG18-CH-2k-01.dcm" },
 {"af193ff3143f299826d55e00696b3218" , "LEADTOOLS_FLOWERS-16-MONO2-RLE.dcm" },
 {"27daf49ec13f58db1d800fc58378852e" , "OT-PAL-8-face.dcm" },
 {"07fef244d4e14358d453c144770b2a55" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm" },
 {"ad96e04c3c50d463cccdfca68b49b071" , "SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm" },
 {"620f0b67a91f7f74151bc5be745b7110" , "GDCMJ2K_TextGBR.dcm" },
 {"d1f5dfac9c3213f548df6a1f9e86acd0" , "MR16BitsAllocated_8BitsStored.dcm" },
 {"dab7df42898362213062eb0d493eb38e" , "GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm" },
 {"205a3ba253b61bf5ebb374523c366ccb" , "D_CLUNIE_CT1_JLSL.dcm" },
 {"719505e71eafd643fa2e114acc83496f" , "ELSCINT1_JP2vsJ2K.dcm" },
 {"7132fec8839d9c8b26cba4a4f17000da" , "D_CLUNIE_XA1_JPLL.dcm" },
 {"cdf5908902fb63e321b224fae86a4128" , "SIEMENS_Sonata-12-MONO2-SQ.dcm" },
 {"bc1774acbf5c025aa3b57184213cc98c" , "JPEGDefinedLengthSequenceOfFragments.dcm" },
 {"e9faa2ce59db8563d7abe50432f91351" , "PHILIPS_GDCM12xBug2.dcm" },
 {"620f0b67a91f7f74151bc5be745b7110" , "ITK_GDCM124_MultiframeSecondaryCaptureInvalid.dcm" },
 {"52d0b90aa78a20cbff6b0beb3bd2c2b3" , "D_CLUNIE_RG2_JPLL.dcm" },
 {"50dd1ae0bf918a9144b50229fa04f8e0" , "FUJI-10-MONO1-ACR_NEMA_2.dcm" },
 {"d7e55bf3c6eb7a627823b532bcac7e16" , "D_CLUNIE_MR1_JPLY.dcm" },
 {"7132fec8839d9c8b26cba4a4f17000da" , "D_CLUNIE_XA1_RLE.dcm" },
 {"ddb5ac3f32badb154ce9a8a18c09c2c0" , "BugGDCM2_UndefItemWrongVL.dcm" },
 {"7f2638cea652bf3452ada158d2ea67c4" , "SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm" },
 {"477d801b27ef3cb21a8685cedbc9b12e" , "JDDICOM_Sample2.dcm" },
 {"cae77b420f407118f288c14eeb81e11f" , "TOSHIBA_MRT150-16-MONO2-ACR_NEMA_2.dcm" },
 {"524014ef866b31e3fd2e33a6148d31bb" , "DCMTK_JPEGExt_12Bits.dcm" },
 {"84ea2f46ad98332a06875da922709007" , "US-MONO2-8-8x-execho.dcm" },
 {"19d6f17c8a0b2cd04049828dcb304046" , "UnexpectedSequenceDelimiterInFixedLengthSequence.dcm" },
 {"bee28068ae9edef69fef792583b3e20f" , "ACUSON-24-YBR_FULL-RLE-b.dcm" },
 {"b171f7581c9cf15b2d32cfd62e9b6038" , "gdcm-US-ALOKA-16.dcm" },
 {"80be78c9365f6a1604191dd703c84ac5" , "DX_J2K_0Padding.dcm" },
 {"a9a72d44b888fb4f1f727af88109cfb3" , "D_CLUNIE_CT2_RLE.dcm" },
 {"151376fcac2fbd55c3b5c5a155e16d26" , "D_CLUNIE_RG1_JPLL.dcm" },
 {"e0a3fd5917c5a13a16981e1d0346af4d" , "D_CLUNIE_VL3_RLE.dcm" },
 {"28ea47a62e1f2ca9d7dcd10ef868701d" , "NM_Kakadu44_SOTmarkerincons.dcm" },
 {"e95cec3c8078cea91b287e45d751cf78" , "CT-SIEMENS-Icone-With-PaletteColor.dcm" },
 {"eacd1cabeaf78554804ca627344cba91" , "GE_DLX-8-MONO2-PrivateSyntax.dcm" },
 {"8f40712ff8e4a7691be144d884546a17" , "CT_16b_signed-UsedBits13.dcm" },
 {"da9fb19d564fa01e4633df5c105f5a58" , "D_CLUNIE_RG3_JPLY.dcm" },
 {"6510f44ff1dd8daf9b49100fd2c81d85" , "DX_GE_FALCON_SNOWY-VOI.dcm" },
 {"5a15bd50c589c9611636dc7813493c03" , "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" },
 {"4948235091edc01603344f79a29835e1" , "CT-MONO2-16-brain.dcm" },
 {"c2de60330cda14759a117b937a7e5a95" , "D_CLUNIE_VL4_RLE.dcm" },
 {"db07030ff68debba674a6ca5c3e6eeb3" , "D_CLUNIE_MR3_RLE.dcm" },
 {"72f55d5cf022e983783185195b77d0e6" , "undefined_length_un_vr.dcm" },
 {"a8e2e35044384f31afc1dcf9163154ff" , "CT-MONO2-16-ort.dcm" },
 {"0fefcf91a55062f1cb20945b3f726eda" , "05148044-mr-siemens-avanto-syngo.dcm" },
 {"5d6aaab5766479ec6e27de4eaa4a6438" , "GE_LOGIQBook-8-RGB-HugePreview.dcm" },
 {"acad4e43da617035315bcddc4bbb96bd" , "RadBWLossLess.dcm" },
 {"b69e23b6479de4d9c3b4ed51aa748f73" , "KODAK-12-MONO1-Odd_Terminated_Sequence.dcm" },
 {"e8458a5483e72cced4d46f93eed1699e" , "CT-MONO2-16-chest.dcm" },
 {"728a698e308f7863ba3d103b80dffc45" , "PhilipsInteraSeqTermInvLen.dcm" },
 {"35144f65c704edf2d69abc40405c3ea7" , "D_CLUNIE_MR4_JPLY.dcm" },
 {"904b38f362b86f626ea7d89eed455709" , "D_CLUNIE_RG3_JPLL.dcm" },
 {"707e26b0fa27f670c35ca5cc2bf3eda2" , "MR_Philips-Intera_BreaksNOSHADOW.dcm" },
 {"04b97df84da7a126be30993cf9cff942" , "GE_CT_With_Private_compressed-icon.dcm" },
 {"04cb0299587f38ba369e1fdcbf07dc25" , "D_CLUNIE_CT1_JLSN.dcm" },
 {"3120b59e3635a912c6a60897f734b2ff" , "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm" },
 {"cee47e57f6b1aaace74bb813e33a74eb" , "00191113.dcm" },
 {"99ba02caa6813cafd254f97fb8377f60" , "MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm" },
 {"0a9449186ab107120640910a58a38ec4" , "SignedShortLosslessBug.dcm" },
 {"2b306840879792f3509e1c1ccedee81d" , "GE_DLX-8-MONO2-Multiframe.dcm" },
 {"fde7dc022ec854a98ffdc2190ceef424" , "PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm" },
 {"f4b5fc5cc1440b865f1a99b85b170ecd" , "LJPEG_BuginGDCM12.dcm" },
 {"ba2717c0d839213e788a74cccf37f1d4" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm" },
 {"904b38f362b86f626ea7d89eed455709" , "D_CLUNIE_RG3_RLE.dcm" },
 {"5e51e64ee21c2f643c13257e054c2990" , "CT-SIEMENS-MissingPixelDataInIconSQ.dcm" },
 {"07964ed19883cb96460d1407795f4306" , "MR-MONO2-12-shoulder.dcm" },
 {"0b818ac3983d76cfdc535a46058a1a53" , "LEADTOOLS_FLOWERS-24-RGB-Uncompressed.dcm" },
 {"7f7aaee92f20ffcd64bebf8ce105bf5d" , "PICKER-16-MONO2-No_DicomV3_Preamble.dcm" },
 {"fde3b9c4853c597383fab1050b491202" , "SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm" },
 {"ba0654f3e7aae75bca8fc24705ddb78a" , "CT-MONO2-8-abdo.dcm" },
 {"db07030ff68debba674a6ca5c3e6eeb3" , "D_CLUNIE_MR3_JPLL.dcm" },
 {"03f25fb3e6e8ae53b9565553e2026a66" , "D_CLUNIE_VL6_RLE.dcm" },
 {"af193ff3143f299826d55e00696b3218" , "LEADTOOLS_FLOWERS-8-MONO2-RLE.dcm" },
 {"d46afca30969faa05583e87b0f6e5211" , "PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm" },
 {"5885ade0feaaf967f5a1c22724c10d02" , "SIEMENS-12-Jpeg_Process_2_4-Lossy-a.dcm" },
 {"13758e32fe34c17b897efcbd26e7e03b" , "gdcm-JPEG-LossLess3a.dcm" },
 {"1e61c6d50ae3ef3ac0550277f166bd1c" , "D_CLUNIE_XA1_JPLY.dcm" },
 {"151376fcac2fbd55c3b5c5a155e16d26" , "D_CLUNIE_RG1_RLE.dcm" },
 {"e77b93eb953f2a21fa75d257da0514fb" , "US-RGB-8-epicard.dcm" },
 {"a101ddd933114bcc0145d047d74f41c9" , "GE_MR_0025xx1bProtocolDataBlock.dcm" },
 {"b9f48f54e75b7f1994cfe1a7152d9ab5" , "rle16loo.dcm" },
 {"cc61470afd2b80014749abbb5bd9109d" , "DMCPACS_ExplicitImplicit_BogusIOP.dcm" },
 {"a9a72d44b888fb4f1f727af88109cfb3" , "D_CLUNIE_CT2_JPLL.dcm" },
 {"477d801b27ef3cb21a8685cedbc9b12e" , "JDDICOM_Sample2-dcmdjpeg.dcm" },
 {"7944f0f8eb06466099fe6cd792ae8bfa" , "MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm" },
 {"323c2919ea590363dd6eb4105a2566a7" , "KODAK_CompressedIcon.dcm" },
 {"486199008daf27f167efee9469fffd52" , "ACUSON-24-YBR_FULL-RLE.dcm" },
 {"07964ed19883cb96460d1407795f4306" , "D_CLUNIE_MR2_RLE.dcm" },
 {"b53c440c32a7bd20d24cc1997bd7c9e6" , "JPEG_LossyYBR.dcm" },
 {"23bf148e163ea7d3f1dbadfc590618fe" , "012345.002.050.dcm" },
 {"52d0b90aa78a20cbff6b0beb3bd2c2b3" , "D_CLUNIE_RG2_RLE.dcm" },
 {"e44869a1044b611e08a9f1396432a44b" , "MR_Philips_Intera_PrivateSequenceImplicitVR.dcm" },
 {"f92d6349e1d80bbdba34a9e1b84eb737" , "SIEMENS_ImageLocationUN.dcm" },
 {"7944f0f8eb06466099fe6cd792ae8bfa" , "PHILIPS_Intera-16-MONO2-Uncompress.dcm" },
 {"0e82ac528c72a2dd69fa1b52e5370d82" , "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" },
 {"0b818ac3983d76cfdc535a46058a1a53" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossless.dcm" },
 {"91b4da2ca9fd378ca404580c84c62984" , "US-GE-4AICL142.dcm" },
 {"2ff486489d600e0ab5f8c829960c516f" , "D_CLUNIE_SC1_RLE.dcm" },
 {"85124dd05ab0567aecaf896373f780da" , "D_CLUNIE_MR4_JPLL.dcm" },
 {"ece67fa0c8a99f8147b26081252cd0a5" , "MR-Brucker-CineTagging-NonSquarePixels.dcm" },
 {"bdf48b10871ac0b3c14587eaba9ca998" , "MR-MONO2-8-16x-heart.dcm" },
 {"0c65a403a0a24957e523bb29e84c31ef" , "LEADTOOLS_FLOWERS-8-MONO2-JpegLossy.dcm" },
 {"f92a60ad3cbc2783db57cac5f800a9c0" , "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm" },
 {"4d469f85f51b3b81bdf38ebba23a68e7" , "D_CLUNIE_NM1_RLE.dcm" },
 {"20e2f6cc2b60ae26adfdd3b3ee0e1915" , "D_CLUNIE_VL1_RLE.dcm" },
 {"84462b24809f4e9d151c7c27004f7922" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm" },
 {"620f0b67a91f7f74151bc5be745b7110" , "DermaColorLossLess.dcm" },
 {"72e203ca5756f4491d84d8240c82a59b" , "OT-MONO2-8-a7.dcm" },
 {"890d0a5afdbf77398d857b523f86e83a" , "D_CLUNIE_MR1_RLE.dcm" },
 {"af193ff3143f299826d55e00696b3218" , "LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm" },
 {"cee47e57f6b1aaace74bb813e33a74eb" , "00191113.dcm" },
 {"07b6f4e2fba920d28e2f29dbb037b640" , "TOSHIBA_J2K_SIZ1_PixRep0.dcm" },
 {"93f60acbb450c62992a1db09a6c19c05" , "TOSHIBA_J2K_OpenJPEGv2Regression.dcm" },
 {"fa02ad71bdf3e6970e18fc9a9df03a2d" , "TOSHIBA_J2K_SIZ0_PixRep1.dcm" },
 {"3475cb96e0308cb84502be1c1531b588" , "NM-PAL-16-PixRep1.dcm" },
 {"2ec6d7c60d6786665cff2b7a31aaeedf" , "MEDILABInvalidCP246_EVRLESQasUN.dcm" },
 {"54b1cec013420f771142ca6ec6c274a4" , "JPEGInvalidSecondFrag.dcm" },

 // sentinel
 { 0, 0 }
};

namespace gdcm
{
int TestIconImageGenerate2(const char *subdir, const char* filename, bool verbose = false)
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
  iig.AutoPixelMinMax(true);
  const unsigned int idims[2] = { 64, 64 };
  //const unsigned int idims[2] = { 600,430 };
  iig.SetOutputDimensions( idims );
  bool b = iig.Generate();

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();

  unsigned int i = 0;
  const char *p = iconimagearray2[i][1];
  while( p != 0 )
    {
    if( strcmp( name, p ) == 0 )
      {
      break;
      }
    ++i;
    p = iconimagearray2[i][1];
    }
  const char *refmd5 = iconimagearray2[i][0];

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

int TestIconImageGenerator2(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return gdcm::TestIconImageGenerate2(argv[0],filename, true);
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
    r += gdcm::TestIconImageGenerate2(argv[0], filename );
    ++i;
    }

  return r;
}
