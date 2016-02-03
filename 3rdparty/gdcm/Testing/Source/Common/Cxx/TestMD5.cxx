/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmMD5.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmTesting.h"

#include <iostream>
#include <cstring> // strcmp

static const char * const gdcmMD5SumFiles[][2] = {
{ "f5ae1418f6ec07ae13522c18ff1e067a"  , "00191113.dcm" },
{ "c2fdbb35ba2a179939a8608e1320c7ac"  , "012345.002.050.dcm" },
{ "4b8bed2f8da2fa6a260764e62eb3731b"  , "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm" },
{ "a01a44c67a88d5a98ecdc214e412c585"  , "05148044-mr-siemens-avanto-syngo.dcm" },
{ "7ddb7cc42cf3191a248a60f66661b380"  , "3E768EB7.dcm" },
{ "a0452c3bb9303ad8ebc7f8a166cc8190"  , "ACUSON-24-YBR_FULL-RLE-b.dcm" },
{ "916b12bdf36d60a0b0ae39451d86ddb4"  , "ACUSON-24-YBR_FULL-RLE.dcm" },
{ "a4d1faf3a3a4c8b0cdf8605ef0ab48b5"  , "ALOKA_SSD-8-MONO2-RLE-SQ.dcm" },
{ "00c78123ea2d8a9f1452f0da37383d85"  , "BugGDCM2_UndefItemWrongVL.dcm" },
{ "0b4cfdaebfa3944940bed290c0a29a5b"  , "CR-MONO1-10-chest.dcm" },
{ "d4f286ee295ac03856370e4ff0fe60a9"  , "CT_16b_signed-UsedBits13.dcm" },
{ "0e03332d250a8a1a5d3c0699b1cf6b1d"  , "CT-MONO2-12-lomb-an2.acr" },
{ "102528014fa6ff9758c76745090b7550"  , "CT-MONO2-16-ankle.dcm" },
{ "a21649de46e5f5609ef6e227999a1dc8"  , "CT-MONO2-16-brain.dcm" },
{ "56caa0050cd61dee0bfb41559dbd6a33"  , "CT-MONO2-16-chest.dcm" },
{ "5e8efa7ebe3a4c47c01a64f80b244b2c"  , "CT-MONO2-16-ort.dcm" },
{ "abe6a2e0b1a1d636b6a775291b44b7d0"  , "CT-MONO2-8-abdo.dcm" },
{ "442e1966eba2fd0eb8fab685c99cc388"  , "CT-SIEMENS-Icone-With-PaletteColor.dcm" },
{ "7dbc75bccddcb7177796c1fa3509ab09"  , "CT-SIEMENS-MissingPixelDataInIconSQ.dcm" },
{ "6155e751818ede1db17aa848d882beda"  , "D_CLUNIE_CT1_J2KI.dcm" },
{ "7e59d70ed013175815a6c8a498d481f1"  , "D_CLUNIE_CT1_J2KR.dcm" },
{ "073960e6c249b45bd6341e3397637478"  , "D_CLUNIE_CT1_JPLL.dcm" },
{ "53d722b71a881e386005ec3c83bd78a1"  , "D_CLUNIE_CT1_RLE.dcm" },
{ "8577a89b3906c01731641e177dda0f88"  , "D_CLUNIE_CT2_JPLL.dcm" },
{ "681c50b2a496cb5c2d7bb964108649d2"  , "D_CLUNIE_CT2_RLE.dcm" },
{ "64f39d9d812d999b689cb9dad069ed3a"  , "D_CLUNIE_MR1_JPLL.dcm" },
{ "c14fd84b399f305ddd71a547ecbaaa13"  , "D_CLUNIE_MR1_JPLY.dcm" },
{ "45346d785b6c3a8be12b271abb745b13"  , "D_CLUNIE_MR1_RLE.dcm" },
{ "82093a4fc15b10b791b3d884a1cdc9c6"  , "D_CLUNIE_MR2_JPLL.dcm" },
{ "43867bf684f5c8eb56e7595c76e02e7a"  , "D_CLUNIE_MR2_JPLY.dcm" },
{ "072779b40d39e704ec4c002226bdfceb"  , "D_CLUNIE_MR2_RLE.dcm" },
{ "60852d347647758175d8f65d2c81e7a6"  , "D_CLUNIE_MR3_JPLL.dcm" },
{ "8e60bb51feba032aa141385f48465ef8"  , "D_CLUNIE_MR3_JPLY.dcm" },
{ "cf1e84527b2a6095037a8c2cfe801335"  , "D_CLUNIE_MR3_RLE.dcm" },
{ "caa51957b182d1048f1077af1c3d99fb"  , "D_CLUNIE_MR4_JPLL.dcm" },
{ "810033c798a3168f051389e0a0a8bcb7"  , "D_CLUNIE_MR4_JPLY.dcm" },
{ "af6b015679b368bb0e066874730a5cb9"  , "D_CLUNIE_MR4_RLE.dcm" },
{ "9a1008f0b45537614523e1fc8dce1b37"  , "D_CLUNIE_NM1_JPLL.dcm" },
{ "dbd2aedf74ca7297ea213d7d19934bba"  , "D_CLUNIE_NM1_JPLY.dcm" },
{ "01b95bc55f4684d0d7f2cb990a4c870f"  , "D_CLUNIE_NM1_RLE.dcm" },
{ "510e107dc1ca40a577af88a20f71a0b4"  , "D_CLUNIE_RG1_JPLL.dcm" },
{ "418bef1f3e70cb02205999bce74fed8e"  , "D_CLUNIE_RG1_RLE.dcm" },
{ "7596cda97de4782951827104f616f7ea"  , "D_CLUNIE_RG2_JPLL.dcm" },
{ "f03d1b8dc233c8c902c9fb4800ec6fe2"  , "D_CLUNIE_RG2_JPLY.dcm" },
{ "2fbc6e918c2d4ad7115addf46d870e78"  , "D_CLUNIE_RG2_RLE.dcm" },
{ "9966b6dbec23248fb2b6da9e7769ba20"  , "D_CLUNIE_RG3_JPLL.dcm" },
{ "d06de3e3b941dce6c2c2c896fe42584b"  , "D_CLUNIE_RG3_JPLY.dcm" },
{ "4ed3c3920c7763b883c234bf8bf95662"  , "D_CLUNIE_RG3_RLE.dcm" },
{ "41dc2884a4f98d484da80c2bdb7e5fcc"  , "D_CLUNIE_SC1_JPLY.dcm" },
{ "ec344a1d113d5070cabebf17c0fa3330"  , "D_CLUNIE_SC1_RLE.dcm" },
{ "14f4e54540008a39b53725800962f703"  , "D_CLUNIE_US1_RLE.dcm" },
{ "44a550cd8bccba4f157dd825489a5e87"  , "D_CLUNIE_VL1_RLE.dcm" },
{ "f4274cbe83a3368d99ee8332421448fe"  , "D_CLUNIE_VL2_RLE.dcm" },
{ "ea3d6a0fe2a65244c285c31eac5066ab"  , "D_CLUNIE_VL3_RLE.dcm" },
{ "6452ff527ac71f780445cf5ff083b0af"  , "D_CLUNIE_VL4_RLE.dcm" },
{ "53be8cfa8cb9709c08732908bb37410a"  , "D_CLUNIE_VL6_RLE.dcm" },
{ "9178b1d061641464ecaf271fc4328948"  , "D_CLUNIE_XA1_JPLL.dcm" },
{ "1f050eec29d9eeed3d406570ea1b9168"  , "D_CLUNIE_XA1_JPLY.dcm" },
{ "99040c48b80d40d6f7d6eed6a3cbc824"  , "D_CLUNIE_XA1_RLE.dcm" },
{ "9cf394a4dde294fc740e7577529ba5ca"  , "D_CLUNIE_CT1_JLSL.dcm" },
{ "b4d14dc9e820e6f1cf17730833a0373a"  , "D_CLUNIE_CT1_JLSN.dcm" },
{ "de20088d529a3bb211933c2d3b7604aa"  , "DCMTK_JPEGExt_12Bits.dcm" },
{ "e959b1f056d40bb4b21d0cbff1f67310"  , "DermaColorLossLess.dcm" },
{ "dee54ccedfed2d4d5562b52a1a7a5cfc"  , "DICOMDIR" },
{ "81ba5ff6f512289c30efdc757c6de231"  , "dicomdir_Acusson_WithPrivate_WithSR" },
{ "7c75c6a8957298bd70bf9b51efd39da1"  , "DICOMDIR_MR_B_VA12A" },
{ "7339affc644067bbd4be9134b597c515"  , "DICOMDIR-Philips-EasyVision-4200-Entries" },
{ "de270e5b601d5f9e235ea651932b546c"  , "dicomdir_Pms_With_heavy_embedded_sequence" },
{ "1ef216d2a08420432172581e0b4b9ffa"  , "dicomdir_Pms_WithVisit_WithPrivate_WithStudyComponents" },
{ "19c0a3e4ba48de7b35f19013bf7ceaa7"  , "dicomdir_With_embedded_icons" },
{ "33e6ad84c695d0d2c8be46ae79cfb5be"  , "DMCPACS_ExplicitImplicit_BogusIOP.dcm" },
{ "e692180e794a05a284edc985728af12b"  , "DX_GE_FALCON_SNOWY-VOI.dcm" },
{ "4838485dda77a0ff1a51463e1742f1c1"  , "DX_J2K_0Padding.dcm" },
{ "16f951766461b7cf1ec318f53c396c3b"  , "ELSCINT1_JP2vsJ2K.dcm" },
{ "e1e3c870656147b186361ab2439379ae"  , "ELSCINT1_LOSSLESS_RICE.dcm" },
{ "c5a985457952a33a10c90139c5812e56"  , "ELSCINT1_PMSCT_RLE1.dcm" },
{ "f2429f2bf3d2d7951a358e94a848977b"  , "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm" },
{ "b92edb08527b520df5f624f6753fb47a"  , "fffc0000UN.dcm" },
{ "40cff8d72a15be2b91d3c972b5c8a481"  , "FUJI-10-MONO1-ACR_NEMA_2.dcm" },
{ "74fc5c1975c8c8277a2900a803e56260"  , "gdcm-ACR-LibIDO.acr" },
{ "bd4c340a9d225c16f1507dbc9013ff47"  , "gdcm-CR-DCMTK-16-NonSamplePerPix.dcm" },
{ "04528010e8679cd668ee3e89cb9b7058"  , "gdcm-JPEG-Extended.dcm" },
{ "95953bc725582368ac5ddac7eb73e73f"  , "gdcm-JPEG-LossLess3a.dcm" },
{ "f715e4b9f640ff3b3b98b87d10f81c36"  , "gdcm-JPEG-LossLessThoravision.dcm" },
{ "06fc6d07dc486b2cb87077ea74774220"  , "gdcm-MR-PHILIPS-16-Multi-Seq.dcm" },
{ "4b09787d27dbd36a02a0ae7d4d3abe16"  , "gdcm-MR-PHILIPS-16-NonSquarePixels.dcm" },
{ "765d1cae48ebac57f675193734f6477c"  , "gdcm-MR-SIEMENS-16-2.acr" },
{ "20e289bad9be67e22d179174f3bc1694"  , "gdcm-US-ALOKA-16.dcm" },
{ "64e2f50c3e3b9b7400f3f29e144b4a34"  , "GE_CT_With_Private_compressed-icon.dcm" },
{ "012993a90206a659e92f6af305889f0b"  , "GE_DLX-8-MONO2-Multiframe.dcm" },
{ "1bec471d81dcb3a39852c03d261c22cd"  , "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm" },
{ "44f09f83983a2b45a6cd42deb1cbcf0a"  , "GE_DLX-8-MONO2-PrivateSyntax.dcm" },
{ "1ef5dca6cda8c21b01ff779e6aa67692"  , "GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm" },
{ "34483fff69d88d647a2edfd8d55210a3"  , "GE_GENESIS-16-MONO2-WrongLengthItem.dcm" },
{ "1a7c56cb02d6e742cc9c856a8ac182e3"  , "GE_LOGIQBook-8-RGB-HugePreview.dcm" },
{ "7285d6a7261889c33626310e1d581ef8"  , "GE_MR_0025xx1bProtocolDataBlock.dcm" },
{ "309d063fbe3bd73dad534e72de032f97"  , "GE_RHAPSODE-16-MONO2-JPEG-Fragments.dcm" },
{ "76d57d019a5af8cd5a7cd2afb1e40f4b"  , "IM-0001-0066.dcm" },
{ "b479bb01798444128d75d90d37cf8546"  , "ITK_GDCM124_MultiframeSecondaryCaptureInvalid.dcm" },
{ "1bb147010022b15e021eabe0eae1a231"  , "JDDICOM_Sample2.dcm" },
{ "cd9afab2d9d31de0029bf4ed1995186c"  , "JDDICOM_Sample2-dcmdjpeg.dcm" },
{ "e4b43fa2fdb4dde13e2a7fd018323241"  , "JPEG_LossyYBR.dcm" },
{ "118dc6986862bf76326ba542813049d2"  , "KODAK-12-MONO1-Odd_Terminated_Sequence.dcm" },
{ "499661b964e8df08860655c8dcc17661"  , "KODAK_CompressedIcon.dcm" },
{ "21de4aa50000b4ed74e4531c2b1d0cc1"  , "LEADTOOLS_FLOWERS-16-MONO2-JpegLossless.dcm" },
{ "042ca0b7551bd96b501fbbdd4275342f"  , "LEADTOOLS_FLOWERS-16-MONO2-RLE.dcm" },
{ "31c7e4c1a2f39871b886c443c6376ba7"  , "LEADTOOLS_FLOWERS-16-MONO2-Uncompressed.dcm" },
{ "9f884f686020c37be9f41a617b9ec8e8"  , "LEADTOOLS_FLOWERS-24-RGB-JpegLossless.dcm" },
{ "bc38dd9c27150dd3d67250830644e609"  , "LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm" },
{ "32ad312f99c425b1631e6e05881e3e33"  , "LEADTOOLS_FLOWERS-24-RGB-Uncompressed.dcm" },
{ "82c2343f6f4b55bf6a31f6cc0f9cf83e"  , "LEADTOOLS_FLOWERS-8-MONO2-JpegLossy.dcm" },
{ "f78abbd1df9ef87ba21d805083d6e3b3"  , "LEADTOOLS_FLOWERS-8-MONO2-RLE.dcm" },
{ "afe156b36b3af19b6a889d640296c710"  , "LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm" },
{ "deaf5e62e2132996ebe759a438195f95"  , "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" },
{ "6b04366e28facddd808b9ea149745309"  , "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" },
{ "f1b63522a4d6ae89eb6bc728cebc14ff"  , "libido1.0-vol.acr" },
{ "b90b87245eddfcb53b26e61cfaa40fcc"  , "LIBIDO-16-ACR_NEMA-Volume.dcm" },
{ "454accc7d6688c0478461f575461c607"  , "LIBIDO-24-ACR_NEMA-Rectangle.dcm" },
{ "3d8ee51f870495bf22b7a51ba0661f90"  , "LIBIDO-8-ACR_NEMA-Lena_128_128.acr" },
{ "9c8b67c4205880f78347b751268af0fa"  , "LJPEG_BuginGDCM12.dcm" },
{ "335428b3dde0d390dbd8bb49f32c673c"  , "MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm" },
{ "abd7b36703f82e3b491e4b3b95cc1c43"  , "MAROTECH_CT_JP2Lossy.dcm" },
{ "63f178c91a573f13e28aabe8b7eaf1bd"  , "MR-Brucker-CineTagging-NonSquarePixels.dcm" },
{ "746a17465b70119264762ead8f1a6763"  , "MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm" },
{ "c87dd2eac7b55cb3eade01eaa373b4e3"  , "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm" },
{ "ca8cd1b0a7c64139919deba04dbddfc9"  , "MR-MONO2-12-an2.acr" },
{ "cb32c2e0c3f9c52939440d39f9fe7b4f"  , "MR-MONO2-12-angio-an1.acr" },
{ "fb0565a0a591ed83106e90d5b76e0cfe"  , "MR-MONO2-12-shoulder.dcm" },
{ "97d9f3dcc35e54478522ead748e24956"  , "MR-MONO2-16-head.dcm" },
{ "fc72513cfea2caf6035dd8910c53bb9a"  , "MR-MONO2-8-16x-heart.dcm" },
{ "1a2e4f0aa20448fdd7783ff938bf99e6"  , "MR_Philips-Intera_BreaksNOSHADOW.dcm" },
{ "aa07de9c01765602fe722e9ef2d8b92a"  , "MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm" },
{ "a8091f92ae895c2ef70143487e29b7d3"  , "MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm" },
{ "5d893aee8147f12b975cde73abdb5d84"  , "MR_Philips_Intera_PrivateSequenceImplicitVR.dcm" },
{ "3bbffc4c87f4f5554fafad5f8a002552"  , "MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm" },
{ "32f899e8f1506bc3fa155da22e9c8813"  , "MR-SIEMENS-DICOM-WithOverlays.dcm" },
{ "db7370f6d18ce7a9c8ab05179eb82cc6"  , "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" },
{ "15ef679db745d4a3a59cab0456a82ace"  , "MR_SIEMENS_forceLoad29-1010_29-1020.dcm" },
{ "be2ce86e93fe5a0cc33c2351b6a4ac66"  , "MR_Spectroscopy_SIEMENS_OF.dcm" },
{ "473dd7ad8ec7c50dbd20470b525eb859"  , "NM-MONO2-16-13x-heart.dcm" },
{ "cc1dd93cd2e2fc19815b015663ea8e66"  , "OT-MONO2-8-a7.dcm" },
{ "61dfde4beae2ecd7fd4acc9cab412daa"  , "OT-PAL-8-face.dcm" },
{ "d97af0d265cee784f3fd6391f17cf8fd"  , "PET-cardio-Multiframe-Papyrus.dcm" },
{ "826226301791aaa2e1dfacb9f56775ae"  , "PHILIPS_Brilliance_ExtraBytesInOverlay.dcm" },
{ "fe35f6ff5e85392143c7216a9e4bc92d"  , "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" },
{ "d6ea0b68da84829a1b4cbd07d7cf6ef5"  , "PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm" },
{ "cc8bbb35c017acd32ba29d20df12ac8b"  , "PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm" },
{ "f80468faedc918845fc31accc247125f"  , "PHILIPS_Intera-16-MONO2-Uncompress.dcm" },
{ "43469b245a5fbbea69db6ed9507a86e4"  , "PICKER-16-MONO2-Nested_icon.dcm" },
{ "2cd10ed50b409549a6a25c4feaa5a989"  , "PICKER-16-MONO2-No_DicomV3_Preamble.dcm" },
{ "9d4f1c087ababf655297bf2129c01911"  , "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" },
{ "8e902a2371c37b9e6f636c3c53ef1f5c"  , "RadBWLossLess.dcm" },
{ "a0e664672011dab175f593bf61026ffc"  , "rle16loo.dcm" },
{ "f35d850d15021800175c585f856a8f54"  , "rle16sti.dcm" },
{ "148b28cb580e6f56d9adce46a985c692"  , "SIEMENS-12-Jpeg_Process_2_4-Lossy-a.dcm" },
{ "90336859a2e275900931d54c41fe033a"  , "SIEMENS_CSA2.dcm" },
{ "4b1498f0edd79cc0f5273f22a4e03615"  , "SIEMENS_GBS_III-16-ACR_NEMA_1.acr" },
{ "c54eb37190f783c79504554990761efd"  , "SIEMENS_GBS_III-16-ACR_NEMA_1-ULis2Bytes.dcm" },
{ "bf49f9fdab81e97482b8fe6f579bc3f7"  , "SIEMENS_ImageLocationUN.dcm" },
{ "765d1cae48ebac57f675193734f6477c"  , "SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm" },
{ "6884e1143f2edf6188643e896e796463"  , "SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm" },
{ "340b9006e47d48bd6d5abe5af628026b"  , "SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm" },
{ "59cc16618e978026cff79e61e17174ec"  , "SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm" },
{ "fde3a66d449ae3788b4c431fde7a7a50"  , "SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm" },
{ "6facfeecd3c531b3a536064aa046fa9e"  , "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm" },
{ "ba687ec0b2e49f5eedd3e22573ba2094"  , "SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm" },
{ "7e00f1e50bd0bedc5db64dff0dea3848"  , "SIEMENS_MAGNETOM-12-MONO2-VRUN.dcm" },
{ "c80b1be0d50619955e54d81bca451a05"  , "SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm" },
{ "544a1fde6df0817ef03ad56f7ff539a8"  , "SIEMENS-MR-RGB-16Bits.dcm" },
{ "33f4765c31a5d2f61f4e3fa4571e2f88"  , "SIEMENS_SOMATOM-12-ACR_NEMA-ZeroLengthUs.acr" },
{ "08443d1e07d98554a695c482ecba2014"  , "SIEMENS_Sonata-12-MONO2-SQ.dcm" },
{ "f49aeced38187d9d8502cf79fb690e0d"  , "SIEMENS_Sonata-16-MONO2-Value_Multiplicity.dcm" },
{ "fd22f759b7dd4b8cccd64f5f5096887b"  , "SignedShortLosslessBug.dcm" },
{ "c8c3e1d395aa50795ca831e12051d9d0"  , "simpleImageWithIcon.dcm" },
{ "c634e9e559b61ea65d222a1d5c5a9d5e"  , "test.acr" },
{ "fc733f08120936e25cb11af6df6f65bf"  , "TG18-CH-2k-01.dcm" },
{ "47a35db3d33ab798e7b41449e22d1120"  , "THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm" },
{ "d85505c6a7be835b482175fbed85ce98"  , "TheralysGDCM120Bug.dcm" },
{ "db53b838d4ed4aca22999d6c70558305"  , "TOSHIBA_MRT150-16-MONO2-ACR_NEMA_2.dcm" },
{ "b3bc682102873b761491839df5686adf"  , "undefined_length_un_vr.dcm" },
{ "98473e5af6a5694ae08d63780bcea96c"  , "US-GE-4AICL142.dcm" },
{ "1b12a27ca6075d55000d1d170949b789"  , "US-IRAD-NoPreambleStartWith0003.dcm" },
{ "44f97431146a33e320ae08fb7dc09a59"  , "US-IRAD-NoPreambleStartWith0005.dcm" },
{ "1006e4c95b63539608adf62060d7fc46"  , "US-MONO2-8-8x-execho.dcm" },
{ "83574afcb319337222472db44f91cbd8"  , "US-PAL-8-10x-echo.dcm" },
{ "0b3ae440e8dbb144205074a664228c1e"  , "US-RGB-8-epicard.dcm" },
{ "0ae1d92cc14706add796368622deaa26"  , "US-RGB-8-esopecho.dcm" },
{ "d23db052b02b2a2c510104486cdf40b2"  , "XA-MONO2-8-12x-catheter.dcm" },
{ "231bb6c25d482422745bb839d6f8610a"  , "PHILIPS_GDCM12xBug2.dcm" },
{ "e2558e4e01d937bfac33b7e0fb07b7b5"  , "PHILIPS_GDCM12xBug.dcm" },
{ "434c7aa1172ce3af0c306d9a2cb28c17"  , "AMIInvalidPrivateDefinedLengthSQasUN.dcm" },
{ "346ba438ade47d75c41e637858a419f8"  , "OsirixFake16BitsStoredFakeSpacing.dcm" },
{ "efe1ff2da3fc0bfcc5df834fa390d5cf"  , "MR16BitsAllocated_8BitsStored.dcm" },
{ "7900d059078278ad0387f7c4aaf2027d"  , "JPEGDefinedLengthSequenceOfFragments.dcm" },
{ "dacb240b2fc701416d80193ad18baad5"  , "IM-0001-0066.CommandTag00.dcm" },
{ "7f2f84d3adef913bb5531f583eceb004"  , "UnexpectedSequenceDelimiterInFixedLengthSequence.dcm" },
{ "f4bf27280aad0ff38d1fe3871a0a8afb"  , "GDCMJ2K_TextGBR.dcm" },
{ "f445a6dc674e2c12c14bf3583c6c3e6f"  , "NM_Kakadu44_SOTmarkerincons.dcm" },
{ "7c0c4eb0c73b4dc5b3a4d961653fc4e2"  , "PhilipsInteraSeqTermInvLen.dcm" },
{ "64528209839b0369c2da530249f4ca58"  , "TOSHIBA_J2K_SIZ1_PixRep0.dcm" },
{ "2498ca4aaf62991c8a1f629a804bfa44"  , "TOSHIBA_J2K_OpenJPEGv2Regression.dcm" },
{ "e4d559b6db04679b54bea40c763b09e4"  , "TOSHIBA_J2K_SIZ0_PixRep1.dcm" },
{ "58ab110be40303952e05d50e64823192"  , "NM-PAL-16-PixRep1.dcm" },
{ "deb7e4ee592efca5d475aaa6fab06459"  , "MEDILABInvalidCP246_EVRLESQasUN.dcm" },
{ "52f727de4f831ff2bae850fdd8b1413a"  , "JPEGInvalidSecondFrag.dcm" },
{ "027faefc8031768dad1afc100f7aee27"  , "SC16BitsAllocated_8BitsStoredJ2K.dcm" },
{ "f7c4fbb93b0347101e21e36f223b4d46"  , "SC16BitsAllocated_8BitsStoredJPEG.dcm" },

{ NULL, NULL}
};

int TestMD5Func(const char* filename, const char *md5ref, bool verbose = false)
{
  if( !filename || !md5ref) return 1;

  if( verbose )
    std::cout << "TestRead: " << filename << std::endl;
  const char *dataroot = gdcm::Testing::GetDataRoot();
  std::string path = dataroot;
  path += "/";
  path += filename;
  path = filename;
  char md5[2*16+1] = {};
  bool b = gdcm::MD5::ComputeFile( path.c_str(), md5);
  if( !b )
    {
    std::cerr << "Fail ComputeFile: " << path << std::endl;
    return 1;
    }
  if( strcmp( md5, md5ref) != 0 )
    {
    std::cout << "Problem with: " << path << std::endl;
    std::cout << "Ref: " << md5ref << " vs " << md5 << std::endl;
    // Let's remove this buggy file:
    //std::cout << "Removing: " << path << std::endl;
    //gdcm::System::RemoveFile(path.c_str());
    return 1;
    }
  return 0;
}

static const char *GetMD5Sum(const char *filename)
{
  typedef const char * const (*md5pair)[2];
  const char *md5filename;
  md5pair md5filenames = gdcmMD5SumFiles;
  int i = 0;
  while( ( md5filename = md5filenames[i][1] ) )
    {
    gdcm::Filename fn( filename );
    if( strcmp( md5filename, fn.GetName() ) == 0 )
      {
      return md5filenames[i][0];
      }
    ++i;
    }
  std::cerr << "Missing Md5 for: " << filename << std::endl;
  return 0;
}

int TestMD5(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    const char *md5 = GetMD5Sum( filename );
    return TestMD5Func(filename, md5, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    const char *md5 = GetMD5Sum( filename );
    r += TestMD5Func( filename, md5 );
    ++i;
    }

  return r;
}
